

import os.path
import math
import sys
import uuid
import json
import functools
import datetime
import csv

import pandas
import numpy
import keras
import librosa
import sklearn.metrics

from . import features, urbansound8k, common, models, stats
from . import settings as Settings



class Generator(keras.utils.Sequence):

    def __init__(self, x_set, y_set, feature_dir, settings, n_classes=10, augment=False):
        self.x, self.y = x_set, y_set
        self.batch_size = settings['batch']
        self.n_classes = n_classes
        self.augment = augment
        self.n_augmentations = settings['augmentations'] if self.augment else 1
        self.feature_dir = feature_dir
        self.feature_settings = features.settings(settings)
        self.settings = settings

    def _load(self, sample):
        def load_chunk(chunk):
            return features.load_sample(chunk,
                            self.feature_settings,
                            feature_dir=self.feature_dir,
                            start_time=chunk.start,
                            window_frames=self.settings['frames'],
                            augment=self.augment)

        # FIXME: use time-shifting augmentation, randomize starts
        wins = features.load_windows(sample,
            self.settings,
            loader=load_chunk,
            overlap=self.settings['voting_overlap'],
            start=0)

        d = numpy.stack(wins)
        s = (6, d.shape[1], d.shape[2], d.shape[3])
        windows = numpy.zeros(shape=s)
        windows[:d.shape[0], :, :, :] = d

        #print('lo', len(wins), d.shape, windows.shape)
        return windows

    def __len__(self):
        np = numpy
        return int(np.ceil(len(self.x) / float(self.batch_size))) * self.n_augmentations
    
    def __getitem__(self, idx):
        # FIXME: take augmentation into account
        from_idx = idx * self.batch_size
        to_idx = (idx + 1) * self.batch_size
        #print('b', from_idx, to_idx)

        X = self.x.iloc[from_idx:to_idx]
        y = self.y.iloc[from_idx:to_idx]

        #print('xx', X.shape, y.shape)

        data = [ self._load(d) for _, d in X.iterrows() ]
        y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        batch = (numpy.stack(data), numpy.array(y))
        #print('x', batch[0].shape)
        return batch


class LogCallback(keras.callbacks.Callback):
    def __init__(self, log_path, score_epoch):
        super().__init__()
    
        self.log_path = log_path
        self.score = score_epoch   

        self._log_file = None
        self._csv_writer = None

    def __del__(self):
        if self._log_file:
            self._log_file.close()
       

    def write_entry(self, epoch, data):
        data = data.copy()

        if not self._csv_writer:
            # create writer when we know what fields
            self._log_file = open(self.log_path, 'w')
            fields = ['epoch'] + sorted(data.keys())
            self._csv_writer = csv.DictWriter(self._log_file, fields)
            self._csv_writer.writeheader()
        
        data['epoch'] = epoch
        self._csv_writer.writerow(data)
        self._log_file.flush() # ensure data hits disk

    def on_epoch_end(self, epoch, logs):
        logs = logs.copy()
    
        more = self.score(epoch, logs) # uses current model
        for k, v in more.items():
            logs[k] = v

        self.write_entry(epoch, logs)


def build_multi_instance(base, windows=6, bands=32, frames=72, channels=1):
    from keras import Model
    from keras.layers import Input, TimeDistributed, GlobalAveragePooling1D
    
    input_shape = (windows, bands, frames, channels)
    
    input = Input(shape=input_shape)
    x = input # BatchNormalization()(input)
    x = TimeDistributed(base)(x)
    x = GlobalAveragePooling1D()(x)
    model = Model(input,x)
    return model


def train_model(out_dir, fold, builder,
                feature_dir, settings, name):
    """Train a single model"""    

    frame_samples = settings['hop_length']
    window_frames = settings['frames']
    epochs = settings['epochs']
    batch_size = settings['batch']
    learning_rate = settings.get('learning_rate', 0.01)

    def generator(data, augment):
        return Generator(data, data.classID, feature_dir=feature_dir, settings=settings, augment=augment)

    model = builder()
    model = build_multi_instance(model, bands=settings['n_mels'], frames=window_frames, windows=6)
    model.summary()

    optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=settings['nesterov_momentum'], nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model_path = os.path.join(out_dir, 'e{epoch:02d}-v{val_loss:.2f}.t{loss:.2f}.model.hdf5')
    checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_acc', mode='max',
                                         period=1, verbose=1, save_best_only=False)

    tensorboard = keras.callbacks.TensorBoard(log_dir=f'./logs/{name}', histogram_freq=0,
                          write_graph=True, write_images=False)

    def voted_score(epoch, logs):
        d = {
            'voted_val_acc': logs['val_acc'], # XXX: legacy compat
        }
        return d

    log_path = os.path.join(out_dir, 'train.csv')
    log = LogCallback(log_path, voted_score)

    train_gen = generator(fold[0], augment=False) # FIXME: enable augmentation
    val_gen = generator(fold[1], augment=False)

    callbacks_list = [checkpoint, log, tensorboard]
    hist = model.fit_generator(train_gen,
                        validation_data=val_gen,
                        callbacks=callbacks_list,
                        epochs=epochs, verbose=1, workers=1)

    df = history_dataframe(hist)
    history_path = os.path.join(out_dir, 'history.csv')
    df.to_csv(history_path)

    return hist

def history_dataframe(h):
    data = {}
    data['epoch'] = h.epoch
    for k, v in h.history.items():
        data[k] = v
    df = pandas.DataFrame(data)
    return df




def parse(args):

    import argparse

    parser = argparse.ArgumentParser(description='Train a model')
    a = parser.add_argument

    common.add_arguments(parser)
    Settings.add_arguments(parser)

    a('--fold', type=int, default=1,
        help='')
    a('--skip_model_check', action='store_true', default=False,
        help='Skip checking whether model fits on STM32 device')
    a('--load', default='',
        help='Load a already trained model')

    a('--name', type=str, default='',
        help='')

    parsed = parser.parse_args(args)

    return parsed



def setup_keras():
    import tensorflow as tf
    from keras.backend import tensorflow_backend as B

    # allow_growth is needed to avoid CUDNN_STATUS_INTERNAL_ERROR on some convolutional layers
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=session_config)
    B.set_session(sess)

def load_training_data(data, fold):
    assert fold >= 1 # should be 1 indexed
    folds = urbansound8k.folds(data)
    assert len(folds) == 10
    train_data = folds[fold-1][0]
    val_data = folds[fold-1][1]
    test_folds = folds[fold-1][2].fold.unique()
    assert len(test_folds) == 1
    assert test_folds[0] == fold, (test_folds[0], '!=', fold) # by convention, test fold is fold number
    return train_data, val_data

def main():
    setup_keras()

    args = parse(sys.argv[1:])
    args = dict(args.__dict__)

    # experiment settings
    feature_dir = args['features_dir']
    fold = args['fold']

    if args['name']:
        name = args['name']
    else:
        t = datetime.datetime.now().strftime('%Y%m%d-%H%M') 
        u = str(uuid.uuid4())[0:4]
        name = "-".join(['unknown', t, u, 'fold{}'.format(fold)])

    output_dir = os.path.join(args['models_dir'], name)

    common.ensure_directories(output_dir, feature_dir)

    # model settings
    exsettings = common.load_settings_path(args['settings_path'])
    for k, v in args.items():
        if v is not None:
            exsettings[k] = v
    exsettings = Settings.load_settings(exsettings)

    feature_settings = features.settings(exsettings)
    train_settings = { k: v for k, v in exsettings.items() if k in Settings.default_training_settings }
    model_settings = { k: v for k, v in exsettings.items() if k in Settings.default_model_settings }

    features.maybe_download(feature_settings, feature_dir)

    data = urbansound8k.load_dataset()
    fold_data = load_training_data(data, fold)

    def build_model():
        m = models.build(exsettings)
        return m

    load_model = args['load']
    if load_model:
        print('Loading existing model', load_model)
        m = keras.models.load_model(load_model)
    else:
        m = build_model()
    m.summary()

    if args['skip_model_check']:
        print('WARNING: model constraint check skipped')
    else:
        print('Checking model contraints')
        ss, ll = stats.check_model_constraints(m)
        print('Stats', ss)
    
    print('Training model', name)
    print('Settings', json.dumps(exsettings))

   
    h = train_model(output_dir, fold_data,
                      builder=build_model,
                      feature_dir = feature_dir,
                      settings=exsettings, name=name)



if __name__ == '__main__':
    main()
