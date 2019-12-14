

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


def dataframe_generator(X, Y, loader, batchsize=10, n_classes=10):
    """
    Keras generator for lazy-loading data based on a pandas.DataFrame
    
    X: data column(s)
    Y: target column
    loader: function will be passed batches of X to load actual training data
    """
        
    assert len(X) == len(Y), 'X and Y must be equal length'

    while True:
        idx = numpy.random.choice(len(X), size=batchsize, replace=False)
        rows = X.iloc[idx, :].iterrows()
        data = [ loader(d) for _, d in rows ]
        y = Y.iloc[idx]
        y = keras.utils.to_categorical(y, num_classes=n_classes)
        batch = (numpy.array(data), numpy.array(y))
        yield batch


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
    
        more = self.score() # uses current model
        for k, v in more.items():
            logs[k] = v

        self.write_entry(epoch, logs)




def train_model(out_dir, train, val, model,
                loader, val_loader, settings, seed=1):
    """Train a single model"""    

    frame_samples = settings['hop_length']
    train_samples = settings['train_samples']
    window_frames = settings['frames']
    val_samples = settings['val_samples']
    epochs = settings['epochs']
    batch_size = settings['batch']
    learning_rate = settings.get('learning_rate', 0.01)

    assert len(train) > len(val) * 5, 'training data should be much larger than validation'

    def top3(y_true, y_pred):
        return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

    optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=settings['nesterov_momentum'], nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model_path = os.path.join(out_dir, 'e{epoch:02d}-v{val_loss:.2f}.t{loss:.2f}.model.hdf5')
    checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_acc', mode='max',
                                         period=1, verbose=1, save_best_only=False)

    def voted_score():
        y_pred = features.predict_voted(settings, model, val,
                        loader=val_loader, method=settings['voting'], overlap=settings['voting_overlap'])
        class_pred = numpy.argmax(y_pred, axis=1)
        acc = sklearn.metrics.accuracy_score(val.classID, class_pred)
        d = {
            'voted_val_acc': acc,
        }
        for k, v in d.items():
            print("{}: {:.4f}".format(k, v))
        return d
    log_path = os.path.join(out_dir, 'train.csv')
    log = LogCallback(log_path, voted_score)


    train_gen = dataframe_generator(train, train.classID, loader=loader, batchsize=batch_size)
    val_gen = dataframe_generator(val, val.classID, loader=val_loader, batchsize=batch_size)

    callbacks_list = [checkpoint, log]
    hist = model.fit_generator(train_gen, validation_data=val_gen,
                        steps_per_epoch=math.ceil(train_samples/batch_size),
                        validation_steps=math.ceil(val_samples/batch_size),
                        callbacks=callbacks_list,
                        epochs=epochs, verbose=1)

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


def build_melspec_model(settings, classifier):

    sr = settings['samplerate']
    shape = (1, sr * 0.72  )
    
    from keras.layers import Input
    from keras import Model
    from kapre.time_frequency import Melspectrogram

    melspec = Melspectrogram(sr=sr, n_mels=settings['n_mels'], 
              n_dft=settings['n_fft'], n_hop=settings['hop_length'], 
              return_decibel_melgram=True,
              trainable_kernel=False, name='melgram')

    input = Input(shape=shape)
    x = melspec(input)
    x = classifier(x)
    model = Model(input,x)
    
    return model


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
    train_data, val_data = load_training_data(data, fold)

    def load(sample, validation):
        d = features.load_audio(sample, exsettings)
        return d

    def build_model():
        m = models.build(exsettings)
        m = build_melspec_model(exsettings, m)
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

    h = train_model(output_dir, train_data, val_data,
                      model=m,
                      loader=functools.partial(load, validation=False),
                      val_loader=functools.partial(load, validation=True),
                      settings=exsettings)



if __name__ == '__main__':
    main()
