

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

from . import features, urbansound8k, common
from .models import sbcnn, dilated, mobilenet, effnet


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




def train_model(out_dir, fold, builder,
                loader, val_loader, settings, seed=1):
    """Train a single model"""    

    frame_samples = settings['hop_length']
    train_samples = settings['train_samples']
    window_frames = settings['frames']
    val_samples = settings['val_samples']
    epochs = settings['epochs']
    batch_size = settings['batch']
    #learning_rate = settings['learning_rate']   

    train, val = fold

    def top3(y_true, y_pred):
        return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

    model = builder()
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
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


default_training_settings = dict(
    epochs=50,
    batch=50,
    train_samples=36000,
    val_samples=3000,
    augment=0,
)



def parse(args):

    import argparse

    parser = argparse.ArgumentParser(description='Train a model')
    a = parser.add_argument

    common.add_arguments(parser)

    a('--fold', type=int, default=0,
        help='')

    a('--name', type=str, default='',
        help='')

    parsed = parser.parse_args(args)

    return parsed

default_model_settings = dict(
    model='sbcnn',
    kernel='3x3',
    pool='3x3',
    frames=72,
)

def parse_dimensions(s):
    pieces = s.split('x')
    return tuple( int(d) for d in pieces )    

def test_parse_dimensions():
    valid_examples = [
        ('3x3', (3,3)),
        ('4x2', (4,2))
    ]
    for inp, expect in valid_examples:
        out = parse_dimensions(inp)
        assert out == expect, (out, '!=', expect) 

test_parse_dimensions()

def load_model_settings(args):
    model_settings = {}
    for k in default_model_settings.keys():
        v = args.get(k, default_model_settings[k]) 
        if k in ('pool', 'kernel'):
            v = parse_dimensions(v)

        model_settings[k] = v
    return model_settings


def settings(args):
    train_settings = {}
    for k in default_training_settings.keys():
        v = args.get(k, default_training_settings[k])
        train_settings[k] = v
    return train_settings


def ldcnn(settings):
    m = dilated.ldcnn_nodelta(frames=settings['frames'], bands=settings['n_mels'], 
                            filters=80, L=57, W=6, fully_connected=5000)
    return m

def sb_cnn(settings):
    m = sbcnn.build_model(bands=settings['n_mels'], channels=1,
                    frames=settings['frames'],
                    pool=parse_dimensions(settings['pool']),
                    kernel=parse_dimensions(settings['kernel']),
                    )
    return m

def mobilenets(settings):
    m = mobilenet.build_model(bands=settings['n_mels'], channels=1,
                    frames=settings['frames'],
                    alpha=0.50,
                    )
    return m

def eff_net(settings):
    m = effnet.build_model(bands=settings['n_mels'], frames=settings['frames'])
    return m

def main():

    args = parse(sys.argv[1:])
    args = dict(args.__dict__)

    # experiment settings
    feature_dir = args['features_dir']
    name = args['experiment']

    fold = args['fold']

    if args['name']:
        name = args['name']
    else:
        t = datetime.datetime.now().strftime('%Y%m%d-%H%M') 
        u = str(uuid.uuid4())[0:4]
        name = "-".join([name, t, u, 'fold{}'.format(fold)])

    output_dir = os.path.join(args['models_dir'], name)

    common.ensure_directories(output_dir, feature_dir)

    # model settings
    exsettings = common.load_experiment(args['experiments_dir'], args['experiment'])
    feature_settings = features.settings(exsettings)
    train_settings = settings(exsettings)
    model_settings = load_model_settings(exsettings)

    features.maybe_download(feature_settings, feature_dir)

    # TODO: allow specifying dataset on commandline

    data = urbansound8k.load_dataset()
    folds, test = urbansound8k.folds(data)
    assert len(folds) == 9

    def load(sample, validation):
        augment = not validation and train_settings['augment'] != 0
        d = features.load_sample(sample, feature_settings, feature_dir=feature_dir,
                        window_frames=model_settings['frames'],
                        augment=augment)
        return d

    def build_model():
        m = sb_cnn(exsettings)
        #m = ldcnn(exsettings)

        m.summary()

        return m

    all_settings = {
        'model': model_settings,
        'features': feature_settings,
        'training': train_settings,
    }

    print('Training model', name)
    print('Settings', json.dumps(all_settings))

    h = train_model(output_dir, folds[fold],
                      builder=build_model,
                      loader=functools.partial(load, validation=False),
                      val_loader=functools.partial(load, validation=True),
                      settings=exsettings)



if __name__ == '__main__':
    main()
