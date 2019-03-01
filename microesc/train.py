

import os.path
import math
import sys
import uuid
import json
import functools
import datetime

import pandas
import numpy
import keras
import librosa

from . import features, urbansound8k, common
from .models import sbcnn


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


def train_model(out_dir, fold, builder,
                loader, val_loader,
                frame_samples, window_frames,
                train_samples=12000, val_samples=3000,
                batch_size=200, epochs=50, seed=1, learning_rate=3e-4):
    """Train a single model"""    

    model = builder()
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=learning_rate),
                  metrics=['accuracy'])


    model_path = os.path.join(out_dir, 'e{epoch:02d}-v{val_loss:.2f}.t{loss:.2f}.model.hdf5')
    checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_acc', mode='max',
                                         period=1, verbose=1, save_best_only=False)
    callbacks_list = [checkpoint]

    train, val = fold
    train_gen = dataframe_generator(train, train.classID, loader=loader, batchsize=batch_size)
    val_gen = dataframe_generator(val, val.classID, loader=val_loader, batchsize=batch_size)

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
        print('v', k, v, args.get(k))
        train_settings[k] = v
    return train_settings



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

    folder = os.path.join(feature_dir, features.settings_id(feature_settings))

    preloaded = {}
    for _, sample in data[data.fold != 10].iterrows():
        for aug in range(-1, exsettings['augmentations']):
            k = (sample.fold, sample.slice_file_name, aug)
            if aug == -1:
                aug = None
            path = features.feature_path(sample, out_folder=folder, augmentation=aug)
            #print(k, path)
            preloaded[k] = numpy.load(path)['arr_0']

    print('preloaded', len(preloaded.keys()))


    def load_file(sample, aug):
        #path = features.feature_path(sample, out_folder=folder, augmentation=aug)
        #mels = numpy.load(path)['arr_0']
        if aug is None:
            aug = -1
        k = (sample.fold, sample.slice_file_name, aug)
        mels = preloaded[k]
        return mels

    def load(sample, validation):
        augment = not validation and train_settings['augment'] != 0
        d = features.load_sample(sample, feature_settings, loader=load_file,
                        window_frames=model_settings['frames'],
                        augment=augment)
        return d

    def build_model():
        m = sbcnn.build_model(bands=feature_settings['n_mels'], channels=1,
                    frames=model_settings['frames'],
                    pool=model_settings['pool'],
                    kernel=model_settings['kernel'],
                    )
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
                      frame_samples=feature_settings['hop_length'],
                      window_frames=model_settings['frames'],
                      epochs=train_settings['epochs'],
                      train_samples=train_settings['train_samples'],
                      val_samples=train_settings['val_samples'],
                      batch_size=train_settings['batch'])



if __name__ == '__main__':
    main()
