

import urllib.request
import os.path
import zipfile
import math
import sys
import uuid
import json

import pandas
import numpy
import keras
import librosa

import features
import preprocess
import urbansound8k
import sbcnn

default_base_url = 'https://storage.googleapis.com/urbansound8k'


def features_url(settings, base=default_base_url):
    id = preprocess.settings_id(settings)
    ext = '.zip'  
    return "{}/{}{}".format(base, id, ext)


def maybe_download_features(settings, workdir):

    feature_dir = os.path.join(workdir, preprocess.settings_id(settings))
    feature_zip = feature_dir + '.zip'
    feature_url = features_url(settings)

    last_progress = None
    def download_progress(count, blocksize, totalsize):
        nonlocal last_progress

        p = int(count * blocksize * 100 / totalsize)
        if p != last_progress:
            print('\r{}%'.format(p), end='\r')
            last_progress = p

    if not os.path.exists(feature_dir):
        
        if not os.path.exists(feature_zip):
            u = feature_url
            print('Downloading...', u)
            urllib.request.urlretrieve(u, feature_zip, reporthook=download_progress)

        # Note: .zip file is kept around
        with zipfile.ZipFile(feature_zip, "r") as archive:
            archive.extractall(workdir)

    return feature_dir


def load_sample(sample, settings, feature_dir, window_frames,
                start_time=None, augment=None):
    n_mels = settings['n_mels']
    sample_rate = settings['samplerate']
    hop_length = settings['hop_length']

    aug = None
    if augment is not None and settings['augmentations'] > 0:
        aug = numpy.random.randint(-1, settings['augmentations'])
        if aug == -1:
            aug = None

    # Load precomputed features
    folder = os.path.join(feature_dir, preprocess.settings_id(settings))
    path = preprocess.feature_path(sample, out_folder=folder, augmentation=aug)
    mels = numpy.load(path)['arr_0']
    assert mels.shape[0] == n_mels, mels.shape
    
    if start_time is None:
        # Sample a window in time randomly
        min_start = max(0, mels.shape[1]-window_frames)
        if min_start == 0:
            start = 0
        else:
            start = numpy.random.randint(0, min_start)
    else:
        start = int(start_time * (sample_rate / hop_length))

    end = start + window_frames
    mels = mels[:, start:end]

    # Normalize the window
    if mels.shape[1] > 0:
        mels = librosa.core.power_to_db(mels, top_db=80, ref=numpy.max)

    # Pad to standard size
    if window_frames is None:
        padded = mels
    else:
        padded = numpy.full((n_mels, window_frames), 0)    
        inp = mels[:, 0:min(window_frames, mels.shape[1])]
        padded[:, 0:inp.shape[1]] = inp

    # add channel dimension
    data = numpy.expand_dims(padded, -1)
    return data

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



def sample_windows(length, frame_samples, window_frames, overlap=0.5):
    """Split @samples into a number of windows of samples
    with length @frame_samples * @window_frames
    """

    ws = frame_samples * window_frames
    start = 0
    while start < length:
        end = min(start + ws, length)
        yield start, end
        start += (ws * (1-overlap))


def train_model(out_dir, fold, builder, loader,
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
    val_gen = dataframe_generator(val, val.classID, loader=loader, batchsize=batch_size)

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

default_model_settings = dict(
    model='sbcnn',
    kernel='3x3',
    pool='3x3',
    frames=72,
)

def parse(args):

    import argparse

    parser = argparse.ArgumentParser(description='Train a model')

    a = parser.add_argument

    a('--features', dest='features_dir', default='./features',
        help='%(default)s')
    a('--out', dest='out_dir', default='./out',
        help='%(default)s')

    a('--name', type=str, default=str(uuid.uuid4()),
        help='default: Autogenerated UUID')
    a('--fold', type=int, default=0,
        help='')

    # Expose feature settings
    for k, v in features.default_settings.items():
        t = int if k != 'feature' else str
        a('--{}'.format(k), type=t, default=v, help='default: %(default)s')

    # Expose training settings
    for k, v in default_training_settings.items():
        t = int
        a('--{}'.format(k), type=t, default=v, help='default: %(default)s')

    # Expose model settings
    for k, v in default_model_settings.items():
        t = str if k != 'frames' else int
        a('--{}'.format(k), type=t, default=v, help='default: %(default)s')

    parsed = parser.parse_args(args)

    return parsed

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


# TODO: set up logging module to write to a file in output dir, synced periodically
def main():

    args = parse(sys.argv[1:])
    args = dict(args.__dict__)

    # experiment settings
    feature_dir = args['features_dir']
    name = args['name']
    output_dir = os.path.join(args['out_dir'], name)
    fold = args['fold']

    os.makedirs(output_dir)
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    # model settings
    model_settings = {}
    for k in default_model_settings.keys():
        model_settings[k] = args[k]

    # feature settings
    feature_settings = {}
    for k in features.default_settings.keys():
        feature_settings[k] = args[k]

    # training settings
    train_settings = {}
    for k in default_training_settings.keys():
        train_settings[k] = args[k]

    maybe_download_features(feature_settings, feature_dir)

    # TODO: allow specifying dataset on commandline
    dataset_path = 'data/UrbanSound8K/'
    if not os.path.exists(os.path.join(dataset_path, 'metadata')):
        os.makedirs(os.path.join(dataset_path, 'metadata'))
    urbansound8k.default_path = dataset_path
    data = urbansound8k.load_dataset()
    folds, test = urbansound8k.folds(data)
    assert len(folds) == 9

    def load(sample):
        d = load_sample(sample, feature_settings, feature_dir=feature_dir,
                        window_frames=model_settings['frames'],
                        augment=train_settings['augment'] != 0)
        return d

    def build_model():
        m = sbcnn.build_model(bands=feature_settings['n_mels'], channels=1,
                    frames=model_settings['frames'],
                    pool=parse_dimensions(model_settings['pool']),
                    kernel=parse_dimensions(model_settings['kernel']),
                    )
        return m

    settings = {
        'model': model_settings,
        'features': feature_settings,
        'training': train_settings,
    }

    print('Training model', name)
    print('Settings', json.dumps(settings))

    h = train_model(output_dir, folds[fold],
                      builder=build_model, loader=load,
                      frame_samples=feature_settings['hop_length'],
                      window_frames=model_settings['frames'],
                      epochs=train_settings['epochs'],
                      train_samples=train_settings['train_samples'],
                      val_samples=train_settings['val_samples'],
                      batch_size=train_settings['batch'])



if __name__ == '__main__':
    main()
