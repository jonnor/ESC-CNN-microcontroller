

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

def load_sample(sample, settings, feature_dir, window_frames=72):
    n_mels = settings['n_mels']
    sample_rate = settings['samplerate']
    hop_length = settings['hop_length']
    
    # Load precomputed features
    aug = None
    if hasattr(sample, 'augmentation'):
        aug = sample.augmentation
    folder = os.path.join(feature_dir, preprocess.settings_id(settings))
    path = preprocess.feature_path(sample, out_folder=folder, augmentation=aug)
    mels = numpy.load(path)['arr_0']
    assert mels.shape[0] == n_mels, mels.shape
    
    # Cut out the relevant part
    start = int(sample.start * (sample_rate / hop_length))
    end = int(sample.end * (sample_rate / hop_length))
    d = (sample.end-sample.start)
    mels = mels[:, start:end]
    #assert mels.shape[1] > 0, (sample)

    if mels.shape[1] > 0:
        mels = librosa.core.power_to_db(mels, top_db=80, ref=numpy.max)
    
    # zero-pad window to standard length
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
    assert len(X) % batchsize == 0, 'input length must be divisible by @batchsize'
        
    sample_idx = 0
    while True:
        batch_data = []
        batch_labels = []

        if sample_idx >= len(X):
            sample_idx = 0
        
        while len(batch_data) < batchsize:
            data = loader(X.iloc[sample_idx])  
            y = Y.iloc[sample_idx]
            y = keras.utils.to_categorical(y, num_classes=n_classes)
            batch_data.append(data)
            batch_labels.append(y)
            sample_idx += 1
            
        batch = (numpy.stack(batch_data), numpy.stack(batch_labels))
        yield batch

def sample_windows(length, frame_samples, window_frames, overlap=0.5):
    """Split @samples into a number of windows of samples
    with length @frame_samples * @window_frames
    """

    # PicakCNN used 950ms, 41 frames for short-frame variant. 50% overlap
    ws = frame_samples * window_frames
    start = 0
    while start < length:
        end = min(start + ws, length)
        yield start, end
        start += (ws * (1-overlap))

def test_windows_shorter_than_window():
    frame_samples=256
    window_frames=64
    fs=16000
    length = 0.4*fs
    w = list(sample_windows(int(length), frame_samples, window_frames))
    assert len(w) == 1, len(w)
    assert w[-1][1] == length

def test_window_typical():
    frame_samples=256
    window_frames=64
    fs=16000
    length = 4.0*fs
    w = list(sample_windows(int(length), frame_samples, window_frames))
    assert len(w) == 8, len(w) 
    assert w[-1][1] == length

test_windows_shorter_than_window()
test_window_typical() 

def expand_training_set(samples, frame_samples, window_frames,
                        sample_rate=16000, cut_length=1.0, augmentations=0):
    chunks = {
        'slice_file_name': [],
        'fold': [],
        'classID': [],
        'start': [],
        'end': [],
        'augmentation': [],
    }
    
    for (index, sample) in samples.iterrows():
        duration = sample.end - sample.start
        length = int(sample_rate * duration)
        
        for aug in range(-1, augmentations):
        
            for win in sample_windows(length, frame_samples, window_frames):
                start, end = win
                chunks['slice_file_name'].append(sample.slice_file_name)
                chunks['fold'].append(sample.fold)
                # to assume class is same as that of parent sample maybe a bit optimistic
                # not certain that every chunk has content representative of class
                # alternative would be multi-instance learning
                chunks['classID'].append(sample.classID) 
                chunks['start'].append(start/sample_rate)
                chunks['end'].append(end/sample_rate)
                chunks['augmentation'].append(None if aug == -1 else aug)
            
    df = pandas.DataFrame(chunks)
    
    if cut_length:
        w = (df.end-df.start > cut_length)
        cleaned = df[w]
        print('cutting {} samples shorter than {} seconds'.format(len(df) - len(cleaned), cut_length))
    
    return cleaned


def train_model(out_dir, fold, builder, loader,
                frame_samples, window_frames,
                train_samples=12000, val_samples=3000,
                batch_size=200, epochs=50, seed=1, learning_rate=3e-4):
    """Train a single model"""    

    model = builder()
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=learning_rate),
                  metrics=['accuracy'])

    train = expand_training_set(fold[0], frame_samples=frame_samples, window_frames=window_frames,
                                augmentations=0)
    val = expand_training_set(fold[1], frame_samples=frame_samples, window_frames=window_frames)

    print('dataset', train.shape, val.shape)

    train = train.sample(train_samples, replace=False, random_state=seed)
    val = val.sample(val_samples, replace=False, random_state=seed)

    model_path = os.path.join(out_dir, 'e{epoch:02d}-v{val_loss:.2f}.model.hdf5')
    checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_acc', mode='max',
                                         period=1, verbose=1, save_best_only=False)
    callbacks_list = [checkpoint]

    train_gen = dataframe_generator(train, train.classID, loader=loader, batchsize=batch_size)
    val_gen = dataframe_generator(val, val.classID, loader=loader, batchsize=batch_size)

    hist = model.fit_generator(train_gen, validation_data=val_gen,
                        steps_per_epoch=math.ceil(len(train)/batch_size),
                        validation_steps=math.ceil(len(val)/batch_size),
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

default_feature_settings = dict(
    feature='mels',
    samplerate=16000,
    n_mels=32,
    fmin=0,
    fmax=8000,
    n_fft=512,
    hop_length=256,
    augmentations=5,
)

default_training_settings = dict(
    epochs=50,
    batch=50,
    train_samples=37000,
    val_samples=3000,
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
    for k, v in default_feature_settings.items():
        t = int if k != 'feature' else str
        a('--{}'.format(k), type=t, default=v, help='default: %(default)s')

    # Expose training settings
    for k, v in default_training_settings.items():
        t = int
        a('--{}'.format(k), type=t, default=v, help='default: %(default)s')

    parsed = parser.parse_args(args)

    return parsed


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
    # TODO: support specifying model on cmdline
    model_settings = dict(
        model='sbcnn',
        frames=72,
    )

    # feature settings
    feature_settings = {}
    for k in default_feature_settings.keys():
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

    def load(sample):
        return load_sample(sample, feature_settings, feature_dir=feature_dir)

    def build_model():
        m = sbcnn.build_model(bands=feature_settings['n_mels'],
                    frames=model_settings['frames'], channels=1, pool=(3,3))
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
