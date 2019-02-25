

import urllib.request
import os.path
import zipfile
import math

import pandas
import numpy
import keras
import librosa

import preprocess


default_base_url = 'https://storage.googleapis.com/urbansound8k'


def features_url(settings, base=default_base_url):
    id = preprocess.settings_id(settings)
    ext = '.zip'  
    return "{}/{}{}".format(base, id, ext)


def maybe_download_features(settings, workdir):

    feature_dir = os.path.join(workdir, preprocess.settings_id(settings))
    feature_zip = feature_dir + '.zip'
    feature_url = features_url(settings)

    def download_progress(count, blocksize, totalsize):
        p = int(count * blocksize * 100 / totalsize)
        print('{}%'.format(p))

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


def train_model(name, fold, builder, loader,
                frame_samples, window_frames, 
                train_samples=35000, val_samples=3000,
                batch_size=200, epochs=50, seed=1, learning_rate=3e-4):
    """Train a single model"""    

    model = builder()
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=learning_rate),
                  metrics=['accuracy'])

    train = expand_training_set(fold[0], frame_samples=frame_samples, window_frames=window_frames,
                                augmentations=0)
    val = expand_training_set(fold[1], frame_samples=frame_samples, window_frames=window_frames)

    train = train.sample(train_samples, replace=False, random_state=seed)
    val = val.sample(val_samples, replace=False, random_state=seed)

    model_name = name+'.e{epoch:02d}-v{val_loss:.2f}.model.hdf5'
    checkpoint = keras.callbacks.ModelCheckpoint(model_name, monitor='val_acc', mode='max',
                                         period=1, verbose=1, save_best_only=False)
    callbacks_list = [checkpoint]

    train_gen = dataframe_generator(train, train.classID, loader=loader, batchsize=batch_size)
    val_gen = dataframe_generator(val, val.classID, loader=loader, batchsize=batch_size)

    hist = model.fit_generator(train_gen, validation_data=val_gen,
                        steps_per_epoch=math.ceil(len(train)/batch_size),
                        validation_steps=math.ceil(len(val)/batch_size),
                        callbacks=callbacks_list,
                        epochs=epochs, verbose=1)
    
    return hist

def history_dataframe(h):
    data = {}
    data['epoch'] = h.epoch
    for k, v in h.history.items():
        data[k] = v
    df = pandas.DataFrame(data)
    return df


def main():

    # experiment settings
    feature_dir = './features'
    output_dir = './out'
    name = 'trainit'
    fold = 0

    # model settings
    # TODO: support specifying model on cmdline
    window_frames = 72

    # TODO: support specifying hyperparameters on cmdline


    # feature settings
    # FIXME: specify feature settings on cmdline
    settings = dict(
        feature='mels',
        samplerate=44100,
        n_mels=128,
        fmin=0,
        fmax=22050,
        n_fft=1024,
        hop_length=1024,
        augmentations=5,
    )

    maybe_download_features(settings, feature_dir)

    data = urbansound8k.load_dataset()
    folds, test = urbansound8k.folds(data)

    def load_sample(sample):
        return train.load_sample(sample, settings, feature_dir='../../scratch/aug')

    def build_model():
        model = sbcnn.build_model(bands=settings['n_mels'], frames=window_frames, channels=1, pool=(3,3))

    h = train.train_model(name, folds[fold],
                          builder=build_model, loader=load_sample,
                          frame_samples=settings['hop_length'], window_frames=window_frames,
                          epochs=1, train_samples=20, val_samples=20, batch_size=5)

    hist = history_dataframe(h)
    hist.to_csv(model_name+'.history.csv')


if __name__ == '__main__':
    main()
