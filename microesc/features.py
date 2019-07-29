
"""Feature extraction"""


import os.path
import urllib.request
import zipfile
import collections

import pandas
import numpy
import keras
import librosa

from . import urbansound8k
from . import settings as Settings


default_base_url = 'https://storage.googleapis.com/urbansound8k'


def settings(base):
    feature_settings = {}
    default = Settings.default_feature_settings
    for k in default.keys():
        feature_settings[k] = base.get(k, default[k])
    return feature_settings


def settings_id(settings):
    keys = sorted([ k for k in settings.keys() if k != 'feature' ])
    feature = settings['feature']

    settings_str = ','.join([ "{}={}".format(k, str(settings[k])) for k in keys ])
    return feature + ':' + settings_str


def feature_path(sample, out_folder, augmentation=None):
    path = urbansound8k.sample_path(sample)
    tokens = path.split(os.sep)
    filename = tokens[-1]
    filename = filename.replace('.wav', '.npz')
    if augmentation is not None:
        filename = filename.replace('.npz', '.aug{}.npz'.format(augmentation))

    out_fold = os.path.join(out_folder, tokens[-2])
    return os.path.join(out_fold, filename)


def compute_mels(y, settings):
    sr = settings['samplerate']
    from librosa.feature import melspectrogram 
    mels = melspectrogram(y, sr=sr,
                         n_mels=settings['n_mels'],
                         n_fft=settings['n_fft'],
                         hop_length=settings['hop_length'],
                         fmin=settings['fmin'],
                         fmax=settings['fmax'])
    return mels


def features_url(settings, base=default_base_url):
    id = settings_id(settings)
    ext = '.zip'  
    return "{}/{}{}".format(base, id, ext)


def maybe_download(settings, workdir):

    feature_dir = os.path.join(workdir, settings_id(settings))
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


def extract_window(inmels, settings, start_time, normalize):

    n_mels = settings['n_mels']
    sample_rate = settings['samplerate']
    hop_length = settings['hop_length']
    window_frames = settings['frames']

    start = int(start_time * (sample_rate / hop_length))
    end = start + window_frames
    #print('s', start, end, inmels.shape[1], end-start/inmels.shape[1])
    mels = inmels[:, start:end]

    # Normalize the window
    if mels.shape[1] > 0:
        if normalize == 'max':
            mels /= (numpy.max(mels) + 1e-9)
            mels = librosa.core.power_to_db(mels, top_db=80)
        elif normalize == 'meanstd':
            mels = librosa.core.power_to_db(mels, top_db=80)
            mels -= numpy.mean(mels)
            mels /= ( numpy.std(mels) + 1e-9)
        else:
            mels = librosa.core.power_to_db(mels, top_db=80, ref=0.0)
    else:
        print('Warning: Sample {} with start {} has 0 length'.format(inmels.shape, start_time))

    # Pad to standard size
    padded = numpy.full((n_mels, window_frames), 0.0, dtype=float)
    inp = mels[:, 0:min(window_frames, mels.shape[1])]
    padded[:, 0:inp.shape[1]] = inp

    # add channel dimension
    data = numpy.expand_dims(padded, -1)
    return data


def load_sample(sample, exsettings, feature_dir,
                overlap=0, start=0, augmentation=None, normalize='meanstd'):

    n_mels = exsettings['n_mels']
    f_settings = settings(exsettings)

    # Load precomputed features
    folder = os.path.join(feature_dir, settings_id(f_settings))
    path = feature_path(sample, out_folder=folder, augmentation=augmentation)
    mels = numpy.load(path)['arr_0']
    assert mels.shape[0] == n_mels, mels.shape

    sample_rate = exsettings['samplerate']
    frame_samples = exsettings['hop_length']
    window_frames = exsettings['frames']

    # augmentations may change the sample duration
    duration = mels.shape[1] * frame_samples/sample_rate
    duration -= start

    # cut into windows, and normalized
    window_length = ((frame_samples * window_frames) / sample_rate)
    hop_length = (1-overlap) * window_length
    n_windows = int(numpy.ceil(4.0 / hop_length))

    starts = [ start + (i*hop_length) for i in range(0, n_windows) ]
    if duration < window_length:
        # make sure short files have at least one window
        starts = [ starts[0] ]
    else:
        starts = [ s for s in starts if s < (duration-(hop_length/2)) ]

    windows = []
    for s in starts:
        d = extract_window(mels, exsettings, s, normalize=normalize)
        windows.append(d)
    d = numpy.stack(windows)

    # single numpy array, zero-padded
    s = (n_windows, n_mels, window_frames, 1)
    windows = numpy.zeros(shape=s)
    windows[:d.shape[0], :, :, :] = d

    return windows

def predict_voted(settings, model, samples, loader, method='mean', overlap=0.5):

    out = []
    for _, sample in samples.iterrows():
        windows = load_windows(sample, settings, loader, overlap=overlap)
        inputs = numpy.stack(windows)

        predictions = model.predict(inputs)
        if method == 'mean':
            p = numpy.mean(predictions, axis=0)
            assert len(p) == 10
            out.append(p)
        elif method == 'majority':
            votes = numpy.argmax(predictions, axis=1)
            p = numpy.bincount(votes, minlength=10) / len(votes)
            out.append(p)

    ret = numpy.stack(out)
    assert len(ret.shape) == 2, ret.shape
    assert ret.shape[0] == len(out), ret.shape
    assert ret.shape[1] == 10, ret.shape # classes

    return ret


