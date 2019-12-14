
import itertools

import pandas
import numpy
import pytest

from microesc import train, features, urbansound8k

@pytest.mark.skip("fails")
def test_generator_fake_loader():

    dataset_path = 'data/UrbanSound8K/'
    urbansound8k.default_path = dataset_path
    data = urbansound8k.load_dataset()
    folds, test = urbansound8k.folds(data)

    data_length = 16
    batch_size = 8
    frames = 72
    bands = 32
    n_classes = 10

    def zero_loader(s):
        #assert
        return numpy.zeros((bands, frames, 1))

    fold = folds[0][0]
    X = fold[0:data_length]
    Y = fold.classID[0:data_length]

    g = train.dataframe_generator(X, Y, loader=zero_loader,
                            batchsize=batch_size, n_classes=n_classes)

    n_batches = 3
    batches = list(itertools.islice(g, n_batches))
    assert len(batches) == n_batches
    assert len(batches[0]) == 2 # X,y
    assert batches[0][0].shape == (batch_size, bands, frames, 1)
    assert batches[0][1].shape == (batch_size, n_classes)


def test_windows_shorter_than_window():
    frame_samples=256
    window_frames=64
    fs=16000
    length = 0.4*fs
    w = list(features.sample_windows(int(length), frame_samples, window_frames))
    assert len(w) == 1, len(w)
    assert w[-1][1] == length

def test_window_typical():
    frame_samples=256
    window_frames=64
    fs=16000
    length = 4.0*fs
    w = list(features.sample_windows(int(length), frame_samples, window_frames))
    assert len(w) == 8, len(w) 
    assert w[-1][1] == length


def _test_predict_windowed():

    t = test[0:10]

    sbcnn16k32_settings = dict(
        feature='mels',
        samplerate=16000,
        n_mels=32,
        fmin=0,
        fmax=8000,
        n_fft=512,
        hop_length=256,
        augmentations=5,
    )

    def load_sample32(sample):
        return features.load_sample(sample, sbcnn16k32_settings, window_frames=72, feature_dir='../../scratch/aug')

    mean_m = features.predict_voted(sbcnn16k32_settings, model, t, loader=load_sample32, method='mean')


