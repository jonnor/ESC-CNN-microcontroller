

"""
urbansound8k.py: Helper for downloading and loading the Urbansound8k dataset as a Pandas DataFrame

Copyright: Jon Nordby <jononor@gmail.com>
License: MIT https://opensource.org/licenses/MIT

The UrbanSound8K dataset is under the Creative Commons Attribution Noncommercial License (by-nc), version 3.
"""

import os.path
import urllib.request
import tarfile

import numpy
import pandas

here = os.path.dirname(__file__)

default_path = os.path.join(here, '../data/UrbanSound8K/')

classes = {
    'car_horn': 1,
    'dog_bark': 3,
    'children_playing': 2,
    'engine_idling': 5,
    'street_music': 9,
    'drilling': 4,
    'air_conditioner': 0,
    'gun_shot': 6,
    'siren': 8,
    'jackhammer': 7
}
classnames = [ ni[0] for ni in sorted(classes.items(), key=lambda kv: kv[1]) ]

download_urls = [
	'https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz',
    'https://serv.cusp.nyu.edu/files/jsalamon/datasets/UrbanSound8K.tar.gz',
    'https://storage.googleapis.com/urbansound8k/UrbanSound8K.tar.gz',
] 
def maybe_download_dataset(workdir):

    if not os.path.exists(workdir):
        os.makedirs(workdir)

    dir_path = os.path.join(workdir, 'UrbanSound8K')
    archive_path = dir_path+'.tar.gz'

    last_progress = None
    def download_progress(count, blocksize, totalsize):
        nonlocal last_progress

        p = int(count * blocksize * 100 / totalsize)
        if p != last_progress:
            print('\r{}%'.format(p), end='\r')
            last_progress = p

    if not os.path.exists(dir_path):
        print('Could not find', dir_path)        

        if not os.path.exists(archive_path):
            u = download_urls[0]
            print('Downloading...', u)
            urllib.request.urlretrieve(u, archive_path, reporthook=download_progress)

        print('Extracting...', archive_path)
        # Note: .zip file is kept around
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(workdir)

    return dir_path


def load_dataset():
    metadata_path = os.path.join(here, 'datasets/UrbanSound8K.csv')

    samples = pandas.read_csv(metadata_path)
    return samples

def sample_path(sample, dataset_path = None):
    if not dataset_path:
        dataset_path=default_path

    return os.path.join(dataset_path, 'audio', 'fold'+str(sample.fold), sample.slice_file_name)


# Split the 10 folds into training, testing, and
def folds(data):
    fold_idxs = folds_idx(n_folds=10) 
    assert len(fold_idxs) == 10

    folds = []
    for fold in fold_idxs:
        train, val, test = fold

        # our folds are 1-indexed instead of 0...
        train = numpy.array(train) + 1
        val = numpy.array(val) + 1
        test = numpy.array(test) + 1
        fold_train = data[data.fold.isin(train)]
        fold_val = data[data.fold.isin(val)]
        fold_test = data[data.fold.isin(test)]

        # post-condition
        train_folds = set(fold_train.fold.unique())
        val_folds = set(fold_val.fold.unique())
        test_folds = set(fold_test.fold.unique())
        assert len(train_folds) == 8, len(train_folds)
        assert train_folds.intersection(val_folds) == set()
        assert train_folds.intersection(test_folds) == set()
        assert val_folds.intersection(test_folds) == set()

        folds.append((fold_train, fold_val, fold_test))
        
    return folds


def ensure_valid_fold(fold, n_folds=10):
    train, val, test = fold
    assert len(train) == n_folds-2, len(train)
    assert 0 <= train[0] < n_folds, train[0]
    assert len(val) == 1, len(val)
    assert 0 <= val[0] < n_folds, val[0]
    assert len(test) == 1, len(test)
    assert 0 <= test[0] < n_folds, test[0]
    assert test[0] != val[0]
    test_overlap = set(train).intersection(set(test))
    val_overlap =  set(train).intersection(set(val))
    assert test_overlap == set(), test_overlap
    assert val_overlap == set(), val_overlap
    assert sorted(train + val + test) == list(range(0, n_folds))
    return True

def folds_idx(n_folds):
    """Generate fold indices for cross-validation.
    Each fold has 1 validation, 1 test set and the remaining train"""
    test_fold = 10

    folds = []
    all_folds = list(range(0, n_folds))
    for idx in range(0, n_folds):
        test = [ all_folds[idx] ]
        # using Python negative index support for lists to wrap around at edges of array
        val =  [ all_folds[idx-1] ]
        train = list(set(all_folds).difference(set(test+val)))
        fold = ( train, val, test )
        ensure_valid_fold(fold)
        folds.append(fold)

    assert len(folds) == n_folds, len(folds)
    return folds


