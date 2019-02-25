
import os.path
import urllib.request

import pandas

here = os.path.dirname(__file__)

default_path = os.path.join(here, '../../data/UrbanSound8K/')

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

def load_dataset(path = None):
    if path is None:
        path = default_path

    u = 'https://storage.googleapis.com/urbansound8k/UrbanSound8K.csv'
    metadata_path = os.path.join(path, 'metadata/UrbanSound8K.csv')
    if not os.path.exists(metadata_path):
        urllib.request.urlretrieve(u, metadata_path)

    samples = pandas.read_csv(metadata_path)
    return samples

def sample_path(sample, dataset_path = None):
    if not dataset_path:
        dataset_path=default_path

    return os.path.join(dataset_path, 'audio', 'fold'+str(sample.fold), sample.slice_file_name)


# Use fold=10 for testing, as recommended by Urbansound8k dataset authors
def folds(data):
    test_fold = 10
    train = data[data.fold != test_fold]
    test = data[data.fold == test_fold]

    folds = []
    for fold in range(1, 10):
        assert fold != test_fold
        fold_train = train[train.fold != fold]
        fold_val = train[train.fold == fold]
        folds.append((fold_train, fold_val))
        
    return folds, test
