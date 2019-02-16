
import os.path
import pandas

here = os.path.dirname(__file__)

default_path = os.path.join(here, '../../data/UrbanSound8K/')

def load_dataset(path = None):
    if path is None:
        path = default_path

    metadata_path = os.path.join(path, 'metadata/UrbanSound8K.csv')
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
