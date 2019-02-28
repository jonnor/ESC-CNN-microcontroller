
import os.path
import urllib.request
import tarfile

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

download_urls = [
    'https://storage.googleapis.com/urbansound8k/UrbanSound8K.tar.gz',
	'https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz',
    'https://serv.cusp.nyu.edu/files/jsalamon/datasets/UrbanSound8K.tar.gz',
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
