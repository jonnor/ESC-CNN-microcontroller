
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


