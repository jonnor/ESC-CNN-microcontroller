
import os.path

import yaml

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def ensure_directories(*dirs):
    for dir in dirs:
        ensure_dir(dir)


def add_arguments(parser):
    a = parser.add_argument

    a('--datasets', dest='datasets_dir', default='./data/datasets',
        help='%(default)s')
    a('--features', dest='features_dir', default='./data/features',
        help='%(default)s')
    a('--models', dest='models_dir', default='./data/models',
        help='%(default)s')

    a('--experiments', dest='experiments_dir', default='./experiments',
        help='%(default)s')
    a('--experiment', dest='experiment', default='', # TODO: add a default
        help='%(default)s')


def load_experiment(folder, name):
    path = os.path.join(folder, name+'.yaml')

    with open(path, 'r') as config_file:
        settings = yaml.load(config_file.read())    
    
    return settings
