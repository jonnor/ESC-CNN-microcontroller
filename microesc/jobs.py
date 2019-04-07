
import sys
import os.path
import subprocess
import datetime
import uuid

import pandas
import numpy

from microesc import common

def arglist(options):
    args = [ "--{}={}".format(k, v) for k, v in options.items() ]
    return args

def command_for_job(options):
    args = [
        'python3', 'train.py'
    ]
    args += arglist(options)
    return args

def generate_train_jobs(experiments, settings_path, folds, overrides):

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M') 
    unique = str(uuid.uuid4())[0:4]   
    def name(experiment, fold):
        name = "-".join([experiment, timestamp, unique])
        return name+'-fold{}'.format(fold)

    def job(exname, experiment):

        for fold in folds:
            n = name(exname, fold)
            
            options = {
                'name': n,
                'fold': fold,
                'settings': settings_path,
            }
            for k, v in experiment.items():
                # overrides per experiment
                options[k] = v

            for k, v in overrides.items():
                options[k] = v

            cmd = command_for_job(options)
            return cmd

    # FIXME: better job name
    jobs = [ job(str(idx), ex) for idx, ex in experiments.iterrows() ] 
    return jobs

def parse(args):

    import argparse

    parser = argparse.ArgumentParser(description='Generate jobs')

    common.add_arguments(parser)
    a = parser.add_argument

    a('--experiments', default='models.csv',
        help='%(default)s')
    a('--check', action='store_true',
        help='Only run a pre-flight check')
    
    parsed = parser.parse_args(args)

    return parsed

def main():
    args = parse(sys.argv[1:])

    experiments = pandas.read_csv(args.experiments)
    settings = common.load_settings_path(args.settings_path)

    overrides = {}
    folds = list(range(0, 9))
    if args.check:
        folds = (1,)
        overrides['train_samples'] = settings['batch']*1
        overrides['val_samples'] = settings['batch']*1

    cmds = generate_train_jobs(experiments, args.settings_path, folds, overrides)

    print('\n'.join(" ".join(cmd) for cmd in cmds))

if __name__ == '__main__':
    main()
