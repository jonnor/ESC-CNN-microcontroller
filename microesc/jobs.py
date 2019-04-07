
import sys
import os.path
import subprocess
import datetime
import uuid
import time

import pandas
import numpy

from microesc import common

def arglist(options):
    def format_arg(k, v):
        if v is None:
            return "--{}".format(k)
        else:
            return "--{}={}".format(k, v)

    args = [ format_arg(k, v) for k, v in options.items() ]
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
                if k == 'modelcheck':
                    if v == 'skip':
                        options['skip_model_check'] = None
                else:
                    options[k] = v

            for k, v in overrides.items():
                options[k] = v

            print('o', options)

            return options

    # FIXME: better job name
    jobs = [ job(str(idx), ex) for idx, ex in experiments.iterrows() ] 
    return jobs

import joblib
import subprocess

def run_job(jobdata, out_dir, verbose=2):
    args = command_for_job(jobdata)
    log_dir = os.path.join(out_dir, jobdata['name'])
    common.ensure_directories(log_dir)
    log_path = os.path.join(log_dir, 'stdout.log') 

    cmdline = ' '.join(args)
    with open(os.path.join(log_dir, 'cmdline'), 'w') as f:
        f.write(cmdline)

    start = time.time()
    print('starting job', cmdline, log_path)

    # Read stdout and write to log, following https://stackoverflow.com/a/18422264/1967571
    exitcode = None
    with open(log_path, 'w') as log_file:
        process = subprocess.Popen(args, shell=False, stdout=subprocess.PIPE)
        for line in iter(process.stdout.readline, b''):
            line = line.decode('utf-8')

            if verbose > 2:
                sys.stdout.write(line)

            log_file.write(line)
            log_file.flush()
        exitcode = process.wait()
    
    end = time.time()
    res = {
        'start': start,
        'end': end,
        'exitcode': exitcode,
    }

def run_jobs(commands, out_dir, n_jobs=5, verbose=1):

    jobs = [joblib.delayed(run_job)(cmd, out_dir) for cmd in commands]
    out = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)
    return out


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
        batches = 1
        overrides['batch'] = 10
        overrides['epochs'] = 1
        overrides['train_samples'] = batches * overrides['batch']
        overrides['val_samples'] = batches * overrides['batch']

    cmds = generate_train_jobs(experiments, args.settings_path, folds, overrides)

    run_jobs(cmds, args.models_dir)

if __name__ == '__main__':
    main()
