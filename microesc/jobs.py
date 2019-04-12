
import sys
import os.path
import subprocess
import datetime
import uuid
import time
import subprocess

import joblib
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

    def create_job(exname, experiment, fold):
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

        return options

    jobs = []
    for fold in folds:
        for idx, ex in experiments.iterrows():
            j = create_job(str(idx), ex, fold)
            jobs.append(j)

    assert len(jobs) == len(experiments) * len(folds), len(jobs)
    return jobs



def run_job(jobdata, out_dir, verbose=2):
    args = command_for_job(jobdata)
    job_dir = os.path.join(out_dir, jobdata['name'])
    common.ensure_directories(job_dir)
    log_path = os.path.join(job_dir, 'stdout.log') 

    cmdline = ' '.join(args)
    with open(os.path.join(job_dir, 'cmdline'), 'w') as f:
        f.write(cmdline)

    start = time.time()
    print('starting job', cmdline)
    print('job log', log_path)

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

        files = os.listdir(job_dir)
        assert 'train.csv' in files, files
        assert 'history.csv' in files, files
        model_files = [ p for p in files if p.endswith('.hdf5') ]
        assert len(model_files) > 0, files

    end = time.time()
    res = {
        'start': start,
        'end': end,
        'exitcode': exitcode,
    }
    return res

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
    a('--jobs', type=int, default=5,
        help='Number of parallel jobs')
    a('--folds', type=int, default=9,
        help='Number of folds to test')

    parsed = parser.parse_args(args)

    return parsed

def main():
    args = parse(sys.argv[1:])

    experiments = pandas.read_csv(args.experiments)
    settings = common.load_settings_path(args.settings_path)

    overrides = {}
    folds = list(range(0, args.folds))
    if args.check:
        batches = 2
        overrides['batch'] = 10
        overrides['epochs'] = 1
        overrides['train_samples'] = batches * overrides['batch']
        overrides['val_samples'] = batches * overrides['batch']

    cmds = generate_train_jobs(experiments, args.settings_path, folds, overrides)
    print('Preparing {} jobs', len(cmds))
    print('\n'.join([ c['name'] for c in cmds ]))

    out = run_jobs(cmds, args.models_dir, n_jobs=args.jobs)
    print(out)
    success = all([ o['exitcode'] == 0 for o in out ])
    assert success

if __name__ == '__main__':
    main()
