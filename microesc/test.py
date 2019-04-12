
import math
import os.path
import sys

import keras
import sklearn
import pandas
import numpy
import keras.metrics

from . import urbansound8k, features, common, stats


def load_model_info(jobs_dir, job_dir):
    experiment, date, time, rnd, fold = job_dir.split('-')
    hist_path = os.path.join(jobs_dir, job_dir, 'train.csv')

    df = pandas.read_csv(hist_path)

    df['epoch'] = df.epoch + 1
    df['fold'] = int(fold[-1])
    df['experiment'] = experiment
    df['run'] = '-'.join([date, time, rnd])
    
    models = []
    for fname in os.listdir(os.path.join(jobs_dir, job_dir)):
        if fname.endswith('model.hdf5'):
            models.append(fname)
    
    def get_epoch(s):
        e = s.split('-')[0].lstrip('e')
        e = int(e)
        return e

    models = sorted(models, key=get_epoch)
    assert models[0].startswith('e01-')
    last_model = models[len(models)-1]
    expected_last = 'e{:02d}-'.format(len(models))
    assert last_model.startswith(expected_last), (last_model, expected_last)

    df['model_path'] = [ os.path.join(jobs_dir, job_dir, m) for m in models ]
    return df

def load_train_history(jobs_dir, limit=None):

    jobs = os.listdir(jobs_dir)
    if limit:
        matching = [ d for d in jobs if limit in d ]
    else:
        matching = jobs

    dataframes = []
    
    for job_dir in matching:

        try:
            df = load_model_info(jobs_dir, job_dir)
        except (FileNotFoundError, ValueError) as e:
            print('Failed to load job {}: {}'.format(job_dir, str(e)))
            continue

        dataframes.append(df)
        

    df = pandas.concat(dataframes)
    return df

def test_load_history():

    jobs_dir = '../../jobs'
    job_id = 'sbcnn44k128aug-20190227-0220-48ba'
    df = load_history()

def pick_best(history, n_best=1):

    def best_by_loss(df):
        return df.sort_values('voted_val_acc', ascending=False).head(n_best)
    return history.groupby(['experiment', 'fold']).apply(best_by_loss)

def evaluate_model(predictor, model_path, val_data, test_data):

    def score(model, data):
        y_true = data.classID
        p = predictor(model, data)
        y_pred = numpy.argmax(p, axis=1)
        # other metrics can be derived from confusion matrix
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        labels = list(range(len(urbansound8k.classnames)))
        confusion = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=labels)
        return acc, confusion

    model = keras.models.load_model(model_path)

    salience_info = { 'foreground': 1, 'background': 2 }
    test_info = { 'val': val_data, 'test': test_data }
    out = {}
    for setname, data in test_info.items():
        for variant, salience in salience_info.items():   
            key = '{}_{}'.format(setname, variant)
            acc, confusion = score(model, data[data.salience == salience])
            print('acc for ', key, acc) 
            out[key] = confusion

    out['val'] = out['val_foreground'] + out['val_background']
    out['test'] = out['test_foreground'] + out['test_background']
    return out

def evaluate(models, folds, testset, predictor, out_dir, dry_run=False):

    def eval_experiment(df):
        results = {}
        by_fold = df.sort_index(level="fold", ascending=True)

        for idx, row in by_fold.iterrows():
            print('Testing model {} fold={}'.format(row['experiment'], row['fold']))

            model_path = row['model_path']
            val = folds[row['fold']][1]
            test = testset
            if dry_run:
                val = test[0:20]
                test = test[0:20]
            
            result = evaluate_model(predictor, model_path, val, test)

            # convert to dict-of-arrays
            for k, v in result.items():
                if results.get(k) is None:
                    results[k] = []
                results[k].append(v)

        exname = df['experiment'].unique()[0]
        results_path = os.path.join(out_dir, '{}.confusion.npz'.format(exname))
        numpy.savez(results_path, **results)
        print('Wrote', results_path)
        return results_path

    out = models.groupby(level='experiment').apply(eval_experiment)

    return out


def parse(args):

    import argparse

    parser = argparse.ArgumentParser(description='Test trained models')
    a = parser.add_argument

    common.add_arguments(parser)

    a('--run', dest='run', default='',
        help='%(default)s')

    a('--check', action='store_true', default='',
        help='Run a check pass, not actually evaluating')

    a('--out', dest='results_dir', default='./data/results',
        help='%(default)s')


    parsed = parser.parse_args(args)

    return parsed

def main():
    
    args = parse(sys.argv[1:])
    out_dir = os.path.join(args.results_dir, args.run)

    common.ensure_directories(out_dir)

    urbansound8k.maybe_download_dataset(args.datasets_dir)
    data = urbansound8k.load_dataset()
    folds, test = urbansound8k.folds(data)
    exsettings = common.load_settings_path(args.settings_path)
    frames = exsettings['frames']
    voting = exsettings['voting']
    overlap = exsettings['voting_overlap']
    settings = features.settings(exsettings)


    all_folds = pandas.concat([f[0] for f in folds])
    train_files = set(all_folds.slice_file_name.unique())
    test_files = set(test.slice_file_name.unique())
    assert len(train_files) > 7000
    assert len(test_files) > 700
    common_files = train_files.intersection(test_files)
    assert len(common_files) == 0

    def load_sample(sample):
        return features.load_sample(sample, settings, start_time=sample.start,
                    window_frames=frames, feature_dir=args.features_dir)

    def predict(model, data):
        return features.predict_voted(exsettings, model, data, loader=load_sample,
                                        method=voting, overlap=overlap)

    history = load_train_history(args.models_dir, args.run)
    n_folds = len(history.fold.unique())
    n_experiments = len(history.experiment.unique())
    print("Found {} experiments across {} folds", n_folds, n_experiments)

    best = pick_best(history)
    print('Best models\n', best[['epoch', 'voted_val_acc']])

    print('Computing model info')
    def get_stats(row):
        ex = row.iloc[0]
        model = ex['model_path']
        model_stats, layer_info = stats.model_info(model)
        layer_info_path = os.path.join(out_dir, '{}.layers.csv'.format(ex['experiment']))
        layer_info.to_csv(layer_info_path)
        return pandas.Series(model_stats)

    model_stats = best.groupby(level='experiment').apply(get_stats)
    print('Model stats\n', model_stats)
    model_stats.to_csv(os.path.join(out_dir, 'stm32stats.csv'))

    print('Testing models...')
    results = evaluate(best, folds, test, predictor=predict, out_dir=out_dir, dry_run=args.check)


if __name__ == '__main__':
    main()


