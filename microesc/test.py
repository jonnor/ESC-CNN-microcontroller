
import math
import os.path
import sys

import keras
import sklearn
import pandas
import numpy
import keras.metrics

from . import urbansound8k, features, common


def load_model_info(jobs_dir, job_dir):
    template, date, time, rnd, fold = job_dir.split('-')
    hist_path = os.path.join(jobs_dir, job_dir, 'train.csv')

    df = pandas.read_csv(hist_path)

    df['epoch'] = df.epoch + 1
    df['fold'] = int(fold[-1])
    df['template'] = template
    df['run'] = '-'.join([date, time, rnd])
    
    models = []
    for fname in os.listdir(os.path.join(jobs_dir, job_dir)):
        if fname.endswith('model.hdf5'):
            models.append(fname)
    
    models = sorted(models)
    assert models[0].startswith('e01-')
    last_model = models[len(models)-1]
    expected_last = 'e{:02d}-'.format(len(models))
    assert last_model.startswith(expected_last), (last_model, expected_last)

    df['model'] = [ os.path.join(jobs_dir, job_dir, m) for m in models ]
    return df

def load_train_history(jobs_dir, limit=None):

    jobs = os.listdir(jobs_dir)
    if limit:
        matching = [ d for d in jobs if limit in d ]
    else:
        matching = jobs
    #assert len(matching) == 9, "Expected 9 folds, found {} matching {}".format(len(matching), job_id)

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
    return history.groupby('fold').apply(best_by_loss)


def evaluate(models, folds, test, predictor):

    val_scores = []
    test_scores = []

    def score(model, data):
        y_true = data.classID
        p = predictor(model, data)
        y_pred = numpy.argmax(p, axis=1)
        # other metrics can be derived from confusion matrix
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        print('acc', acc)
        confusion = sklearn.metrics.confusion_matrix(y_true, y_pred)
        return confusion

    # validation
    for i, m in enumerate(models):
        validation_fold = folds[i][1]
        s = score(m, validation_fold)
        val_scores.append(s)

    # test
    for i, m in enumerate(models):
        s = score(m, test)
        test_scores.append(s)
        

    return val_scores, test_scores


def parse(args):

    import argparse

    parser = argparse.ArgumentParser(description='Test trained models')
    a = parser.add_argument

    common.add_arguments(parser)

    a('--run', dest='run', default='',
        help='%(default)s')
    a('--model', dest='model', default='',
        help='%(default)s')

    a('--out', dest='results_dir', default='./data/results',
        help='%(default)s')


    parsed = parser.parse_args(args)

    return parsed

def main():
    
    args = parse(sys.argv[1:])
    if not args.run:
        args.run = args.experiment

    out_dir = os.path.join(args.results_dir, args.experiment)

    common.ensure_directories(out_dir)

    urbansound8k.maybe_download_dataset(args.datasets_dir)
    data = urbansound8k.load_dataset()
    folds, test = urbansound8k.folds(data)
    exsettings = common.load_experiment(args.experiments_dir, args.experiment)
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
    best = pick_best(history)

    print('Loading models...')
    models = best['model'].apply(lambda p: keras.models.load_model(p))
    print('Best model', best.voted_val_acc)

    print('Testing models...')
    val, test = evaluate(models, folds, test, predictor=predict)

    results_path = os.path.join(out_dir, 'confusion.npz')
    numpy.savez(results_path, val=val, test=test)

    print('Wrote to', results_path)

if __name__ == '__main__':
    main()


