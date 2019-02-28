
import math
import os.path
import sys
import collections

import keras
import sklearn
import pandas
import numpy
import keras.metrics

import urbansound8k
import preprocess
import features
import train

Sample = collections.namedtuple('Sample', 'start end fold slice_file_name')

def load_windows(sample, settings, loader, window_frames, overlap):
    sample_rate = settings['samplerate']
    frame_samples = settings['hop_length']

    windows = []

    duration = sample.end - sample.start
    length = int(sample_rate * duration)

    for win in train.sample_windows(length, frame_samples, window_frames, overlap=overlap):
        chunk = Sample(start=win[0]/sample_rate,
                       end=win[1]/sample_rate,
                       fold=sample.fold,
                       slice_file_name=sample.slice_file_name)    
        d = loader(chunk)
        windows.append(d)

    return windows

def predict_voted(settings, model, samples, loader, window_frames, method='mean', overlap=0.5):

    out = []
    for _, sample in samples.iterrows():
        windows = load_windows(sample, settings, loader, window_frames, overlap=overlap)
        inputs = numpy.stack(windows)

        predictions = model.predict(inputs)
        if method == 'mean':
            p = numpy.mean(predictions, axis=0)
            assert len(p) == 10
            out.append(p)
        elif method == 'majority':
            votes = numpy.argmax(predictions, axis=1)
            p = numpy.bincount(votes) / len(votes)
            out.append(p)

    ret = numpy.stack(out)
    assert len(ret.shape) == 2, ret.shape
    assert ret.shape[0] == len(out), ret.shape
    assert ret.shape[1] == 10, ret.shape # classes

    return ret

def load_history(jobs_dir, job_id):


    matching = [ d for d in os.listdir(jobs_dir) if job_id in d ]
    assert len(matching) == 9, "Expected 9 folds, found {} matching {}".format(len(matching), job_id)

    dataframes = []
    
    for job_dir in matching:
        fold = job_dir.split('-fold')[1]
        hist_path = os.path.join(jobs_dir, job_dir, 'history.csv')
    
        df = pandas.read_csv(hist_path)
        del df['Unnamed: 0']
        df['epoch'] = df.epoch + 1
        df['fold'] = fold
        
        models = []
        for fname in os.listdir(os.path.join(jobs_dir, job_dir)):
            if fname.endswith('model.hdf5'):
                models.append(fname)
        
        models = sorted(models)
        assert models[0].startswith('e01')
        assert models[len(models)-1].startswith('e{:0d}'.format(len(models)))
        df['model'] = [ os.path.join(jobs_dir, job_dir, m) for m in models ]
        dataframes.append(df)

    df = pandas.concat(dataframes)
    return df

def test_load_history():

    jobs_dir = '../../jobs'
    job_id = 'sbcnn44k128aug-20190227-0220-48ba'
    df = load_history()

def pick_best(history, n_best=1):

    def best_by_loss(df):
        return df.sort_values('val_loss', ascending=True).head(n_best)
    return history.groupby('fold').apply(best_by_loss)

    # best_by_loss.plot(y='val_acc', kind='bar', subplots=True)


def evaluate(models, folds, test, predictor):

    val_scores = []
    test_scores = []

    def score(model, data):
        y_true = data.classID
        p = predictor(model, data)
        y_pred = numpy.argmax(p, axis=1)
        # other metrics can be derived from confusion matrix
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


def test_predict_windowed():

    from sklearn.metrics import accuracy_score
    t = test[0:10]

    sbcnn16k32_settings = dict(
        feature='mels',
        samplerate=16000,
        n_mels=32,
        fmin=0,
        fmax=8000,
        n_fft=512,
        hop_length=256,
        augmentations=5,
    )

    def load_sample32(sample):
        return train.load_sample(sample, sbcnn16k32_settings, window_frames=72, feature_dir='../../scratch/aug')

    mean_m = predict_windowed(sbcnn16k32_settings, model, t, loader=load_sample32, method='mean')
    accuracy_score(t.classID, mean_m)

def parse(args):

    import argparse

    parser = argparse.ArgumentParser(description='Test trained models')

    a = parser.add_argument

    a('--jobs', dest='jobs_dir', default='./jobs',
        help='%(default)s')
    a('--out', dest='out_dir', default='./results',
        help='%(default)s')
    a('--features', dest='features_dir', default='./features',
        help='%(default)s')


    a('--experiment', dest='experiment', default='',
        help='%(default)s')



    parsed = parser.parse_args(args)

    return parsed

def main():
    
    args = parse(sys.argv[1:])

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    history = load_history(args.jobs_dir, args.experiment)
    best = pick_best(history)


    data_path = '../../data'
    urbansound8k.default_path = os.path.join(data_path, 'UrbanSound8K')
    urbansound8k.maybe_download_dataset(data_path)
    data = urbansound8k.load_dataset()

    folds, test = urbansound8k.folds(data)

    print('Loading models...')
    models = best['model'].apply(lambda p: keras.models.load_model(p))

    print('Testing models...')

    sbcnn16k32_settings = dict(
        feature='mels',
        samplerate=16000,
        n_mels=30,
        fmin=0,
        fmax=8000,
        n_fft=512,
        hop_length=256,
        augmentations=0,
    )

    frames=72

    settings = sbcnn16k32_settings
    def load_sample(sample):
        # FIXME: unhardcode
        return train.load_sample(sample, settings, window_frames=frames, feature_dir=args.features_dir)

    def predict(model, data):
        return predict_voted(settings, model, data, loader=load_sample, window_frames=frames, method='mean', overlap=0.5)

    val, test = evaluate(models, folds, test, predictor=predict)

    # FIXME: put experiment name into filename
    results_path = os.path.join(args.out_dir, 'confusion.npz')
    numpy.savez(results_path, val=val, test=test)

    print('Wrote to', results_path)

if __name__ == '__main__':
    main()


