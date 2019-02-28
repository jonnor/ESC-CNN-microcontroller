
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

def predict_windowed(settings, model, samples, loader, window_frames, method='mean', overlap=0.5):
    sample_rate = settings['samplerate']
    Sample = collections.namedtuple('Sample', 'start end fold slice_file_name')
    frame_samples = settings['hop_length']

    out = []
    for _, sample in samples.iterrows():
        duration = sample.end - sample.start
        length = int(sample_rate * duration)
        windows = []
        
        for win in train.sample_windows(length, frame_samples, window_frames, overlap=overlap):
            chunk = Sample(start=win[0]/sample_rate,
                           end=win[1]/sample_rate,
                           fold=sample.fold,
                           slice_file_name=sample.slice_file_name)    
            d = loader(chunk)
            windows.append(d)

        inputs = numpy.stack(windows)
        predictions = model.predict(inputs)

        if method == 'mean':
            p = numpy.mean(predictions, axis=0)
            assert len(p) == 10
            out.append(p)
        elif method == 'majority':
            votes = numpy.argmax(predictions, axis=1)
            p = numpy.bincount(votes)
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


def score_folds(models, folds, test, predictor, top_k=3, overlap=0.5):

    val_acc_top = []
    test_acc_top = []
    scores = {
        'val_acc_win': [],
        'test_acc': [],
    }

    def score(model, data):
        y_true = keras.utils.to_categorical(data.classID)
        y_pred = predictor(model, data)

        #confusion = sklearn.metrics.confusion_matrix(y_true, y_pred)

        acc = numpy.array(keras.metrics.categorical_accuracy(y_pred, y_true))
        top = numpy.array(keras.metrics.top_k_categorical_accuracy(y_pred, y_true, k=top_k))
        return acc, top

    # validation
    for i, m in enumerate(models):
        validation_fold = folds[i][1]
        acc, top = score(m, validation_fold)
        val_acc_top.append(top)
        scores['val_acc_win'].append(acc)

    # test
    for i, m in enumerate(models):
        acc, top = score(m, test)
        test_acc_top.append(top)
        scores['test_acc'].append(acc)


    scores['val_acc_top{}_win'.format(top_k)] = val_acc_top
    scores['test_acc_top{}'.format(top_k)] = test_acc_top
    
    return pandas.DataFrame(scores)


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
        return predict_windowed(settings, model, data, loader=load_sample, window_frames=frames, method='mean', overlap=0.5)

    # TODO: should we use confusion matrix instead? Can derive all other stats from that?

    scores = score_folds(models, folds, test, predictor=predict, top_k=3)

    # FIXME: put experiment name into filename
    results_path = os.path.join(args.out_dir, 'results.csv')
    scores.to_csv(results_path)

    print('Wrote to', results_path)

if __name__ == '__main__':
    main()


