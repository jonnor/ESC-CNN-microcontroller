
import os.path
import sys
import json
import math

import numpy
import seaborn
import matplotlib
import matplotlib.pyplot as plt     
import pandas

from . import common, urbansound8k

groups = {
    'social_activity': [ 'street_music', 'children_playing', 'dog_bark' ],
    'construction': [ 'drilling', 'jackhammer' ],
    'road_noise': [ 'engine_idling', 'car_horn', 'siren' ],
    'domestic_machines': [ 'air_conditioner' ],
    'danger': [ 'gun_shot' ],
}

def plot_confusion(cm, classnames, normalize=False, percent=False):

    fmt = '.2f'
    if normalize:
        cm = cm_normalize(cm)
    if percent:
        cm = cm_normalize(cm)*100
        fmt = ".1f"

    fig, ax = plt.subplots(1, figsize=(10,8))
    seaborn.heatmap(cm, annot=True, ax=ax, fmt=fmt);

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix') 
    ax.xaxis.set_ticklabels(classnames, rotation=60)
    ax.yaxis.set_ticklabels(classnames, rotation=0)
    return fig

def cm_normalize(cm):
    rel = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    return rel

def cm_class_accuracy(cm):
    rel = cm_normalize(cm)
    return numpy.diag(rel)

def cm_accuracy(cm):
    correct = numpy.sum(numpy.diag(cm))
    total = numpy.sum(numpy.sum(cm, axis=1))
    return correct/total

def grouped_confusion(cm, groups):
    groupnames = list(groups.keys())
    groupids = list(range(len(groupnames)))
    group_cm = numpy.zeros(shape=(len(groupids), len(groupids)))
    groupid_from_classid = {}
    for gid, name in zip(groupids, groupnames):
        classes = groups[name]
        for c in classes:
            cid = urbansound8k.classes[c]
            groupid_from_classid[cid] = gid

    for true_c in range(cm.shape[0]):
        for pred_c in range(cm.shape[1]):
            v = cm[true_c, pred_c]
            true_g = groupid_from_classid[true_c]
            pred_g = groupid_from_classid[pred_c]
            group_cm[true_g, pred_g] += v

    return group_cm, groupnames


def print_accuracies(accs, title):

    m = numpy.mean(accs)
    s = numpy.std(accs)
    print('{} | mean: {:.3f}, std: {:.3f}'.format(title, m, s))
    [ print("{:.3f}".format(v), end=',') for v in accs ]
    print('\n')


def get_accuracies(confusions):
    accs = [ cm_accuracy(confusions[f]) for f in range(0, len(confusions)) ]
    assert len(accs) == 10, len(accs) 
    return pandas.Series(accs) 

def plot_accuracy_comparison(experiments, ylim=(0.60, 0.80), figsize=(12, 4)):

    df = experiments.copy()
    df.index = experiments.nickname
    acc = df.confusions_test.apply(get_accuracies).T
    fig, ax = plt.subplots(1, figsize=figsize)
    
    acc.boxplot(ax=ax)

    ax.set_ylabel('Accuracy')
    ax.set_ylim(ylim)

    #ax.set_xticks(experiments.nickname)
    #ax.set_xlabel('Model')

    return fig

def plot_accuracy_vs_compute(experiments, ylim=(0.60, 0.80),
                                perf_metric='utilization', figsize=(12,8)):
    # TODO: color experiment groups
    # TODO: add error bars?

    acc = experiments.confusions_test.apply(get_accuracies).T
    df = experiments.copy()
    df['accuracy'] = acc.mean()
    numpy.testing.assert_allclose(df.test_acc_mean, df.accuracy)
    df['experiment'] = df.index

    fig, ax = plt.subplots(1, figsize=figsize)
    df.plot.scatter(ax=ax, x=perf_metric, y='accuracy', logx=True)

    # Y axis
    ax.set_ylim(ylim)
    ax.set_ylabel('Accuracy')

    if perf_metric == 'utilization':
        # mark feasible regions
        alpha = 0.2
        ax.axvspan(xmin=0, xmax=0.5, alpha=alpha, color='green')
        ax.axvspan(xmin=0.5, xmax=1.0, alpha=alpha, color='orange')
        xmax = df.utilization.max()*2.0
        ax.axvspan(xmin=1.0, xmax=xmax, alpha=alpha, color='red')
        ax.set_xlim(ax.get_xlim()[0], xmax)
        
        def format_utilization(tick_val, tick_pos):
            return '{:d}%'.format(int(tick_val*100))

        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_utilization))
        ax.set_xlabel('CPU utilization')

    # Add markers
    def add_labels(row):
        xy = row[perf_metric], row.accuracy
        label = "{}".format(row.nickname) 
        ax.annotate(label, xy,
                    xytext=(5,20),
                    textcoords='offset points',
                    size=10,
                    rotation=25,
                    color='darkslategrey')
    df.apply(add_labels, axis=1)

    return fig


def load_results(input_dir, confusion_suffix='.confusion.npz'):

    files = os.listdir(input_dir)
    files = [ f for f in files if f.endswith(confusion_suffix) ]
    names = [ f.rstrip(confusion_suffix) for f in files ]

    df = pandas.DataFrame({
        'experiment': names,
        'result_path': [ os.path.join(input_dir, f) for f in files ],
    })
    df = df.set_index('experiment')

    def load_confusions(row):
        path = row['result_path']
        results = numpy.load(path)
        for k, v in results.items():
            row['confusions_'+k] =v
        return row

    df = df.apply(load_confusions, axis=1)

    # model statistics
    stat_path = os.path.join(input_dir, 'stm32stats.csv')
    model_stats = pandas.read_csv(stat_path, dtype={'experiment': str})
    model_stats.set_index('experiment', inplace=True)
    df = df.join(model_stats)

    return df

def load_device_results(results_dir, suffix='.device.json'):

    frames = []   
    for filename in os.listdir(results_dir):
        if filename.endswith(suffix):
            experiment = filename.rstrip(suffix)
            p = os.path.join(results_dir, filename)
            with open(p, 'r') as f:
                contents = f.read()
                contents = contents.replace("'", '"') # hack invalid JSON
                d = json.loads(contents)
                d['experiment'] = experiment
                df = pandas.DataFrame([d])
                frames.append(df)

    df = pandas.concat(frames)
    df.set_index('experiment', inplace=True)
    return df

def parse(args):
    import argparse

    parser = argparse.ArgumentParser(description='Test trained models')
    a = parser.add_argument

    common.add_arguments(parser)

    a('--run', dest='run', default='',
        help='%(default)s')
    a('--results', dest='results_dir', default='./data/results',
        help='%(default)s')
    a('--out', dest='out_dir', default='./report/results',
        help='%(default)s')
    a('--skip-device', dest='skip_device', action='store_true')

    parsed = parser.parse_args(args)

    return parsed


def main():
    args = parse(None)

    input_dir = os.path.join(args.results_dir, args.run)
    out_dir = args.out_dir

    df = load_results(input_dir)

    df['val_acc_mean'] = df.confusions_val.apply(get_accuracies).mean(axis=1)
    df['test_acc_mean'] = df.confusions_test.apply(get_accuracies).mean(axis=1)
    df = df.sort_index()

    # TODO: add std-dev
    df['foreground_val_acc_mean'] = df.confusions_val_foreground.apply(get_accuracies).mean(axis=1)
    df['foreground_test_acc_mean'] = df.confusions_test_foreground.apply(get_accuracies).mean(axis=1)
    df['background_test_acc_mean'] = df.confusions_test_background.apply(get_accuracies).mean(axis=1)


    #df['grouped_test_acc_mean'] = grouped_confusion(df.confusions_test, groups).apply(get_accuracies).mean(axis=1)
    #df['grouped_foreground_test_acc_mean'] = grouped_confusion(df.confusions_test_foreground, groups).apply(get_accuracies).mean(axis=1)

    # FIXME: this should come from the results
    models = pandas.read_csv('models.csv')
    models.index = [ str(i) for i in models.index ]
    df = df.join(models)

    # TODO: also add experiment settings
    # FIXME: unhardcode
    df.voting_overlap = 0.0
    df.window_length = 0.72
    df['classifications_per_second'] = 1 / (df.window_length * (1-df.voting_overlap))

    # device performance
    if not args.skip_device:
        dev = load_device_results(input_dir)
        df = df.join(dev)
        numpy.testing.assert_allclose(df.macc, df.maccs_frame)

        df['utilization'] = df.duration_avg * df.classifications_per_second
    else:
        df['utilization'] = 0.0

    print('res\n', df[['nickname', 'maccs_frame', 'test_acc_mean', 'val_acc_mean']])

    def save(fig, name):
        p = os.path.join(out_dir, name)
        fig.savefig(p, bbox_inches='tight', pad_inches=0)


    # Split the variations from all models
    width_variations = df.nickname.str.startswith('Stride-DS-5x5-')
    fig = plot_accuracy_comparison(df[width_variations != True])
    save(fig, 'models_accuracy.png')

    perf_metric = 'maccs_frame' if args.skip_device else 'utilization'
    fig = plot_accuracy_vs_compute(df, perf_metric=perf_metric)
    save(fig, 'models_efficiency.png')


    classnames = urbansound8k.classnames
    best = df.sort_values('test_acc_mean', ascending=False).head(1).iloc[0]

    confusion_matrix = numpy.mean(best.confusions_test, axis=0)
    fig = plot_confusion(confusion_matrix, classnames, percent=True)
    save(fig, 'confusion_test.png')

    cm = numpy.mean(best.confusions_test_foreground, axis=0)
    group_cm, groupnames = grouped_confusion(cm, groups)
    fig = plot_confusion(group_cm, groupnames, percent=True)
    save(fig, 'grouped_confusion_test_foreground.png')

    confusion_columns = [ c for c in df.columns if c.startswith('confusion') ]
    df = df.drop(confusion_columns, axis=1)
    df.to_csv(os.path.join(out_dir, 'results.csv'))

if __name__ == '__main__':
    main()
