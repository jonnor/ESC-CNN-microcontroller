
import os.path
import sys

import numpy
import seaborn
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
    assert len(accs) == 9, len(accs) 
    return pandas.Series(accs) 

def plot_accuracy_comparison(experiments, ylim=(0.65, 0.85)):
    
    acc = experiments.confusions_test.apply(get_accuracies).T
    fig, ax = plt.subplots(1)
    acc.boxplot(ax=ax)
    ax.set_ylim(ylim)

    return fig

def plot_accuracy_vs_compute(experiments, ylim=(0.65, 0.75)):
    # TODO: color experiment groups
    # TODO: add error bars?

    acc = experiments.confusions_test.apply(get_accuracies).T
    df = experiments.copy()
    df['accuracy'] = acc.mean()

    fig, ax = plt.subplots(1)
    df.plot.scatter(ax=ax, x='maccs_frame', y='accuracy')
    ax.set_ylim((0.65, 0.80))

    return fig


def load_results(input_dir, confusion_suffix='.confusion.npz'):

    files = os.listdir(input_dir)
    files = [ f for f in files if f.endswith(confusion_suffix) ]
    names = [ f.rstrip(confusion_suffix) for f in files ]

    df = pandas.DataFrame({
        'experiment': names,
        'result_path': [ os.path.join(input_dir, f) for f in files ],
    })

    def load_confusions(row):
        path = row['result_path']
        results = numpy.load(path)
        for k, v in results.items():
            row['confusions_'+k] =v
        return row


    stat_path = os.path.join(input_dir, 'stm32stats.csv')
    model_stats = pandas.read_csv(stat_path, index_col='experiment')

    df = df.apply(load_confusions, axis=1)
    df = df.join(model_stats)

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
    a('--out', dest='out_dir', default='./report/img',
        help='%(default)s')

    parsed = parser.parse_args(args)

    return parsed


def main():
    args = parse(None)

    input_dir = os.path.join(args.results_dir, args.run)
    out_dir = args.out_dir

    df = load_results(input_dir)
    acc = df.confusions_test.apply(get_accuracies).mean()
    df['test_acc_mean'] = acc

    print('res\n', df[['experiment', 'test_acc_mean', 'maccs_frame']])

    def save(fig, name):
        p = os.path.join(out_dir, name)
        fig.savefig(p, bbox_inches='tight', pad_inches=0)

    fig = plot_accuracy_comparison(df)
    save(fig, 'models_accuracy.png')

    fig = plot_accuracy_vs_compute(df)
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

if __name__ == '__main__':
    main()
