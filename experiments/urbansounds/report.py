
import numpy
import urbansound8k

import seaborn as sns
import matplotlib.pyplot as plt     

def plot_confusion(cm, classnames, normalize=False):

    if normalize:
        cm = cm_normalize(cm)

    fig, ax = plt.subplots(1, figsize=(10,8))
    sns.heatmap(cm, annot=True, ax=ax);

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

def main():

    cm = numpy.load('results/confusion.npz')

    print(list(cm.keys()))

    val, test = cm['val'], cm['test']

    print('v', val.shape, test.shape)

    classnames = urbansound8k.classnames
    val_fig = plot_confusion(100*numpy.mean(val, axis=0), classnames, normalize=True)
    test_fig = plot_confusion(100*numpy.mean(test, axis=0), classnames, normalize=True) 

    c_acc = cm_class_accuracy(numpy.mean(val, axis=0))
    print(c_acc, numpy.mean(c_acc)) 

    folds_acc = [ cm_accuracy(val[f]) for f in range(0, len(val)) ]

    print('acc', folds_acc, numpy.mean(folds_acc))

    val_fig.savefig('val.cm.png')
    test_fig.savefig('test.cm.png')

    print('wrote')

if __name__ == '__main__':
    main()
