

import pandas
import numpy
import matplotlib
from matplotlib import pyplot as plt


def plot():
    models = pandas.read_csv('models.csv')
    
    fig, ax = plt.subplots(1)
    print(models.head(10))
    print(models.index)

    #models.plot(kind='scatter', ax=ax, x='parameters', y='accuracy')

    n_labels = len(models['name'])

    colors = matplotlib.cm.rainbow(numpy.linspace(0, 1, n_labels))     # create a bunch of colors

    for i, r in models.iterrows():
        ax.plot(r['parameters']/1000, r['accuracy'], 'o', label=r['name'],
                markersize=5, color=colors[i], linewidth=0.1)

    ax.legend(loc='best')

    fig.savefig('perf.png')


if __name__ == '__main__':
    plot()
