import numpy
from matplotlib import pyplot as plt
from sklearn.metrics import log_loss

def plot_logloss(figsize=(6, 3)):
    fig, ax = plt.subplots(1, figsize=figsize)

    yhat = numpy.linspace(0.0, 1.0, 300)
    losses_0 = [log_loss([0], [x], labels=[0,1]) for x in yhat]
    losses_1 = [log_loss([1], [x], labels=[0,1]) for x in yhat]

    ax.plot(yhat, losses_0, label='true=0')
    ax.plot(yhat, losses_1, label='true=1')
    ax.legend()

    ax.set_ylim(0, 8)
    ax.set_xlim(0, 1)

    return fig

def main():
    fig = plot_logloss()
    fig.tight_layout()
    out = (__file__).replace('.py', '.png')
    fig.savefig(out, bbox_inches='tight')

if __name__ == '__main__':
    main()


