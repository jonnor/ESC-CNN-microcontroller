
import sys
import os.path
from functools import partial

import pandas
from matplotlib import pyplot as plt



def plot_models(data_path):

    fig, ax = plt.subplots(1)
    df = pandas.read_csv(data_path)

    # TODO: make it be flops instead
    df.plot(y='accuracy', x='params', ax=ax)    

    # TODO: highlight feasible region

    def add_labels(row):  
        ax.annotate(row.name, row.values,
                    xytext=(10,-5), 
                    textcoords='offset points',
                    size=18, 
                    color='darkslategrey')

        df.apply(annotate_df, axis=1)

    return fig

plots = {
    'urbansound8k-existing-models.png': partial(models('urbansound8k-existing.csv')),
}


def main():

    here = os.path.dirname(__file__)

    if len(sys.argv) > 1:
        plotname = sys.argv[1]
    
        plot_func = plots.get(plotname, None)
        if not plot_func:
            sys.stderr.write("Plot {} not found. Supported: \n{}".format(plotname, plots.keys()))
            return 1

        out = os.path.join(here, 'plots', plotname)
        fig = plot_func()
        fig.savefig(out)


if __name__ == '__main__':
    main()
