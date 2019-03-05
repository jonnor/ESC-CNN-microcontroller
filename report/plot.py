
import sys
import os.path
from functools import partial

import pandas
from matplotlib import pyplot as plt

def check_missing(df, field, name='name'):
    missing = df[df[field].isna()]
    if len(missing):
        print('WARN. Missing "{}" for {}'.format(field, list(missing[name])))

def logmel_models(data_path):        
    df = pandas.read_csv(data_path)
    df = df[df['features'].str.contains('logmel')]
    
    df.index = df['name']
    df['params'] = df['kparams']*1e3
    df['window'] = df.frames * df.hop/df.samplerate
    df['t_step'] = df.hop/df.samplerate
    df['f_res'] = df.samplerate/df.bands
    df['macc_s'] = df['mmacc']*1e6 / df.window
    return df
    
def model_table(data_path):
    df = logmel_models(data_path)
    
    table = pandas.DataFrame()
    table['Accuracy (%)'] = df.accuracy*100
    table['Multiply-Adds / second'] = [ "{}M".format(int(v/1e6)) for v in df.macc_s ]
    table['Model parameters'] = [ "{}k".format(int(v/1e3)) for v in df.params ]
    table['Data augmentation'] = df.augmentation
    table = table.sort_values('Accuracy (%)', ascending=False)
    #table['Time resolution (ms)'] = (df.t_step*1000).astype(int)
    
    return table.to_latex()
    
def plot_models(data_path, figsize=(8,4)):
    df = logmel_models(data_path)
    
    fig, ax = plt.subplots(1, figsize=figsize)

    check_missing(df, 'accuracy')
    check_missing(df, 'kparams')
    check_missing(df, 'mmacc')

    df.plot.scatter(x='params', y='macc_s', logx=True, logy=True, ax=ax)
    ax.set_xlabel('Model parameters')
    ax.set_ylabel('Multiply-Adds / second')
    
    # highlight feasible region
    feasible_x = (512e3/4) * (1950/480)
    print('params', feasible_x)
    feasible_y = 8.0e6
    x = [ 0, feasible_x, feasible_x, 0 ]
    y = [ 0,  0, feasible_y, feasible_y ]
    ax.fill(x, y, color='green', alpha=0.5)
    
    linestyle = dict(color='black', linewidth=0.5)
    ax.axvline(feasible_x, **linestyle)
    ax.axhline(feasible_y, **linestyle)
    
    def add_labels(row):
        xy = row.params, row.macc_s
        label = "{}  {:.1f}%".format(row['name'], 100*row.accuracy) 
        ax.annotate(label, xy,
                    xytext=(5,40), 
                    textcoords='offset points',
                    size=12,
                    rotation=25,
                    color='darkslategrey')
    df.apply(add_labels, axis=1)

    return fig


plots = {
    'urbansound8k-existing-models-logmel.png': partial(plot_models, 'urbansound8k-existing.csv'),
    'urbansound8k-existing-models-logmel.tex': partial(model_table, 'urbansound8k-existing.csv'),
}


def main():

    plotname = os.path.basename(sys.argv[1])

    here = os.path.dirname(__file__)
    
    plot_func = plots.get(plotname, None)
    if not plot_func:
        sys.stderr.write("Plot {} not found. Supported: \n{}".format(plotname, plots.keys()))
        return 1

    out = plot_func()

    out_path = os.path.join(here, 'plots', plotname)
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    ext = os.path.splitext(plotname)[1]
    if ext == '.png':
        out.savefig(out_path)
    elif ext == '.tex':
        with open(out_path, 'w') as f:
            f.write(out)
    else:
        raise ValueError('Unknown extension {}'.format(ext))

if __name__ == '__main__':
    main()
