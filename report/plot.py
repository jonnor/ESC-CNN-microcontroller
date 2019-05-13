
import sys
import os.path
from functools import partial

import pandas
from matplotlib import pyplot as plt
import librosa.display
import numpy

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
    table['MACC / second'] = [ "{} M".format(int(v/1e6)) for v in df.macc_s ]
    table['Model parameters'] = [ "{} k".format(int(v/1e3)) for v in df.params ]
    #table['Data augmentation'] = df.augmentation
    table = table.sort_values('Accuracy (%)', ascending=False)
    #table['Time resolution (ms)'] = (df.t_step*1000).astype(int)
    
    return table.to_latex(column_format='lrrr')
    
def plot_models(data_path, figsize=(12,4), max_params=128e3, max_maccs=4.5e6):
    df = logmel_models(data_path)
    
    fig, ax = plt.subplots(1, figsize=figsize)

    check_missing(df, 'accuracy')
    check_missing(df, 'kparams')
    check_missing(df, 'mmacc')

    df.plot.scatter(x='params', y='macc_s', logx=True, logy=True, ax=ax)
    ax.set_xlabel('Model parameters')
    ax.set_ylabel('MACC / second')
    
    # highlight feasible region
    feasible_x = max_params
    feasible_y = max_maccs
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

    fig.tight_layout()

    return fig

urbansound8k_examples = {
    'air_conditioner': ['fold9/75743-0-0-17.wav', 'fold9/75743-0-0-17.wav' ],
    'car_horn': ['fold7/34241-1-2-0.wav'],
    'children_playing': ['fold8/204526-2-0-166.wav', 'fold9/60935-2-0-4.wav'],
    'dog_bark': ['fold3/52077-3-0-8.wav', 'fold4/47926-3-2-0.wav'],
    'drilling': ['fold6/167701-4-9-0.wav', 'fold3/103199-4-2-3.wav'],
    'engine_idling': ['fold10/102857-5-0-19.wav', 'fold7/209992-5-2-42.wav'],
    'gun_shot': ['fold4/7064-6-4-0.wav', 'fold3/148838-6-0-0.wav'],
    'jackhammer': ['fold1/180937-7-1-1.wav', 'fold10/162134-7-13-3.wav'],
    'siren': ['fold4/24347-8-0-36.wav', 'fold10/93567-8-0-18.wav'],
    'street_music': ['fold7/157940-9-0-6.wav', 'fold1/155202-9-0-126.wav']
}

def flatten(list):
    out = []
    for x in list:
        for y in x:
            out.append(y)
    return out

def plot_spectrogram(f, ax=None, cmap=None):
    y, sr = librosa.load(f, sr=44100)

    fig = None
    if not ax:
        fig, ax = plt.subplots(1, figsize=(16,4))

    S = numpy.abs(librosa.stft(y))
    S = librosa.amplitude_to_db(S, ref=numpy.max)

    kwargs = dict(
        ax=ax, y_axis='log', x_axis='time', sr=sr,
    )
    if cmap is not None:
        kwargs['cmap'] = cmap
    librosa.display.specshow(S, **kwargs)
    return fig

def plot_spectrograms(files, titles, out=None):
    assert len(files) == len(titles)
    
    fig, axs = plt.subplots(2, len(files)//2, sharex=True, figsize=(16,6))
    axs = flatten(axs)
    
    for i, (path, title, ax) in enumerate(zip(files, titles, axs)):
        plot_spectrogram(path, ax=ax)
        ax.set_title(title)
        if i != 0 and i != len(files)/2:
            ax.set_ylabel('')
            ax.set_yticks([])
        if i < len(files)/2:
            ax.set_xlabel('')
    
    if out:
        fig.savefig(out, bbox_inches='tight', pad_inches=0)
    return fig

def plot_examples(examples):
    examples = urbansound8k_examples
    here = os.path.dirname(__file__)
    base = os.path.join(here, '../microesc/../data/datasets/UrbanSound8K/audio/')
    paths = [ os.path.join(base, e[0]) for e in examples.values() ]
    fig = plot_spectrograms(paths, examples.keys())
    return fig


plots = {
    'urbansound8k-existing-models-logmel.png': partial(plot_models, 'urbansound8k-existing.csv'),
    'urbansound8k-existing-models-logmel.tex': partial(model_table, 'urbansound8k-existing.csv'),
    'urbansound8k-examples.png': partial(plot_examples, urbansound8k_examples),
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
        out.savefig(out_path, bbox_inches='tight')
    elif ext == '.tex':
        with open(out_path, 'w') as f:
            f.write(out)
    else:
        raise ValueError('Unknown extension {}'.format(ext))

if __name__ == '__main__':
    main()
