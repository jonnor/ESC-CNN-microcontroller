
import sys
import pandas
import numpy
import yaml

settings = yaml.load(open("../experiments/ldcnn20k60.yaml").read())

settings = pandas.DataFrame({
    'setting': list(settings.keys()),
    'value': list(settings.values()),
})
settings = settings.set_index('setting')
    
print(settings)

names = {
    'samplerate': 'Samplerate (Hz)',
    'n_mels': 'Melfilter bands',
    'n_fft': 'FFT length (samples)',
    'hop_length': 'FFT hop (samples)', # TODO: also time-resolution milliseconds
    'frames': 'Classification window', # TODO: also in milliseconds?
    'batch': 'Minibatch size',
    'epochs': 'Epochs',
    'train_samples': 'Training samples/epoch',
    'val_samples': 'Validation samples/epoch',
    'learning_rate': 'Learning rate',
    'nesterov_momentum': 'Nesterov momentum',
}

table = settings.loc[list(names.keys())]
table = table.rename(names)

out = table.to_latex(header=True, index=True)
print(out)

outpath = sys.argv[1] 
with open(outpath, 'w') as f:
    f.write(out)
