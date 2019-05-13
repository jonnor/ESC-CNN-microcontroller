
import sys
import pandas
import numpy
import yaml

df = pandas.read_csv('results/results.csv')
print(df)

# conv_block 	n_stages 	conv_size 	downsample_size 	filters
# TODO: move to model stats


#     'duration_avg': 'CPU (seconds)',

def strformat(fmt, series):
    return [fmt.format(i) for i in series]

df = df.sort_values('nickname', ascending=True)

#width_multiple = df.nickname.str.startswith('Stride-DS-5x5-')
#df = df.loc[width_multiple == False]


conv_shorthand = {
    'depthwise_separable': 'DS',
    'bottleneck_ds': 'BTLN-DS',
    'effnet': 'Effnet',
    'conv': 'standard',
}

def downsample_from_name(name):
    if name.startswith('Stride'):
        return 'stride'
    else:
        return 'maxpool'

table = pandas.DataFrame({
    'Model': df.nickname,
    'Downsample': [ "{} {}".format(downsample_from_name(n), s) for n, s in zip(df.nickname, df.downsample_size) ],
    'Convolution': [ conv_shorthand[i] for i in df.conv_block ],
    'L': df.n_stages,
    'F': df.filters,
    'MACC': strformat("{:d} K", (df.maccs_frame / 1000).astype(int)),
    'RAM': strformat("{:d} kB", (df.ram_usage_max/1024).astype(int)),
    'FLASH': strformat("{:d} kB", (df.flash_usage/1024).astype(int)),
}, index=df.index)

out = table.to_latex(header=True, index=False)
print(out)

outpath = sys.argv[1] 
with open(outpath, 'w') as f:
    f.write(out)
