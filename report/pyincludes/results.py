
import sys
import pandas
import numpy
import yaml

df = pandas.read_csv('results/results.csv')
print(df)

#width_variations = df.nickname.str.startswith('Stride-DS-5x5-')
#df = df[width_variations != True]
df = df.sort_values('nickname', ascending=True)

def accuracies(df, col):
    mean = df[col+'_mean'] * 100
    std = df[col+'_std'] * 100

    fmt = [ "{:.1f}% +-{:.1f}".format(*t) for t in zip(mean, std) ]
    return fmt

def cpu_use(df):
    usage = (df.utilization * 1000 * 1/df.classifications_per_second).astype(int)
    return ["{:d} ms".format(i).ljust(3) for i in usage]

table = pandas.DataFrame({
    'Model': df.nickname,
    'CPU use': cpu_use(df),
    'Accuracy': accuracies(df, 'test_acc'),
    'FG Accuracy': accuracies(df, 'foreground_test_acc'),
    'BG Accuracy': accuracies(df, 'background_test_acc'),
}, index=df.index)


out = table.to_latex(header=True, index=False, column_format='lrrrr')
out = out.replace('+-', '$\pm$') # XXX: Latex gets mangled by to_table it seems
print(out)

outpath = sys.argv[1] 
with open(outpath, 'w') as f:
    f.write(out)
