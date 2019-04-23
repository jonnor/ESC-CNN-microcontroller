
import sys
import pandas
import numpy
import yaml

df = pandas.read_csv('img/results.csv')
print(df)

#     'duration_avg': 'CPU (seconds)',
#    'maccs_frame': 'Compute (MAC)',
table = pandas.DataFrame({
    'Model': df.nickname,
    'CPU (%)': (df.utilization * 100).astype(int),
    'RAM (kB)': (df.ram_usage_max/1024).astype(int),
    'Accuracy': (df.test_acc_mean * 100).round(1),
    'Accuracy (foreground)': (df.foreground_test_acc_mean * 100).round(1),
#    'Accuracy (background)': (df.background_test_acc_mean * 100).round(1),
}, index=df.index)

out = table.to_latex(header=True, index=True)
print(out)

outpath = sys.argv[1] 
with open(outpath, 'w') as f:
    f.write(out)
