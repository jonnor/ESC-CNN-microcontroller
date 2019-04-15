
import sys
from microesc import urbansound8k
import pandas
import numpy

data = urbansound8k.load_dataset()
by_class = data.groupby('class')  
foreground_ratio = by_class.apply(lambda r: numpy.mean(r['salience'] == 1))

table = pandas.DataFrame({
    'Samples': by_class.count()['classID'],
    'Average duration': by_class.apply(lambda r: '%.2fs' % (r.end-r.start).mean()),
    'In foreground': [ "{}%".format(int(100*r)) for r in foreground_ratio ]
})
out = table.to_latex(header=True, index=True)
print(out)

outpath = sys.argv[1] 
with open(outpath, 'w') as f:
    f.write(out)
