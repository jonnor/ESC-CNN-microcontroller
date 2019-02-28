
import urbansound8k
import pandas

data = pandas.DataFrame({'name': urbansound8k.classnames})
table = data.to_latex(header=True, index=True)
print(table)
