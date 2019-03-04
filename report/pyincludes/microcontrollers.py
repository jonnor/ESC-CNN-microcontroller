
import pandas
import os.path

here = os.path.dirname(__file__)
data = pandas.read_csv(os.path.join(here, '../microcontrollers.csv'))

data = data.rename(columns={
    'name': 'Name',
    'architecture': 'Architecture',
    'sram_kb': 'RAM (kB)',
    'flash_kb': 'Flash (kB)',
    'cpufreq_mhz': 'CPU (MHz)',
    'price_1k_usd': 'Price (USD)',
})
table = data.to_latex(header=True, index=False)
print(table)
