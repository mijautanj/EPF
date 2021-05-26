
import pandas as pd
import matplotlib.pyplot as plt

PATH = './data/elspotPrices/elspot-prices_2019_hourly_sek.xls'
NAME = PATH.split(".")[0]

#Todo: add dtype
df = pd.read_excel(PATH, usecols='A:G', header=2)  
df.rename(columns={ df.columns[0]: 'Date' }, inplace = True )

print(df.head())

labels = df.columns[3:7]
df.plot(x ='Date', y=labels, ylabel = "SEK/MWh", title=NAME, subplots='true')
plt.show()