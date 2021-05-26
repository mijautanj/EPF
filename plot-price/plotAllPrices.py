import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams.update({'font.size': 20})


def readPrices():
    yearlyData = []
    years = []
    for i in range (2015,2021):
        print(i)
        years.append(i)
        PATH = '../data/elspotPrices/elspot-prices_' + str(i) + '_hourly_sek.xls'

        subset = pd.read_excel(PATH, usecols='A,B,D:G', header=2, dtype={'Date': str, 'Hours': str, 'SE1': float, 'SE2': float, 'SE3': float, 'SE4': float }) 
        subset.rename(columns={subset.columns[0]: 'Date' }, inplace = True ) #Rename the first column to Date
        subset['Date'] = pd.to_datetime(subset['Date']) #Setting the Date column to be of datetime data type
        subset['Month'] = subset['Date'].dt.strftime('%b') #Adding which month the dates belong
        subset['MonthDay'] = subset['Date'].dt.strftime('%m-%d') #Adding which month the dates belong
        yearlyData.append(subset)

    priceData = pd.concat(yearlyData, ignore_index=True)
    return priceData



def plotPrice(priceData, priceArea):
    fullset = priceData[['Date',priceArea]]
    ax = fullset.plot(x='Date', color='#174D7F',figsize=(16,5.5))
    ax.get_legend().remove()
    plt.title('Electricity price for ' + priceArea + ' price area')
    plt.ylabel('SEK/MWh')
    plt.tight_layout() 
    plt.savefig(priceArea + '.png')
    plt.show()



if __name__ == "__main__":
    priceData = readPrices()
    priceAreas = ['SE1', 'SE2','SE3','SE4']

    oneAxis = True
    grid = False
    autoCorrelation = False

    for i in priceAreas:
        plotPrice(priceData, i)
