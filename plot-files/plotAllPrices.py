import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams.update({'font.size': 20})

allData = []
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

    
    allData.append(subset)



#Plot all data for all years in one plot
plot=False
region='SE1'
if plot:
    dummy = 0
    for j in range (2):
        fig, axes = plt.subplots(nrows=2, ncols=2) 
        for i, ax in enumerate(axes.flat):
            allData[dummy].plot(x='Month', y=region, ax=ax, title=str(years[dummy]), legend=True, ylim=[-50,2750])
            dummy += 1

        plt.suptitle('Electricity price for ' + region + ' price area') #Main title
        fig.text(0.04, 0.5, 'SEK/MWh', va='center', rotation='vertical') #Getting a global label
        plt.show()



#Show training, validation and test set for all areas
PRICEAREA = 'SE3'
plot2=True
if plot2:
    fullset = pd.concat(allData, ignore_index=True)[['Date',PRICEAREA]]
    #trainset = pd.concat(allData[:6], ignore_index=True)[['Date',PRICEAREA]]
    #valset = pd.concat(allData[6:7], ignore_index=True)[['Date',PRICEAREA]]
    #testset = pd.concat(allData[7:], ignore_index=True)[['Date',PRICEAREA]]

    ax = fullset.plot(x='Date', color='#174D7F',figsize=(16,5.5))
    ax.get_legend().remove()
    #ax = trainset.plot(x='Date', color='#174D7F',figsize=(16,5.5))
    #valset.plot(x='Date', color='#FF7000', ax=ax)
    #testset.plot(x='Date', color='#1F7000', ax=ax)
    #ax.legend(['Training set','Validation set','Test set'])
    plt.title('Electricity price for ' + PRICEAREA + ' price area')
    plt.ylabel('SEK/MWh')
    plt.tight_layout() 
    plt.savefig(PRICEAREA + '.png')
    plt.show()


#Full dataset, average over all years and all price areas
plot3=False
if plot3:
    values = []
    for year in allData:
        year['Mean'] = year.mean(axis=1) #Taking mean of all price areas for each year
        values.append(year[['Date', 'Month', 'Mean']])

    means = pd.concat(values) #All mean values for years 2013-2020

    means['index1'] = means.index #Get indexvalues.
    months=allData[-1]['Month'] #Get the months from 2020

    index_avgs = means.groupby(['index1']).mean().reset_index()


    both = [ months,index_avgs]
    plottis = pd.concat(both, axis=1)

    plottis.plot( y='Mean', color='#174D7F', label='Mean price', figsize=(16,5.5))
    plt.title('The average year over all price areas and all years (2013-2020)')
    plt.ylabel('SEK/MWh')

    plt.tight_layout() 
    plt.savefig('mean.png')
    plt.show()
 
