import pandas as pd
from functools import reduce
from SMHI_API import SMHI_API_func

PARAMETERS = {
    "press": 9,
    "temp" : 1, 
    "wind" : 4,
    "hum" : 6,
    "cloud" : 16,
    }

#URL = 'https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/1/station/98230/period/corrected-archive/data.csv'

BASE_URL = ['https://opendata-download-metobs.smhi.se/api/version/1.0', 'period/corrected-archive/data.csv']

#Reading the Stations excel created with the active stations
PATH = '../../data/metobs_airtemperatureInstant_active_sites.xls'
stations = pd.read_excel(PATH, usecols='A:G', dtype={'Id': int, 'Namn': 'string', 'Latitud': float, 'Longitud': float, 'Höjd (m)': float, 'Aktiv': 'string', 'Elområde': int }) #Read excel, columns A:T, use row 2 as header (labels)
#stations = stations.iloc[0:5]


#Final, averaged list 
final = []

#For each variable (temp, wind, hum etc), do
for key,values in PARAMETERS.items():
    paramName = key #Current variable string name
    parameter = values #Current variable paramterer int number

    #dataframes, each index represents a price area
    dataframes = [[],[],[],[]]  
   
    #Creating empty datetime df for creating structure of end result
    rng = pd.date_range(start='2015-01-01', end='2021-01-01', freq='H', closed='left')
    dates = pd.DataFrame(rng, columns=['DateTime']) 
    #Creating dataframes list, each index is a list respresenting a price area. Ex: price area 1 is in dataframe[0]
    dataframes = [[dates],[dates],[dates],[dates]]

    #For each active station, do:
    for idx, station in stations['Id'].iteritems():
        #Current station read
        print("Station: ", stations['Namn'][idx])
        #Finding which price area the station belongs to
        elomrade = stations['Elområde'][idx]
        #Creating url string for SMHI API
        url = f'{BASE_URL[0]}/parameter/{parameter}/station/{station}/{BASE_URL[1]}'
        #Try accessing station and parameter from SMHI api
        try:
            df = SMHI_API_func(url)
            #Adding the read dataframe to the right index
            dataframes[elomrade-1].append(df)
            #Success case
            print("Station: ", stations['Namn'][idx],"\nSuccessfully read for parameter:", paramName)
        except:
            #Error case, API endpoint does not exists
            print("Station: ", stations['Namn'][idx],"\nNot found for parameter:", paramName)
            pass
    
    #dataframes now containing four lists filled with dataframes of timeseries values, one for each price area

    #Empty list for the average values of each price area
    meanDfs = []
    for i in range(len(dataframes)):
        #Merge together all values of variables belonging to same station (and parameter ex temp)
        df_merged = reduce(lambda  left,right: pd.merge(left,right, on=['DateTime'], how='outer'), dataframes[i])
        #Take mean of one DateTime point, taking mean on all the columns in each row
        df_merged['Mean'] = df_merged.mean(axis=1)
        meanDf = df_merged[['DateTime', 'Mean']].copy() #Extracting dataframe with only mean and DateTime
        meanDf.rename(columns={'Mean': 'SE' + str(i+1) + '-avg-' + paramName }, inplace = True) #Specifying column name instead of "Mean"
        meanDfs.append(meanDf) #appending the df containing mean parameter values of current price area

    #Merge all price area mean dataframes to one dataframe
    df_merged = reduce(lambda  left,right: pd.merge(left,right, on=['DateTime'], how='outer'), meanDfs)
    #Add merged df for this parameter (temp, wind, hum), do again for next parameter
    final.append(df_merged)

#Merging averaged df of all parameters into one big dataframe on DateTime index
df_merged = reduce(lambda  left,right: pd.merge(left,right, on=['DateTime'], how='outer'), final)
print(df_merged)
df_merged.to_excel('SMHIdataSet1.xlsx', engine='xlsxwriter')
