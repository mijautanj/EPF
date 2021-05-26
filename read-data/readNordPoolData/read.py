from readFunctions import readOneYear, readOneYearHydro, CETtoUTC, calVars, upsampleHydro 
import pandas as pd
from functools import reduce


BASE_URL = ['../../data/', '_hourly.xls','_hourly_sek.xls','_weekly.xls']

infoDict = {
    "price" : ['-price','elspotPrices/elspot-prices_'], #historical prices
    "SEprod" : ['-prod', 'productionSEareas/production-se-areas_'], #production price areas
    "SEcons" : ['-cons','consumptionSEareas/consumption-se-areas_'], #consumption price areas
    "CountriesProd" : ['-prod','productionCountries/production-per-country_'], #production per country
    "CountriesCons" : ['-cons','consumptionCountries/consumption-per-country_'], #consumption per country
    "SEprodProg" : ['-prod-prog', 'productionPrognosis/production-prognosis_'], #production prognosis price areas
    "SEconsProg" : ['-cons-prog', 'consumptionPrognosis/consumption-prognosis-se_'], #consumption prognosis price areas
    "windProg" : ['-wind-prog', 'windpowerPrognosis/wind-power-se-prognosis_'], #wind prognosis price areas
    "hydroRes" : ['-hydro-res', 'hydroReservoir/hydro-reservoir_'] #hydro reservior levels countries
}


#Creating empty DateTime df for keeping same structure in final df
rng = pd.date_range(start='2014-12-28', end='2021-01-05', freq='H', closed='left')
dates = pd.DataFrame(rng, columns=['DateTime']) 

#Creating empty list with DateTime
data_frames = [dates]

for key, values in infoDict.items():
    suffix = values[0] #extracting suffix from dict
    spec_path = values[1] #extracting specific path from dict
    allYearsSet = [] #Empty list with all years data for one variable

    for i in range(2015,2021):
        if key == "price":
            PATH = BASE_URL[0] + spec_path + str(i) + BASE_URL[2] #File ending of prices
            subset = readOneYear(PATH)
        elif key == "hydroRes":
            PATH = BASE_URL[0] + spec_path + str(i) + BASE_URL[3] #File ending of hydroRes
            subset = readOneYearHydro(PATH)
        elif key == "SEprodProg":
            PATH = BASE_URL[0] + spec_path + str(i) + BASE_URL[1]  #File ending of rest
            subset = readOneYear(PATH,'A,B,H:K')
        else:
            PATH = BASE_URL[0] + spec_path + str(i) + BASE_URL[1] #File ending of rest
            subset = readOneYear(PATH)
        
        print(PATH)
        subset = subset.add_suffix(suffix)
        subset.rename(columns={"DateTime"+suffix:"DateTime"}, inplace=True)
        allYearsSet.append(subset)

    completeSet = pd.concat(allYearsSet, ignore_index=True) #concatenating all years for each variable
    completeSet = CETtoUTC(completeSet) #Converting to UTC
    data_frames.append(completeSet) #adding to large df list
    

#Merge all dataframes together in one large on "DateTime"
df_merged = reduce(lambda  left,right: pd.merge(left,right, on=['DateTime'], how='outer'), data_frames)
df_merged.sort_values(by='DateTime', inplace=True, ignore_index=True)


#Extrapolating hydro reservoir values
hydro_cols = [col for col in df_merged.columns if 'hydro' in col] #Finding columns with hydro levels
for i in hydro_cols:
    t = upsampleHydro(df_merged[['DateTime',i]]) 
    df_merged[i] = t[i]


#Inserting lagged price and moving average with 3h and 12h window
price_cols = [col for col in df_merged.columns if 'price' in col] #Finding columns with hydro levels
#lagged = []
for name in price_cols:
    df_merged[name+'-lag-24h']=df_merged[name].shift(24)
    df_merged[name+'-lag-48h']=df_merged[name].shift(48)
    df_merged[name+'-lag-168h']=df_merged[name].shift(168)
    df_merged[name+'-avg-3h'] = df_merged[name].rolling(window=3).mean()
    df_merged[name+'-avg-12h'] = df_merged[name].rolling(window=12).mean()



#Removing lines with at least 26 empty cells
df_merged.dropna(thresh=26, inplace=True)
df_merged.reset_index(drop=True, inplace=True)

#Deleting these two columns because they often lack values
df_merged.drop(['Nordic-prod','Nordic-cons'], axis=1, inplace=True)

#Removing duplicate columns
cols = [c for c in df_merged.columns if "_y" in c]
df_merged.drop(cols, inplace=True, axis=1)

df_merged.columns = df_merged.columns.str.replace('_x$', '', regex=True)


#Inserting calendar variables
df_merged = calVars(df_merged)

#Saving as excel file
df_merged.to_excel('NPdataSetLag.xlsx', engine='xlsxwriter')
