from readFunctions import readOneYearFuncFirst, readOneYearFuncOther, readOneYearProdProg, readOneYearHydroRes
import pandas as pd
from functools import reduce



def readAllYears():
    #Creating empty list for storing the dataframes
    price = []
    SEprod = []
    SEcons = []
    CountriesProd = []
    CountriesCons = []
    SEprodProg = []
    SEconsProg = []
    windProg = []
    hydroRes = []

    

    #Read yearly files
    for i in range (2015,2021):

        #Historical prices
        PATH = '../../../data/elspotPrices/elspot-prices_' + str(i) + '_hourly_sek.xls'
        subsetPrice = readOneYearFuncFirst(PATH) #One function which reads in and adds all calendar variables
        subsetPrice.columns = [col+'-price' if 'SE' in col or 'SYS' in col else col for col in subsetPrice.columns] #Adding suffix for SE and SYS cols
        price.append(subsetPrice)

        #SE-areas production
        PATH = '../../../data/productionSEareas/production-se-areas_' + str(i) + '_hourly.xls'
        subsetSEprod = readOneYearFuncOther(PATH).add_suffix("-prod") #Adding suffix for all read values
        subsetSEprod = subsetSEprod.rename(columns={"DateTime-prod":"DateTime"}) #Changing DateTime col name back
        SEprod.append(subsetSEprod)

        #SE-areas consumption
        PATH = '../../../data/consumptionSEareas/consumption-se-areas_' + str(i) + '_hourly.xls'
        subsetSEcons = readOneYearFuncOther(PATH).add_suffix("-cons")
        subsetSEcons = subsetSEcons.rename(columns={"DateTime-cons":"DateTime"})
        SEcons.append(subsetSEcons)

        #Countries production
        PATH = '../../../data/productionCountries/production-per-country_' + str(i) + '_hourly.xls'
        subsetCounProd = readOneYearFuncOther(PATH).add_suffix("-prod")
        subsetCounProd = subsetCounProd.rename(columns={"DateTime-prod":"DateTime"})
        CountriesProd.append(subsetCounProd)

        #Countries consumption
        PATH = '../../../data/consumptionCountries/consumption-per-country_' + str(i) + '_hourly.xls'
        subsetCounCon = readOneYearFuncOther(PATH).add_suffix("-cons")
        subsetCounCon = subsetCounCon.rename(columns={"DateTime-cons":"DateTime"})
        CountriesCons.append(subsetCounCon)

        #SE-areas production prognosis
        PATH = '../../../data/productionPrognosis/production-prognosis_' + str(i) + '_hourly.xls'
        subsetProdProg = readOneYearProdProg(PATH).add_suffix("-prod-prog")
        subsetProdProg = subsetProdProg.rename(columns={"DateTime-prod-prog":"DateTime"})
        SEprodProg.append(subsetProdProg)
      
        #SE-areas consumption prognosis
        PATH = '../../../data/consumptionPrognosis/consumption-prognosis-se_' + str(i) + '_hourly.xls'
        subsetConsProg = readOneYearFuncOther(PATH).add_suffix("-cons-prog")
        subsetConsProg = subsetConsProg.rename(columns={"DateTime-cons-prog":"DateTime"})
        SEconsProg.append(subsetConsProg)

        #Hydro reservoir
        PATH = '../../../data/hydroReservoir/hydro-reservoir_' + str(i) + '_weekly.xls'
        subsetHydroRes = readOneYearHydroRes(PATH).add_suffix("-hydro-res")
        subsetHydroRes = subsetHydroRes.rename(columns={"DateTime-hydro-res":"DateTime"})
        hydroRes.append(subsetHydroRes)
        

        #Wind prognosis
        PATH = '../../../data/windpowerPrognosis/wind-power-se-prognosis_' + str(i) + '_hourly.xls'
        subsetWindProg = readOneYearFuncOther(PATH).add_suffix("-wind-prog")
        subsetWindProg = subsetWindProg.rename(columns={"DateTime-wind-prog":"DateTime"})
        windProg.append(subsetWindProg)


    #Creating dataframes for each variable
    priceSet = pd.concat(price, ignore_index=True)
    SEprodSet = pd.concat(SEprod,ignore_index=True)
    SEconsSet = pd.concat(SEcons, ignore_index=True)
    counProdSet = pd.concat(CountriesProd, ignore_index=True)
    coundConSet = pd.concat(CountriesCons,ignore_index=True)
    SEprodProgSet = pd.concat(SEprodProg,ignore_index=True)
    SEconsProgSet = pd.concat(SEconsProg,ignore_index=True)
    windProgSet = pd.concat(windProg,ignore_index=True)
    hydroResSet = pd.concat(hydroRes,ignore_index=True)

    #Creating empty DateTime df for keeping same structure in final df
    rng = pd.date_range(start='2015-01-01', end='2021-01-01', freq='H', closed='left')
    dates = pd.DataFrame(rng, columns=['DateTime']) 

    #Inserting all dataframes into list, including DateTime structure df
    data_frames = [priceSet, SEprodSet, SEconsSet, counProdSet, coundConSet, SEprodProgSet, SEconsProgSet, windProgSet, hydroResSet]
    data_frames.insert(0, dates)
    
    #Merge all dataframes together on "DateTime"
    df_merged = reduce(lambda  left,right: pd.merge(left,right, on=['DateTime'], how='outer'), data_frames)
    
    print(df_merged)
    
    #--- Converting to UTC-----------
    #Finding CET dateTime values
    DatetimeIndexCET = pd.DatetimeIndex(priceSet['DateTime']).tz_localize('CET', ambiguous='infer', nonexistent='shift_backward')
    #Converting to UTC
    priceSet['DateTime'] = DatetimeIndexCET.tz_convert('UTC')
    #Removing timezone
    priceSet['DateTime'] = priceSet['DateTime'].dt.tz_localize(None)
    priceSet.info()
    #Removing emptly lines (with at least 4 empty values in row)
    priceSet.dropna(thresh=4, inplace=True)

    print(df_merged)
    df_merged.to_excel('NPdataOld.xlsx', engine='xlsxwriter')

    return 

readAllYears()