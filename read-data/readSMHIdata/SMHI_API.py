import pandas as pd

PARAMETERS = {
    "airPressure": 9,
    "airTemperature" : 1, 
    "windSpeed" : 4,
    "relativeHumidity" : 6,
    "totalCloudCover" : 16,
    }

#INFO:
#PARAMETER                  INFO                                                UNIT
# airPressure:              vid havsytans nivå, momentanvärde, 1 gång/tim       hPascal
# aireTmperature:           momentanvärde, 1 gång/tim                           °C  
# windSpeed:                medelvärde 10 min, 1 gång/tim                       m/s  
# relativeHumidity:         momentanvärde, 1 gång/tim                           %
# totalCloudCover:          momentanvärde, 1 gång/tim                           %
# maxOfMeanParticipation:   max av medel under 15 min, 4 gånger/tim             mm/s


  
def getDf(URL):
    df = pd.read_csv(URL, header=None, skip_blank_lines=True, encoding='utf-8', names=['AllData']) #just reading the csv without formatting, colname Alldata
    labelRow = df[df['AllData'].str.contains("Datum")] #Finding row with labels
    labelRowIdx = labelRow.index.tolist()[0] #Finding the index of labelrow

    NameRowIdx = df[df['AllData'].str.contains("Stationsnamn")].index.tolist()[0] #Finding row with Stationname
    df = pd.DataFrame(data = df['AllData'].str.split(';', expand=True)) #expanding dataframe by splitting the ; separated cols

    if df[0].str.contains("Stationsnamn").any(): #Checking if station name really in first column
        name = df.iloc[NameRowIdx+1,0] #Finding the station name
    else:
        name = "Unknown" #Too bad :()

    df[name] = name #Creating column with station name
    labels = df.iloc[labelRowIdx].values #Extracting all the labels and 
    labels[-1] = "Stationsnamn" #Changing last label to be of type Stationname

    #Removing excessive rows
    rowsDf = df[labelRowIdx+1:] #Extracting the rows after labelrow as values
    rowsDf.columns=(labels) #Extracting labels from row labelrow in csv 

    #Removing excessive cols
    colsDf = rowsDf.reset_index(drop=True).drop('',axis=1).drop('Tidsutsnitt:',axis=1) #Resetting index and dropping 2 unnecessary cols
    return colsDf


def dateTimeParser(df):
    cols = df.columns[0:2]
    dateTimeString = df[cols[0]] + ' ' + df[cols[1]] #Creating complete DateTime string for all rows
    dateTimeDf = pd.to_datetime(dateTimeString, format='%Y-%m-%d %H').dt.round('h') #Creating DateTime DataFrame for (rounded to hours)
    df.insert(2,'DateTime',dateTimeDf) #Inserting DateTime dataframe
    df.drop(cols.values, axis=1, inplace=True) #Dropping the two old cols with 'Datum' and 'Tid (UTC)'
    floatTypeDf = df.astype({df.columns[1]: float}) #Converting variable values to float type
    return floatTypeDf


def selectYears(df,start,end):
    clearedDf = df[df['DateTime'].dt.year.between(start,end)] #Removing year not in between start and end
    clearedDf.reset_index(drop=True, inplace=True) #Updating indexing for new df
    return clearedDf

def oneColData(df):
    newdf = pd.DataFrame(df.iloc[:,0:2]) #Only use DateTime and variable columns
    newdf.rename(columns={newdf.columns[1]: df['Stationsnamn'][1] }, inplace = True ) #Renaming "StationName" to actual station name
    return newdf



def SMHI_API_func(URL):
    df = getDf(URL)
    df = dateTimeParser(df)
    df = selectYears(df,2015,2020)
    df = oneColData(df)
    return df


