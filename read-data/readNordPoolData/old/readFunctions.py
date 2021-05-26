import pandas as pd
import matplotlib.pyplot as plt

def readOneYearFuncFirst(PATH, rows=None):

    subset = pd.read_excel(PATH, usecols='A:G', header=2, dtype={'Date': str, 'Hours': str, 'SYS': float, 'SE1': float, 'SE2': float, 'SE3': float, 'SE4': float }) #Read excel, columns A:T, use row 2 as header (labels)

    #CREATING TIME COLUMNS
    subset.rename(columns={subset.columns[0]: 'Date' }, inplace = True ) #Rename the first column to Date
    #Creating dateTime column
    dateTimeString = subset['Date']+ ' '+ subset['Hours'].astype(str).str[0:2] #Creating complete DateTime string for all rows
    dateTimeDf = pd.to_datetime(dateTimeString, format='%d-%m-%Y %H') #Creating DateTime DataFrame for 
    subset.insert(2,'DateTime',dateTimeDf) #Inserting DateTime dataframe

   

    subset['Date'] = pd.to_datetime(subset['Date'], format='%d-%m-%Y') #Setting the Date column to be of datetime dtype 
    subset.insert(3,'MonthInt', subset['Date'].dt.strftime('%m')) #Values 1-12
    subset.insert(4,'MonthStr', subset['Date'].dt.strftime('%b')) #Strings Jan-Dec
    subset.insert(5,'WeekInt', subset['Date'].dt.strftime('%W').astype(int)+1) #Values 1-52
    subset.insert(6,'WeekdayInt', subset['Date'].dt.weekday+1) #Values 1-7 where 1 is monday
    subset.insert(7,'WeekdayStr', subset['Date'].dt.strftime('%A')) #Strings Monday-Wednesday
    subset.insert(8,'DayInt', subset['Date'].dt.strftime('%d')) #Values 1-31

    #Setting the types:
    convert_dict = {'MonthInt': int,'MonthStr': 'string', 'WeekdayStr': 'string','DayInt': int}
    subset = subset.astype(convert_dict)

    newset = subset.drop(['Date','Hours'], axis=1) #Remove separate columns for date and hours, only keep DateTime
    if rows != None:
        newset = newset.iloc[0:rows]    #If amount of rows is specified, only return those
    return newset


def readOneYearFuncOther(PATH, rows=None):
    subset = pd.read_excel(PATH, usecols='A:G', header=2, dtype={'Date': str, 'Hours': str, 'SYS': float, 'SE1': float, 'SE2': float, 'SE3': float, 'SE4': float }) #Read excel, columns A:T, use row 2 as header (labels)

    #CREATING TIME COLUMNS
    subset.rename(columns={subset.columns[0]: 'Date' }, inplace = True ) #Rename the first column to Date
    #Creating dateTime column
    dateTimeString = subset['Date']+ ' '+ subset['Hours'].astype(str).str[0:2] #Creating complete DateTime string for all rows
    dateTimeDf = pd.to_datetime(dateTimeString, format='%d-%m-%Y %H') #Creating DateTime DataFrame for 
    subset.insert(2,'DateTime',dateTimeDf) #Inserting DateTime dataframe

    subset['Date'] = pd.to_datetime(subset['Date'], format='%d-%m-%Y') #Setting the Date column to be of datetime dtype 


    newset = subset.drop(['Date','Hours'], axis=1) #Remove separate columns for date and hours, only keep DateTime
    if rows != None:
        newset = newset.iloc[0:rows]    #If amount of rows is specified, only return those
    return newset


def readOneYearProdProg(PATH, rows=None):
    #Reason for this function is that columns of read excel file changed
    subset = pd.read_excel(PATH, usecols='A,B,H:L', header=2, dtype={'Date': str, 'Hours': str, 'SYS': float, 'SE1': float, 'SE2': float, 'SE3': float, 'SE4': float }) #Read excel, columns A:T, use row 2 as header (labels)

    #CREATING TIME COLUMNS
    subset.rename(columns={subset.columns[0]: 'Date' }, inplace = True ) #Rename the first column to Date
    #Creating dateTime column
    dateTimeString = subset['Date']+ ' '+ subset['Hours'].astype(str).str[0:2] #Creating complete DateTime string for all rows
    dateTimeDf = pd.to_datetime(dateTimeString, format='%d-%m-%Y %H') #Creating DateTime DataFrame for 
    subset.insert(2,'DateTime',dateTimeDf) #Inserting DateTime dataframe

    subset['Date'] = pd.to_datetime(subset['Date'], format='%d-%m-%Y') #Setting the Date column to be of datetime dtype 
    newset = subset.drop(['Date','Hours'], axis=1) #Remove separate columns for date and hours, only keep DateTime
    if rows != None:
        newset = newset.iloc[0:rows]    #If amount of rows is specified, only return those
    return newset



def readOneYearHydroRes(PATH, rows=None):
    subset = pd.read_excel(PATH, usecols='A:G', header=2, dtype={'Date': str, 'NO': float, 'SE': float, 'FI': float})
    subset.rename(columns={subset.columns[0]: 'Date' }, inplace = True ) #Rename the first column to Week

    #Creating dateTime column
    weekString = ((subset['Date'].astype(str).str[0:2]).astype(int)-1).astype(str) #Creating week string
    yearString = subset['Date'].astype(str).str[-2:] #Creating year string
    dateString = "20" + yearString + "-" + weekString + "-" + "1" #Final datestring
    dateSeries = pd.to_datetime(dateString,format = '%Y-%W-%w') #Parsing to dateTIme
    subset.insert(1,'DateTime',dateSeries) #Inserting DateTime dataframe
    
    #Up-smapling to hourly resolution (all the values inside the week will be added to the mean)
    subset2 = subset.set_index('DateTime').resample('H').mean()
    #Interpolating missing values to be linear points
    subset3 = subset2.interpolate('linear')

    if rows != None:
        newset = newset.iloc[0:rows]    #If amount of rows is specified, only return those
    return subset3.reset_index()




def smallDataFrame(PATH):
    PATH = '../NordPool-data/elspotPrices/elspot-prices_2019_hourly_sek.xls'
    rows = 20 #Only read this many rows
    df = readOneYearFuncFirst(PATH,rows)
    dfSmall = df.drop(['SYS','MonthInt', 'MonthStr', 'DayInt', 'WeekdayInt', 'WeekdayStr'], axis=1) #Dropping all unnecessary columns
    return dfSmall


#PATH = '../../data/hydroReservoir/hydro-reservoir_2015_weekly.xls'
#df = readOneYearHydroRes(PATH)
#print(df)
#df.to_excel("check.xlsx", engine='xlsxwriter')
   