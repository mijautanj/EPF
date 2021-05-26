import pandas as pd

def readOneYear(PATH, columns=None):
    if columns == None:
        columns='A:G'
    subset = pd.read_excel(PATH, usecols=columns, header=2, dtype={'Date': str, 'Hours': str, 'SYS': float, 'SE1': float, 'SE2': float, 'SE3': float, 'SE4': float }) #Read excel, columns A:T, use row 2 as header (labels)

    #CREATING TIME COLUMNS
    subset.rename(columns={subset.columns[0]: 'Date' }, inplace = True ) #Rename the first column to Date
    #Creating dateTime column
    dateTimeString = subset['Date']+ ' '+ subset['Hours'].astype(str).str[0:2] #Creating complete DateTime string for all rows
    dateTimeDf = pd.to_datetime(dateTimeString, format='%d-%m-%Y %H') #Creating DateTime DataFrame for 
    subset.insert(2,'DateTime',dateTimeDf) #Inserting DateTime dataframe

    subset['Date'] = pd.to_datetime(subset['Date'], format='%d-%m-%Y') #Setting the Date column to be of datetime dtype 

    newset = subset.drop(['Date','Hours'], axis=1) #Remove separate columns for date and hours, only keep DateTime
    return newset

def readOneYearHydro(PATH):
    subset = pd.read_excel(PATH, usecols='A:G', header=2, dtype={'Date': str, 'NO': float, 'SE': float, 'FI': float})
    subset.rename(columns={subset.columns[0]: 'Date' }, inplace = True ) #Rename the first column to Week

    #Creating dateTime column
    weekString = ((subset['Date'].astype(str).str[0:2]).astype(int)-1).astype(str) #Creating week string
    yearString = subset['Date'].astype(str).str[-2:] #Creating year string
    dateString = "20" + yearString + "-" + weekString + "-" + "1" #Final datestring
    dateSeries = pd.to_datetime(dateString,format = '%Y-%W-%w') #Parsing to dateTIme
    subset.insert(1,'DateTime',dateSeries) #Inserting DateTime dataframe

    subset.drop(columns='Date', inplace=True) #Dropping Week-Year string column
    return subset

def upsampleHydro(df):
    #Up-smapling to hourly resolution (all the values inside the week will be added to the mean)
    subset1 = df.set_index('DateTime').resample('H').mean()
    #Interpolating missing values to be linear points
    subset2 = subset1.interpolate('linear')
    subset3 = subset2.reset_index()
    return subset3


def CETtoUTC(df):
    #--- Converting from CET to UTC-----------
    #Finding CET dateTime values
    DatetimeIndexCET = pd.DatetimeIndex(df['DateTime']).tz_localize('CET', ambiguous='infer', nonexistent='shift_backward')
    #Converting to UTC
    df['DateTime'] = DatetimeIndexCET.tz_convert('UTC')
    #Removing timezone stamp
    df['DateTime'] = df['DateTime'].dt.tz_localize(None)
    #Removing emptly lines (with at least 4 empty values in row)
    df.dropna(thresh=4, inplace=True)
    return df



def calVars(df):
    df.insert(1,'MonthInt', df['DateTime'].dt.strftime('%m')) #Values 1-12
    df.insert(2,'MonthStr', df['DateTime'].dt.strftime('%b')) #Strings Jan-Dec
    df.insert(3,'WeekInt', df['DateTime'].dt.strftime('%W').astype(int)+1) #Values 1-52
    df.insert(4,'WeekdayInt', df['DateTime'].dt.weekday+1) #Values 1-7 where 1 is monday
    df.insert(5,'WeekdayStr', df['DateTime'].dt.strftime('%A')) #Strings Monday-Wednesday
    df.insert(6,'DayInt', df['DateTime'].dt.strftime('%d')) #Values 1-31

    #Setting the types:
    type_dict = {'MonthInt': int,'MonthStr': 'string', 'WeekdayStr': 'string','DayInt': int}
    df = df.astype(type_dict)
    return df


