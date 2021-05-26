import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np


def readData():
    print("Reading data")
    PATH = "../read-data/fullDataSet.h5"
    df = pd.read_hdf(PATH, index_col=0)
    print("....")
    print("Dataset successfully read!")
    return df

#1. Removing lines with empty cells caused by lags (first 168 values)
#2. Linearly extrapolating for missing values
#3. Removing last row which is empty
#4. Setting datatype
def cleanData(df, shiftWeather=None, shiftProg=None):
    if shiftWeather != None:
        weatherCols = [c for c in df.columns if 'avg-hum' in c or 'avg-cloud' in c or 'avg-temp' in c or 'avg-press' in c or 'avg-wind' in c]
        df[weatherCols] = df[weatherCols].shift(-shiftWeather,axis=0)
    if shiftProg != None:
        progCols = [c for c in df.columns if 'prog' in c]
        df[progCols] = df[progCols].shift(-shiftProg,axis=0)

    df2 = df.copy(deep=True).iloc[169:-shiftWeather,:] #remove empty rows caused by shift (and start 00:00) and lags

    def showNan(df):
        df1 = df[df.isna().any(axis=1)]
        print(df1.index)

    #showNan(df2)
    df3 = df2.interpolate('linear')
    #showNan(df3)

    df4 = df3.dropna()
    #showNan(df4)

    type_dict = {'MonthInt': int,'MonthStr': 'string', 'WeekInt': int, 'WeekdayInt':int, 'WeekdayStr': 'string','DayInt': int}
    df4 = df4.astype(type_dict).reset_index(drop=True)
    return df4


def extractPriceArea(df, priceArea):
    #globalCols = [c for c in df.columns if "SE1" not in c and "SE2" not in c and "SE3" not in c and "SE4" not in c]
    #print(globalCols)
    dateTimeCol = ['DateTime']
    #Removed MonthStr, WeekInt, WeekdayStr, DayInt
    globalCols = [#'WeekInt', 
            'MonthInt', 'WeekdayInt',
            #'SE-prod', 'NO-prod', 'FI-prod', 'DK-prod', 
            #'SE-cons', 'NO-cons', 'FI-cons', 'DK-cons', 
            #'NO-hydro-res', 'SE-hydro-res', 'FI-hydro-res',
            'SE-hydro-res']
            #'SYS-price', 'SYS-price-lag-24h', 'SYS-price-lag-48h', 'SYS-price-lag-168h', 'SYS-price-avg-3h', 'SYS-price-avg-12h']

    #TODO! Think about how the prognosed values should be accounted for
    localCols = [c for c in df.columns if priceArea in c and "price-avg" not in c] # and "prog" and "lag" not in c]

    allCols = dateTimeCol + localCols + globalCols

    priceAreaDf = df[allCols]
    return priceAreaDf, allCols


def scale(df, targetName, allCols):
    pd.options.mode.chained_assignment = None #Errors
    timeSeries = df[allCols]
    colsToBeScaled = [c for c in timeSeries.columns if c != 'DateTime' and c!= targetName]

    #print(timeSeries)
    #print(timeSeries.idxmax())
    #print(timeSeries.iloc[27511])

    #--------------Scaling Targeted price-------------------
    targetScaler = MinMaxScaler(feature_range=(0, 1))
    timeSeries.insert(2,'Scaled-target', targetScaler.fit_transform(timeSeries[[targetName]]))
    #print(timeSeries)
    #print(timeSeries.idxmax())

    #----------------Scaling Other features-----------------
    if colsToBeScaled != []:
        scaler = MinMaxScaler(feature_range=(0, 1))
        timeSeries[colsToBeScaled] = scaler.fit_transform(timeSeries[colsToBeScaled])
    return timeSeries, targetScaler



    #-----------Splitting dataframe to sequences-----------
def weeklySequencing(data, obs_steps, f_steps):
    sequences = []
    targets = []
    
    #Split data into sequences with obs_steps length for every 24th value
    last_possible_value = data.tail(1).index.item() - (obs_steps-1) - (f_steps-1)
    n = 0
    for index, row in data.iterrows():
        startIdx = n + index
        if startIdx <= last_possible_value:
            seq = data.loc[startIdx: startIdx + obs_steps-1]
            sequences.append(seq)
            n += 23

    #Extract target sequence with f_steps length following each X sequence 
    #remove = []
    for i, seq in enumerate(sequences):
        targetIdx = seq.index[-1]+1
        targetSeq = data.loc[targetIdx:targetIdx+f_steps-1][['DateTime', 'Scaled-target']]
        if targetSeq.shape[0] == f_steps: 
            #Append only if target is f_step long
            targets.append(targetSeq)
        #else:
            #If target horizon not long enough we need to remove train seq
         #   remove.append(i)

    #Removing train sequences which do not have forecast values
    #remove = sorted(remove, reverse=True)
    #for r in remove:
       #sequences.pop(r)

    return sequences, targets

def dailySequencing(data, obs_steps, f_steps):
    sequences = []
    targets = []
    weatherCols = [c for c in data.columns if 'avg-hum' in c or 'avg-cloud' in c or 'avg-temp' in c or 'avg-press' in c or 'avg-wind' in c]
    #Split data into sequences with obs_steps length for every 24th value
    last_possible_value = data.tail(1).index.item() - (obs_steps-1) - (f_steps-1)
    n = 0
    for index, row in data.iterrows():
        startIdx = n + index
        if startIdx <= last_possible_value:
            seq = data.loc[startIdx: startIdx + obs_steps-1]
            
            #Adding 7-days forecast weather values
            for w in weatherCols:
              for i in range (7):
                name=w+"-day-"+str(i+1)
                seq[name] = np.array(data.loc[startIdx + (i+1)*24 : startIdx + obs_steps-1 + (i+1)*24][w])

            #Adding to sequence-list
            sequences.append(seq)
            n += 23

    #Extract target sequence with f_steps length following each X sequence 
    for i, seq in enumerate(sequences):
        targetIdx = seq.index[-1]+1
        targetSeq = data.loc[targetIdx:targetIdx+f_steps-1][['DateTime', 'Scaled-target']]
        if targetSeq.shape[0] == f_steps: 
            #Append only if target is f_step long
            targets.append(targetSeq)


    
    #print(sequences[-1])
    #print(targets[-1])
    return sequences, targets


def randomSplitIdx(sequences):
    random.seed(22)

    length = len(sequences)
    idx = np.arange(0, length).tolist() 
    random.shuffle(idx)

    trainIdx = idx[0:round(0.7*length)]
    valIdx = idx[round(0.7*length):round(0.85*length)]
    testIdx = idx[round(0.85*length):]

    return trainIdx, valIdx, testIdx 

def regularSplitIdx(sequences):
    length = len(sequences)
    idx = np.arange(0, length).tolist() 

    trainIdx = idx[0:round(0.7*length)]
    valIdx = idx[round(0.7*length):round(0.85*length)]
    testIdx = idx[round(0.85*length):]

    return trainIdx, valIdx, testIdx 

def getFromIdx(x, y, idx):

    X_mapping = map(x.__getitem__, idx)
    X_list = list(X_mapping)

    y_mapping = map(y.__getitem__, idx)
    y_list = list(y_mapping)

    return X_list, y_list, idx


def dataSplit(seq, tar):
    #Creating empty data dictionary
    dataDict = {}
    
    #Extracting randomized and 80-20-20 splitted indices
    trainIdx, valIdx, testIdx = randomSplitIdx(seq)
    #Putting them inside list to iterate over
    indexList = [trainIdx, valIdx, testIdx]
    #Names for keys in dictionary
    names = ["train", "val","test"]
    
    #set list of X values (df:s), y values (df:s) and indices (list) as values for dict
    for i in range(len(indexList)):
        dataDict[names[i]] = getFromIdx(seq,tar,indexList[i])

    print("\nTrain, val and testset divided!")
    print("Total number of samples: ", len(seq))
    print("Training samples: \t ", len(trainIdx))
    print("Validation samples: \t ", len(valIdx))
    print("Test samples:\t\t ", len(testIdx), "\n")

    return dataDict




def extractValuesLSTM(sequences, targets, targetName):
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    # Extracting columns which are not DateTime and are not the unscaled price values
    cols = [c for c in sequences[0].columns if c != 'DateTime' and c!= targetName] #All cols except unscaled target and DateTime
    #cols = [c for c in sequences[0].columns if c == "Scaled-target"] #Only historical prices
    #print(cols)
    X = np.array([i[cols].values for i in sequences])
    y = np.array([i['Scaled-target'].values for i in targets ])
    #print(X.shape)
    #print(y.shape)
    #print(X[0])
    
    #X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
    y = np.reshape(y, (y.shape[0], y.shape[1], 1))

    return X, y


def extractValuesMLP(sequences, targets, targetName):
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    # Extracting columns which are not DateTime and are not the unscaled price values
    cols = [c for c in sequences[0].columns if c != 'DateTime' and c!= targetName] #All cols except unscaled target and DateTime
    #cols = [c for c in sequences[0].columns if c == "Scaled-target"] #Only historical prices
    #print(cols)
    X = np.array([i[cols].values for i in sequences])
    y = np.array([i['Scaled-target'].values for i in targets ])
    #print(X.shape)
    #print(y.shape)
    #print(X[0])
    
    X = np.reshape(X, (X.shape[0],-1))
    #y = np.reshape(y, (y.shape[0], y.shape[1], 1))
    return X, y



PRICEAREA = 'SE1'
TARGETNAME = 'SE1-price'

def obtainDataDict(PRICEAREA, TARGETNAME, weeklySequence=False, dailySequence=False):
    #-------------------Preparing data----------------
    #Preparing complete dataset
    if weeklySequence:
        shiftWeather = 168
    if dailySequence:
        shiftWeather = 24
    shiftProg = 24
    df = readData() #reading data
    df = cleanData(df, shiftWeather, shiftProg) #removing empty cells, linearly extrapolating
    df, allcols = extractPriceArea(df, PRICEAREA) #extracting values of SE1
    df, targetScaler = scale(df, TARGETNAME, allcols) #Scaling
    #print(df)
    print(df.info())

    #Sequencing data
    if weeklySequence:
        print("\n---WEEKLY SEQUENCING---\n") 
        obs_steps = 168
        f_steps = 240
        sequences, targets = weeklySequencing(df, obs_steps, f_steps)
        #print(sequences[0])
        #print(targets[0])
        #print(sequences[-1])
        #print(targets[-1])

    if dailySequence:
        print("\n----DAILY SEQUENCING----\n") 
        obs_steps = 24
        f_steps = 240
        sequences, targets = dailySequencing(df, obs_steps, f_steps)
        #print(sequences[0])
        #print(targets[0])
        #print(sequences[-1])
        #print(targets[-1])

    dataDict = dataSplit(sequences, targets)
    return df, dataDict, targetScaler


#df, dataDict, targetScaler = obtainDataDict(PRICEAREA,TARGETNAME, dailySequence=True)
#X_test, y_test = extractValues(dataDict["test"][0],dataDict["test"][1], TARGETNAME)
