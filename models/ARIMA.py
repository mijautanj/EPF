import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

from helpFunc.ancillaryFunctions import evaluate, minMaxLoss, saveModel
from helpFunc.plots import plotWorstBest, plotAllPred
from helpFunc.sequenceData import obtainDataDict


def arimaPredict(data):
    stepwise_fit = auto_arima(data, trace=True, suppress_warnings=True, seasonal=True) #Trace prints all iterations
    #stepwise_fit.summary()
    stepwise_fit.fit(data)
    prediction = stepwise_fit.predict(240)
    return prediction


def prediction(dataDict):
    for i in range(len(dataDict["test"][1])):
        Xtest = dataDict["test"][0][i]
        ytest = dataDict["test"][1][i]
        predict = arimaPredict(Xtest[TARGETNAME])
        ytest["Target"] = targetScaler.inverse_transform(np.array(ytest["Scaled-target"]).reshape(-1,1))
        ytest["Pred-Unscaled"] = np.array(predict.reshape(-1,1))
    return dataDict


def arimaSave(dataDict, targetName, modelName, indicesMin, indicesMax,errors,finalValLoss, paramDict):
    saveModel(dataDict, targetName, modelName, indicesMin, indicesMax, errors, finalValLoss, paramDict)

if __name__ == "__main__":
    print("*****------------Running ARIMA-------------******\n")
    #----------------Specifying pricearea------------
    PRICEAREA = 'SE1'
    TARGETNAME = 'SE1-price'
    df, dataDict, targetScaler = obtainDataDict(PRICEAREA, TARGETNAME, weeklySequence=True, dailySequence=False)
    
    #Checking if timeseries stationary:
    #ad_test(df[TARGETNAME])

    newDataDict = prediction(dataDict)
    testMAE, errors = evaluate(newDataDict["test"][1])
    print(errors)

    plotWorstBestPrediction = True
    plotAllPredictions = False
    
    if plotWorstBestPrediction:
        indicesMin,indicesMax = minMaxLoss(testMAE,k=2)
        plotWorstBest(newDataDict,TARGETNAME,"ARIMA",indicesMin,indicesMax)
    if plotAllPredictions:
        plotAllPred(newDataDict,TARGETNAME,"ARIMA")
    

    arimaSave(newDataDict, TARGETNAME, "ARIMA", 0, 0, errors,0, {})


    print("*****------------ARIMA FINISHED------------******\n")