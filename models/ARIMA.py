from pmdarima import auto_arima
import warnings
from statsmodels.tsa.stattools import adfuller
from helpFunc.sequenceData import *
from helpFunc.metrics import *
warnings.filterwarnings("ignore")

from helpFunc.ancillaryFunctions import evaluate, minMaxLoss
from helpFunc.plots import plotWorstBest, plotAllPred


def arimaPredict(data):
    stepwise_fit = auto_arima(data, trace=True, suppress_warnings=True, seasonal=True)
    #stepwise_fit.summary()
    stepwise_fit.fit(data)
    prediction = stepwise_fit.predict(240)
    return prediction
#print("Best model:  ARIMA(2,1,2)(0,0,0)[0] AIC=505358.386")
#print("ARIMA(1,1,0)(2,0,0)[24] intercept   : AIC=503193.290")

def prediction(dataDict):
    for i in range(len(dataDict["test"][1])):
        Xtest = dataDict["test"][0][i]
        ytest = dataDict["test"][1][i]
        predict = arimaPredict(Xtest[TARGETNAME])
        ytest["Target"] = targetScaler.inverse_transform(np.array(ytest["Scaled-target"]).reshape(-1,1))
        ytest["Pred-Unscaled"] = np.array(predict.reshape(-1,1))
    return dataDict


if __name__ == "__main__":
    print("*****------------Running ARIMA-------------******\n")
    #----------------Specifying pricearea------------
    PRICEAREA = 'SE1'
    TARGETNAME = 'SE1-price'
    df, dataDict, targetScaler = obtainDataDict(PRICEAREA, TARGETNAME, weeklySequence=True, dailySequence=False)
    
    #Checking if timeseries stationary:
    #ad_test(df[TARGETNAME])

    newDataDict = prediction(dataDict)
    ptestMAE, errors = evaluate(newDataDict["test"][1])
    print(errors)

    plotWorstBestPrediction = True
    plotAllPredictions = False
    
    if plotWorstBestPrediction:
        indicesMin,indicesMax = minMaxLoss(testMAE,k=2)
        plotWorstBest(newDataDict,TARGETNAME,"ARIMA",indicesMin,indicesMax)
    if plotAllPredictions:
        plotAllPred(newDataDict,TARGETNAME,"ARIMA")


    print("*****------------ARIMA FINISHED------------******\n")