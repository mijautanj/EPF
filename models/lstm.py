import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Keras imports
from tensorflow.keras.layers import Dropout, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM

#Helpfunctions imports
from helpFunc.sequenceData import obtainDataDict, extractValuesLSTM
from helpFunc.ancillaryFunctions import evaluate, minMaxLoss
from helpFunc.metrics import MAE, sMAPE, MAPE, RMSE
from helpFunc.plots import plotLossFunction,plotWorstBest, plotAllPred




# MNIST class
class LSTM_class():
    def __init__(self,
                 n_hidden1, 
                 n_hidden2,
                 dropout1,
                 dropout2,
                 batch_size,
                 lossMetric,
                 epochs,
                 learningRate,
                 patience,
                 priceArea,
                 targetName,
                 dailySequence,
                 weeklySequence,
                 verboseTraining,
                 plotLoss,
                 plotWorstBestPrediction,
                 plotAllPredictions):

        self.modelName="LSTM"
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.batch_size = batch_size
        self.epochs = epochs
        self.learningRate = learningRate
        self.lossMetric = lossMetric
        self.patience=patience
        self.priceArea=priceArea
        self.targetName=targetName
        self.dailySequence=dailySequence
        self.weeklySequence=weeklySequence
        self.verboseTraining=verboseTraining
        self.plotLoss=plotLoss
        self.plotWorstBestPrediction=plotWorstBestPrediction
        self.plotAllPredictions=plotAllPredictions
        self.__x_train, self.__x_val, self.__x_test, self.__y_train, self.__y_val, self.__y_test, self.targetScaler, self.dataDict, self.df = self.lstm_data()
        self.__model = self.lstm_model()
        
        
    # load data from sequenceData
    def lstm_data(self):
        PRICEAREA = self.priceArea
        TARGETNAME = self.targetName
        df, dataDict, targetScaler = obtainDataDict(PRICEAREA, TARGETNAME, weeklySequence=self.weeklySequence, dailySequence=self.dailySequence)
        X_train, y_train = extractValuesLSTM(dataDict["train"][0],dataDict["train"][1],TARGETNAME)
        X_val, y_val = extractValuesLSTM(dataDict["val"][0],dataDict["val"][1],TARGETNAME)
        X_test, y_test = extractValuesLSTM(dataDict["test"][0],dataDict["test"][1],TARGETNAME)
        return X_train, X_val, X_test, y_train, y_val, y_test, targetScaler, dataDict, df
    
    # model
    def lstm_model(self):        
        n_steps = self.__x_train.shape[1]
        n_features = self.__x_train.shape[2]
        n_output = self.__y_train.shape[1]

        model = Sequential()
        model.add(LSTM(self.n_hidden1, input_shape=(n_steps, n_features), return_sequences=True))
        model.add(Dropout(self.dropout1))
        model.add(LSTM(self.n_hidden2, return_sequences=False))
        model.add(Dropout(self.dropout2))

        model.add(Dense(n_output))
        opt = Adam(learning_rate=self.learningRate)
        model.compile(optimizer=opt, loss=self.lossMetric, metrics=self.lossMetric)
        return model
    

    # fit lstm model
    def lstmFit(self):
        early_stopping = EarlyStopping(patience=self.patience, verbose=1)
        fittedModel = self.__model.fit(self.__x_train, self.__y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=self.verboseTraining,
                       callbacks=[early_stopping], 
                       shuffle=False, 
                       validation_data=(self.__x_val, self.__y_val))
        if self.plotLoss:
            plotLossFunction(fittedModel)

    # evaluate lstm model
    def lstmPredict(self):
        #Fitting (training) the model
        self.lstmFit()
       
        #Forecasting for all test samples
        y_hat_test = self.__model.predict(self.__x_test, batch_size=self.batch_size)

        #Saving all predicted values and unscaled values in dataDict
        for i, j in enumerate(self.dataDict["test"][1]):
            predict = np.array(y_hat_test[i]).reshape(-1,1)
            j["Predicted"] = predict
            j["Pred-Unscaled"] = self.targetScaler.inverse_transform(predict)
            j["Target"] = self.targetScaler.inverse_transform(np.array(j["Scaled-target"]).reshape(-1,1))

        return

    def lstmEvaluate(self):
        testMAE, errors = evaluate(self.dataDict["test"][1])
        if self.plotWorstBestPrediction:
            indicesMin,indicesMax = minMaxLoss(testMAE,k=2)
            plotWorstBest(self.dataDict,self.targetName,self.modelName,indicesMin,indicesMax)
        if self.plotAllPredictions:
            plotAllPred(self.dataDict,self.targetName,self.modelName)
        print(errors)
    



if __name__ == "__main__":

    parameters = {
        #Verbose/vizualize settings
        'verboseTraining': 1,
        'plotLoss': True,
        'plotWorstBestPrediction': True,
        'plotAllPredictions': True,

        #Data parameters
        'priceArea' : 'SE1',
        'targetName' : 'SE1-price',
        'dailySequence': False,
        'weeklySequence': True,

        #Network/training parameters
        'n_hidden1': 64,
        'n_hidden2': 64,
        'dropout1': 0.2,
        'dropout2': 0.2,
        'batch_size': 32,
        'epochs': 1000,
        'learningRate': 0.001,
        'lossMetric' :'mae',
        'patience': 200
   
    }

    _lstm = LSTM_class(**parameters)
    _lstm.lstmPredict()
    _lstm.lstmEvaluate()
