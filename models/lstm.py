import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Keras imports
from tensorflow.keras.layers import Dropout, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#Helpfunctions imports
from helpFunc.sequenceData import obtainDataDict, extractValuesLSTM
from helpFunc.ancillaryFunctions import evaluate, minMaxLoss, saveModel
from helpFunc.plots import plotLossFunction,plotWorstBest, plotAllPred




# MNIST class
class LSTM_class():
    def __init__(self,
                 n_hidden1, 
                 n_hidden2,
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
                 plotAllPredictions,
                 errors=None,
                 finalValLoss=None,
                 indicesMin=None,
                 indicesMax=None):

        self.modelName="LSTM"
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.batch_size = batch_size
        self.epochs = epochs
        self.learningRate = learningRate
        self.lossMetric = lossMetric
        self.patience=patience
        self.priceArea=priceArea
        self.targetName=targetName
        self.errors = errors
        self.finalValLoss = finalValLoss
        self.indicesMin = indicesMin 
        self.indicesMax = indicesMax
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
        model.add(LSTM(self.n_hidden2, return_sequences=False))

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
            plotLossFunction(fittedModel, self.modelName)
        self.finalValLoss = fittedModel.history["val_loss"][-1]

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


    def lstmEvaluate(self):
        testMAE, errors = evaluate(self.dataDict["test"][1])
        self.errors = errors
        print(self.errors)
        if self.plotWorstBestPrediction:
            self.indicesMin,self.indicesMax = minMaxLoss(testMAE,k=2)
            plotWorstBest(self.dataDict,self.targetName,self.modelName,self.indicesMin,self.indicesMax)
        if self.plotAllPredictions:
            plotAllPred(self.dataDict,self.targetName,self.modelName)
        
    def lstmSave(self):
        paramDict = {
            "NH1": self.n_hidden1,
            "NH2": self.n_hidden2,
            "EPOCHS": self.epochs,
            "BS": self.batch_size,
            "LR": self.learningRate,
            "PAT": self.patience,
            "DAY": self.dailySequence,
            "WEEK": self.weeklySequence
        }
        
        saveModel(self.dataDict, self.targetName, self.modelName, self.indicesMin, 
        self.indicesMax, self.errors, self.finalValLoss, paramDict)



if __name__ == "__main__":
    print("*****------------Running LSTM-------------******\n")

    parameters = {
        #Verbose/vizualize settings
        'verboseTraining': 1,
        'plotLoss': True,
        'plotWorstBestPrediction': True,
        'plotAllPredictions': False,

        #Data parameters
        'priceArea' : 'SE1',
        'targetName' : 'SE1-price',
        'dailySequence': False,
        'weeklySequence': True,

        #Network/training parameters
        'n_hidden1': 64,
        'n_hidden2': 32,
        'batch_size': 32,
        'epochs': 600,
        'learningRate': 0.0001,
        'lossMetric' :'mae',
        'patience': 300
   
    }

    _lstm = LSTM_class(**parameters)
    _lstm.lstmPredict()
    _lstm.lstmEvaluate()
    _lstm.lstmSave()
    print("*****------------LSTM FINISHED------------******\n")
