import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Dense
from tensorflow.keras.models import Sequential
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM


from helpFunc.sequenceData import obtainDataDict, extractValuesLSTM
from helpFunc.metrics import *



# MNIST class
class LSTM_class():
    def __init__(self,
                 l1_out=None, 
                 l2_out=None,
                 l1_drop=None, 
                 l2_drop=None,
                 batch_size=None,
                 lossMetric = 'mae', 
                 epochs=None,
                 priceArea=None,
                 targetName=None):

        self.l1_out = l1_out
        self.l2_out = l2_out
        self.l1_drop = l1_drop
        self.l2_drop = l2_drop
        self.batch_size = batch_size
        self.epochs = epochs
        self.lossMetric = lossMetric
        self.priceArea=priceArea
        self.targetName=targetName
        self.__x_train, self.__x_val, self.__x_test, self.__y_train, self.__y_val, self.__y_test, self.target_scaler, self.dataDict, self.df = self.lstm_data()
        self.__model = self.lstm_model()
        
        params = """
        priceArea:\t{0}
        targetName:\t{1}
        l1_out:\t{2}
        l2_out:\t{3}
        l1_drop:\t{4}
        l2_drop:\t{5}
        batch_size:\t{6}
        epochs:\t{7}
        lossMetric:\t{8}
        """.format(self.priceArea,
        self.targetName, 
        self.l1_out,
        self.l2_out, 
        self.l1_drop,
        self.l2_drop,
        self.batch_size,
        self.epochs,
        self.lossMetric)
        print(params)
        
    # load mnist data from keras dataset
    def lstm_data(self):
        PRICEAREA = self.priceArea
        TARGETNAME = self.targetName
        df, dataDict, targetScaler = obtainDataDict(PRICEAREA, TARGETNAME, weeklySequence=True, dailySequence=False)
        X_train, y_train = extractValuesLSTM(dataDict["train"][0],dataDict["train"][1],TARGETNAME)
        X_val, y_val = extractValuesLSTM(dataDict["val"][0],dataDict["val"][1],TARGETNAME)
        X_test, y_test = extractValuesLSTM(dataDict["test"][0],dataDict["test"][1], TARGETNAME)
        return X_train, X_val, X_test, y_train, y_val, y_test, targetScaler, dataDict, df
    
    # mnist model
    def lstm_model(self):        
        n_steps = self.__x_train.shape[1]
        n_features = self.__x_train.shape[2]
        n_output = self.__y_train.shape[1]

        model = Sequential()
        model.add(LSTM(self.l1_out, activation='relu', input_shape=(n_steps, n_features), return_sequences=True))
        model.add(Dropout(self.l1_drop))
        model.add(LSTM(self.l2_out, activation='relu', return_sequences=False))
        model.add(Dropout(self.l2_drop))

        model.add(Dense(n_output))
        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss=self.lossMetric, metrics=self.lossMetric)
        return model
    


    # fit lstm model
    def lstm_fit(self):
        early_stopping = EarlyStopping(patience=100, verbose=1)
        self.__model.fit(self.__x_train, self.__y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=1,
                       callbacks=[early_stopping], 
                       shuffle=False, 
                       validation_data=(self.__x_val, self.__y_val))
    

    # evaluate lstm model
    def lstm_evaluate(self,plotLoss=False):
        #Fitting (training) the model
        self.lstm_fit()
        #Printing loss of training
        if plotLoss:
            self.plot_loss()
        #evaluation = self.__model.evaluate(self.__x_test, self.__y_test, batch_size=self.batch_size, verbose=0)
        
        #Calculating validation metrics tranformed back to original
        Yp_val = self.__model.predict(self.__x_val).squeeze()
        Y_val = self.target_scaler.inverse_transform(self.__y_val.squeeze())
        mae_validation = np.mean(MAE(Y_val, Yp_val))
        smape_validation = np.mean(sMAPE(Y_val, Yp_val)) * 100

        #Calculating test metrics tranformed back to original
        Yp_test = self.__model.predict(self.__x_test).squeeze()
        Y_test = self.target_scaler.inverse_transform(self.__y_test.squeeze())
        mae_test= np.mean(MAE(Y_test, Yp_test))
        smape_test = np.mean(sMAPE(Y_test, Yp_test)) * 100
        
        #Plotting best and worst prediction for test values
        maxIdx, minIdx = self.worst_best_prediction(Yp_test, Y_test)
        self.plotPredictions(maxIdx, minIdx)

        return_values = {'loss': mae_validation, 'MAE Val': mae_validation, 'MAE Test': mae_test,
                     'sMAPE Val': smape_validation, 'sMAPE Test': smape_test, 
                     'status': STATUS_OK}
        return return_values


    def plot_loss(self):
        print(self.__model.history.history)
        loss = self.__model.history.history["loss"]
        val_loss = self.__model.history.history["val_loss"]
        epochs = range(len(loss))
        plt.figure()
        plt.plot(epochs, loss, "b", label="Training loss")
        plt.plot(epochs, val_loss, "r", label="Validation loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


    def worst_best_prediction(self, Yp_test, Y_test):
        print(len(Y_test))
        loss = len(Y_test)*[None] 
        for i in range(len(Y_test)):
            loss[i] = MAE(Y_test, Yp_test)
        maxIdx = loss.index(max(loss))
        minIdx = loss.index(min(loss))
        loss.pop(maxIdx)
        loss.pop(minIdx)
        secondMaxIdx = loss.index(max(loss))
        secondMinIdx = loss.index(min(loss))
        return maxIdx, minIdx, secondMaxIdx, secondMinIdx

    def plotPredictions(self, maxIdx, minIdx):
        badPred = self.dataDict["test"][1][maxIdx]
        print(badPred) 
        #goodPred = self.dataDict["test"][1][maxIdx]
        title1 = "Worst predicted 10-days ahead for "+self.targetName
        #title2 = "Best predicted 10-days ahead for "+self.targetName
        self.plotFunc(badPred, self.targetName, title1, "lst_pred_bad") 



    def plotFunc(self, predDf, targetName, title, directStr):
        #Plot actual values of week-df
        ax = predDf[['Scaled-target', 'DateTime']].plot(x='DateTime', legend="observed", color='#174D7F',figsize=(16,5.5))
        #Plot predicted values of week-df
        predDf.plot(x='DateTime', y='Pred-Unscaled', ax=ax, color="r")

        ax.set_xlabel('Date')
        ax.set_ylabel('SEK/MWh')
        ax.legend(['Observed values','Prediction'])
        #plt.title('Example of predicted price for SE1')
        plt.title(title)
        plt.tight_layout() 
        plt.savefig(directStr +'.png')
        plt.show()



def visualize_testPred(df, newDataDict, targetName, maxIdx,secondMaxIdx, minIdx, secondMinIdx):
    #ax = df[['SE1-price', 'DateTime']].plot(x='DateTime', legend="observed", color='#174D7F',figsize=(16,5.5))
    #for i in newDataDict["test"][1]:
      #  i.plot(x='DateTime', y='Pred-Unscaled', ax=ax)


    examplePredBad = newDataDict["test"][1][maxIdx] #Prediction with highest loss
    examplePredBad2 = newDataDict["test"][1][secondMaxIdx] #Prediction with 2nd highest loss

    examplePredGood = newDataDict["test"][1][minIdx] #Prediction with lowest loss
    examplePredGood2 = newDataDict["test"][1][secondMinIdx] #Prediction with 2nd lowest loss

    title1 = "Worst predicted 10-days ahead for "+targetName
    title2 = "Second worst predicted 10-days ahead for "+targetName
    title3 = "Best predicted 10-days ahead for "+targetName
    title4 = "Second best predicted 10-days ahead for "+targetName
    plotPred(examplePredBad,targetName, title1, "pred_bad")
    plotPred(examplePredBad2,targetName, title2, "pred_bad2")
    plotPred(examplePredGood,targetName, title3, "pred_good")
    plotPred(examplePredGood2,targetName, title4, "pred_good2")



def run_lstm(args):
    _lstm = LSTM_class(**args)
    lstm_evaluation = _lstm.lstm_evaluate(plotLoss=False)
    print("Obtained result loss:{0} \n".format(lstm_evaluation))
    return lstm_evaluation


#lstmm = LSTM_test()
#lstm_evaluation = lstmm.lstm_evaluate()



parameters = {
    #Data-parameters
    'priceArea' : 'SE1',
    'targetName' : 'SE1-price',

    #Network-parameters
    'l1_drop': 0.2,
    'l2_drop': 0.2,
    'l1_out': 128,
    'l2_out': 64,
    'batch_size': 128,
    'epochs': 2,
}



run_lstm(parameters)
