import pickle
import numpy as np
import pandas as pd
from epftoolbox.evaluation import DM
import matplotlib.pyplot as plt


def DMplot(p_values, forecasts,title,savefig):
        # Defining color map
    red = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0.5, 50)[1:], [0]])
    green = np.concatenate([np.linspace(0.5, 1, 50), np.zeros(50)])
    blue = np.zeros(100)
    rgb_color_map = np.concatenate([red.reshape(-1, 1), green.reshape(-1, 1), 
                                    blue.reshape(-1, 1)], axis=1)
    rgb_color_map = mpl.colors.ListedColormap(rgb_color_map)

    # Generating figure
    plt.imshow(p_values.astype(float).values, cmap=rgb_color_map, vmin=0, vmax=0.1)
    plt.xticks(range(len(forecasts.columns)), forecasts.columns, rotation=90.)
    plt.yticks(range(len(forecasts.columns)), forecasts.columns)
    plt.plot(range(p_values.shape[0]), range(p_values.shape[0]), 'wx')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()

    if savefig:
        plt.savefig(title + '.png', dpi=300)
        plt.savefig(title + '.eps')

    plt.show()


def plot_multivariate_DM_test(real_price, forecasts, norm=1, title='DM test', savefig=True, path=''):
    # Computing the multivariate DM test for each forecast pair
    p_values = pd.DataFrame(index=forecasts.columns, columns=forecasts.columns) 

    for model1 in forecasts.columns:
        for model2 in forecasts.columns:
            # For the diagonal elemnts representing comparing the same model we directly set a 
            # p-value of 1
            if model1 == model2:
                p_values.loc[model1, model2] = 1
            else:
                p_values.loc[model1, model2] = DM(p_real=real_price.values, 
                                                  p_pred_1=forecasts.loc[:, model1].values.reshape(-1,240), 
                                                  p_pred_2=forecasts.loc[:, model2].values.reshape(-1,240), 
                                                  norm=norm, version='multivariate')


    #DMplot(p_values, forecasts,title,savefig)




def dmTest(true,model1,model2):
    return DM(p_real=np.array(true), p_pred_1=np.array(model1), p_pred_2=np.array(model2), norm=1, version='multivariate')


def performCompleteDMtest(lstmTestTarget, cnnTestTarget, mlpTestTarget,arimaTestTarget):
    df = pd.DataFrame(columns=["TRUE", "LSTM", "MLP", "CNN", "ARIMA"])
  
    print(df)
    for i in range(len(lstmTestTarget)):
        print(lstmTestTarget[i]["Target"])
        df["TRUE"].append(lstmTestTarget[i]["Target"])
        df["LSTM"].append(lstmTestTarget[i]["Pred-Unscaled"])
        df["MLP"].append(mlpTestTarget[i]["Pred-Unscaled"])
        df["CNN"].append(cnnTestTarget[i]["Pred-Unscaled"])
        df["ARIMA"].append(arimaTestTarget[i]["Pred-Unscaled"])

    print(df)

    #forecasts =  pd.DataFrame(data)
    #true = pd.Series(true)
    #print(forecasts)
    print("HHEHHEHEHE")
    
    #plot_multivariate_DM_test(true, forecasts)

    #testing lstm
    #p_value1 = dmTest(true,lstm,cnn)
    #p_value2 = dmTest(true,lstm,mlp)
    

def performSingleDMtest(lstmTestTarget, cnnTestTarget, mlpTestTarget,arimaTestTarget):
    lstm = []
    mlp = []
    true = []
    cnn = []
    arima=[]

    for i in range(len(lstmTestTarget)):
        true.append(lstmTestTarget[i]["Target"].values)
        lstm.append(lstmTestTarget[i]["Pred-Unscaled"].values)
        mlp.append(mlpTestTarget[i]["Pred-Unscaled"].values)
        cnn.append(cnnTestTarget[i]["Pred-Unscaled"].values)
        arima.append(arimaTestTarget[i]["Pred-Unscaled"].values)

  
    #testing lstm
    print("\nLSTM")
    p_value1 = dmTest(true,lstm,arima)
    p_value2 = dmTest(true,lstm,cnn)
    p_value3 = dmTest(true,lstm,mlp)
    print("ARIMA: ", p_value1)
    print("CNN: ", p_value2)
    print("MLP: ", p_value3)

    #testing cnn
    print("\nCNN")
    p_value1 = dmTest(true,cnn,arima)
    p_value2 = dmTest(true,cnn,lstm)
    p_value3 = dmTest(true,cnn,mlp)
    print("ARIMA: ", p_value1)
    print("LSTM: ", p_value2)
    print("MLP: ", p_value3)

    #testing mlp
    print("\nMLP")
    p_value1 = dmTest(true,mlp,arima)
    p_value2 = dmTest(true,mlp,lstm)
    p_value3 = dmTest(true,mlp,cnn)
    print("ARIMA: ", p_value1)
    print("LSTM: ", p_value2)
    print("CNN: ", p_value3)

    #testing ARIMA
    print("\nARIMA")
    p_value1 = dmTest(true,arima,mlp)
    p_value2 = dmTest(true,arima,lstm)
    p_value3 = dmTest(true,arima,cnn)
    print("MLP: ", p_value1)
    print("LSTM: ", p_value2)
    print("CNN: ", p_value3)
    


#A function to load models and plot best and worst
if __name__ == "__main__":
    lstmString = "LSTM_13.384791__NH1_64__NH2_64__DROP1_0.3__DROP2_0.3__EPOCHS_1400__BS_64__LR_0.001__PAT_300__DAY_False__WEEK_True"
    cnnString = "CNN_14.282786__NH1_64__NH2_256__NH3_512__FC_64__DROP1_0__DROP2_0__DROP3_0__EPOCHS_1400__BS_64__LR_0.001__PAT_300__DAY_False__WEEK_True"
    mlpString = "MLP_15.640279__NH1_1024__NH2_512__NH3_512__DROP1_0.5__DROP2_0.3__DROP3_0.3__EPOCHS_1200__BS_64__LR_0.001__PAT_300__DAY_False__WEEK_True"
    arimaString = "ARIMA_24.821584"

    lstmResult = pickle.load(open("./week/" + lstmString + ".pkl", "rb"))
    cnnResult = pickle.load(open("./week/" + cnnString + ".pkl", "rb"))
    mlpResult = pickle.load(open("./week/" + mlpString + ".pkl", "rb"))
    arimaResult = pickle.load(open("./week/" + arimaString + ".pkl", "rb"))

    #performSingleDMtest(lstmResult["dataDict"]["test"][1], cnnResult["dataDict"]["test"][1], mlpResult["dataDict"]["test"][1],arimaResult["dataDict"]["test"][1])

    performCompleteDMtest(lstmResult["dataDict"]["test"][1], cnnResult["dataDict"]["test"][1], mlpResult["dataDict"]["test"][1],arimaResult["dataDict"]["test"][1])


  