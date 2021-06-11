import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update({'font.size': 20})

def MAE(p_real, p_pred):
    return np.mean(np.abs(p_real - p_pred))


def plotLoss(lstmLoss,cnnLoss,mlpLoss):
    data = {
        "MLP": mlpLoss,
        "LSTM": lstmLoss,
        "CNN": cnnLoss,        
    }
    df =  pd.DataFrame(data)
    df.index.name = 'TEST SAMPLE'
    print(df)
    sns.set(style='ticks', context='talk')
  

    #colors=['#FF8600','#27187E','#246A73']
    sns.set_palette(['#04724D', '#FF8600','#63326E'])
    plt.figure(figsize=(15,5.5))

    ax = sns.swarmplot(data=df)
    sns.set(font_scale=2)
    ax.set(ylabel='MAE', title="Swarmplot for test loss of each model")
    plt.tight_layout() 
    plt.savefig('losses-snsDAY.png')
    
   
    plt.show()


def plotPred(df,targetName, title, directStr):
    #Plot actual values of week-df
    #color='#174D7F'
    ax = df[['Target', 'DateTime']].plot(x='DateTime', legend="observed", color='#174D7F',figsize=(16,5.5))
    #Plot predicted values of week-df
    df.plot(x='DateTime', y='Pred-Unscaled', ax=ax, color="r", style='--')

    ax.set_xlabel('Date')
    ax.set_ylabel('SEK/MWh')
    ax.legend(['Observed values','Prediction'])
    #ax.set_ylim([150,350])

    plt.title(title + " predicted 10-days ahead for " + targetName)
    plt.tight_layout() 
    plt.savefig(directStr +'.png')


def plotAllTestLosses(lstmTestTarget, cnnTestTarget, mlpTestTarget):
    lstmLoss = []
    cnnLoss = []
    mlpLoss = []
    for i in range(len(lstmResult["dataDict"]["test"][1])):
        lstmLoss.append(MAE(lstmTestTarget[i]["Target"], lstmTestTarget[i]["Pred-Unscaled"]))
        cnnLoss.append(MAE(cnnTestTarget[i]["Target"], cnnTestTarget[i]["Pred-Unscaled"]))
        mlpLoss.append(MAE(mlpTestTarget[i]["Target"], mlpTestTarget[i]["Pred-Unscaled"]))

    
    totalLoss = np.array(lstmLoss)+np.array(cnnLoss)+np.array(mlpLoss)
    maxIdx = np.argmax(totalLoss)
    minIdx = np.argmin(totalLoss)

    



#A function to load models and plot best and worst
if __name__ == "__main__":
    lstmString = "LSTM_13.384791__NH1_64__NH2_64__DROP1_0.3__DROP2_0.3__EPOCHS_1400__BS_64__LR_0.001__PAT_300__DAY_False__WEEK_True"
    cnnString = "CNN_14.282786__NH1_64__NH2_256__NH3_512__FC_64__DROP1_0__DROP2_0__DROP3_0__EPOCHS_1400__BS_64__LR_0.001__PAT_300__DAY_False__WEEK_True"
    mlpString = "MLP_15.640279__NH1_1024__NH2_512__NH3_512__DROP1_0.5__DROP2_0.3__DROP3_0.3__EPOCHS_1200__BS_64__LR_0.001__PAT_300__DAY_False__WEEK_True"

    lstmResult = pickle.load(open("./week/" + lstmString + ".pkl", "rb"))
    cnnResult = pickle.load(open("./week/" + cnnString + ".pkl", "rb"))
    mlpResult = pickle.load(open("./week/" + mlpString + ".pkl", "rb"))

    plotAllTestLosses(lstmResult["dataDict"]["test"][1], cnnResult["dataDict"]["test"][1], mlpResult["dataDict"]["test"][1])