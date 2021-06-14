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
    df =  pd.DataFrame(data, columns=["MLP","LSTM","CNN"])
    df.index.name = 'TEST SAMPLE'
    print(df)
    #sns.set(style='ticks', context='talk')
  

    #colors=['#FF8600','#27187E','#246A73']
    sns.set_palette(['#04724D', '#FF8600','#63326E'])
    plt.figure(figsize=(15,5.5))

    ax = sns.swarmplot(data=df)
    sns.set(font_scale=2)
    ax.set(ylabel='MAE', title="Swarm plot for test loss of each model")
    plt.tight_layout() 
    plt.savefig('losses-snsDAY.png')
    
   
    plt.show()


def plotPred(lstmTestTarget, cnnTestTarget, mlpTestTarget, dirStr):
    #Plot actual values of week-df
    #color='#174D7F'
    
    #Plot predicted values of week-df
    ax = lstmTestTarget.plot(x='DateTime', y='Pred-Unscaled', label="LSTM",  color='#FF8600', style='--',figsize=(16,5.5))
    cnnTestTarget.plot(x='DateTime', y='Pred-Unscaled', label="CNN", ax=ax, color='#63326E', style='-.')
    mlpTestTarget.plot(x='DateTime', y='Pred-Unscaled', label="MLP", ax=ax, color='#04724D', style=':')
    ax = lstmTestTarget[['Target', 'DateTime']].plot(x='DateTime', ax=ax, label="Observed value", color='#174D7F',figsize=(16,5.5))


    ax.set_xlabel('Date')
    ax.set_ylabel('SEK/MWh')
    ax.legend(["LSTM", "CNN", "MLP", "Observed value",])
    #ax.set_ylim([150,350])

    plt.title(dirStr + " overall predicted 10-days ahead for SE1")
    plt.tight_layout() 
    plt.savefig(dirStr +'.png')


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
    #plotPred(lstmTestTarget[maxIdx], cnnTestTarget[maxIdx], mlpTestTarget[maxIdx], "Worst")
    #plotPred(lstmTestTarget[minIdx], cnnTestTarget[minIdx], mlpTestTarget[minIdx], "Best")
    
    plotLoss(lstmLoss, cnnLoss, mlpLoss)

    



#A function to load models and plot best and worst
if __name__ == "__main__":
    lstmString = "LSTM_0.015264__NH1_64__NH2_64__DROP1_0.3__DROP2_0.3__EPOCHS_800__BS_64__LR_0.001__PAT_300__DAY_True__WEEK_False"
    cnnString = "CNN_19.65366__NH1_64__NH2_256__NH3_512__FC_64__DROP1_0__DROP2_0__DROP3_0__EPOCHS_1400__BS_64__LR_0.001__PAT_300__DAY_True__WEEK_False"
    mlpString = "MLP_17.344015__NH1_1024__NH2_512__NH3_512__DROP1_0.5__DROP2_0.3__DROP3_0.3__EPOCHS_1300__BS_64__LR_0.001__PAT_300__DAY_True__WEEK_False"

    lstmResult = pickle.load(open("./day/" + lstmString + ".pkl", "rb"))
    cnnResult = pickle.load(open("./day/" + cnnString + ".pkl", "rb"))
    mlpResult = pickle.load(open("./day/" + mlpString + ".pkl", "rb"))

    plotAllTestLosses(lstmResult["dataDict"]["test"][1], cnnResult["dataDict"]["test"][1], mlpResult["dataDict"]["test"][1])