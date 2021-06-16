import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random

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
    

    sns.set_palette(['#04724D', '#FF8600','#63326E'])
    plt.figure(figsize=(15,5.5))

    ax = sns.swarmplot(data=df)
    sns.set(font_scale=2)
    ax.set(ylabel='MAE', title="Swarm plot for test loss of each model")
    plt.tight_layout() 
    plt.savefig('./plots/losses-.png')
    
    plt.show()
    plt.close()


def plotSeveralPred(lstmTestTarget, cnnTestTarget, mlpTestTarget, dirStr):
    ax = lstmTestTarget.plot(x='DateTime', y='Pred-Unscaled', label="LSTM",  color='#FF8600', style='--',figsize=(16,5.5))
    cnnTestTarget.plot(x='DateTime', y='Pred-Unscaled', label="CNN", ax=ax, color='#63326E', style='-.')
    mlpTestTarget.plot(x='DateTime', y='Pred-Unscaled', label="MLP", ax=ax, color='#04724D', style=':')
    lstmTestTarget[['Target', 'DateTime']].plot(x='DateTime', ax=ax, label="Observed value", color='#174D7F',figsize=(16,5.5))


    ax.set_xlabel('Date')
    ax.set_ylabel('SEK/MWh')
    ax.legend(["LSTM", "CNN", "MLP", "Observed value",])

    plt.title(dirStr + " overall predicted 10-days ahead for SE1")
    plt.tight_layout() 
    plt.savefig("./plots/overall-" + dirStr +'.png')
    plt.show()


def plotPred(df, modelName, no):
    ax = df[['Target', 'DateTime']].plot(x='DateTime', legend="observed", color='#174D7F',figsize=(16,5.5))
    #Plot predicted values of week-df
    df.plot(x='DateTime', y='Pred-Unscaled', ax=ax, color="r", style='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('SEK/MWh')
    ax.legend(['Observed values','Prediction'])
    #ax.set_ylim([150,350])

    plt.title("Random 10-days ahead prediction of " + modelName + " number: " + no)
    plt.tight_layout() 
    plt.savefig("./plots/random/" + modelName + "-" + no +'.png')



def plotResults(lstmTestTarget, cnnTestTarget, mlpTestTarget, plotAllLosses=True, plotBestWorst=True, plotRandom=True):
    lstmLoss = []
    cnnLoss = []
    mlpLoss = []

    for i in range(len(lstmTestTarget)):
        lstmLoss.append(MAE(lstmTestTarget[i]["Target"], lstmTestTarget[i]["Pred-Unscaled"]))
        cnnLoss.append(MAE(cnnTestTarget[i]["Target"], cnnTestTarget[i]["Pred-Unscaled"]))
        mlpLoss.append(MAE(mlpTestTarget[i]["Target"], mlpTestTarget[i]["Pred-Unscaled"]))

    if plotAllLosses:
        plotLoss(lstmLoss, cnnLoss, mlpLoss)


    if plotBestWorst:
        totalLoss = np.array(lstmLoss)+np.array(cnnLoss)+np.array(mlpLoss)
        maxIdx = np.argmax(totalLoss)
        minIdx = np.argmin(totalLoss)
        plotSeveralPred(lstmTestTarget[maxIdx], cnnTestTarget[maxIdx], mlpTestTarget[maxIdx], "Worst")
        plotSeveralPred(lstmTestTarget[minIdx], cnnTestTarget[minIdx], mlpTestTarget[minIdx], "Best")
    
    #random.seed(10) #SKITBRA!!!
    random.seed(77) #SKITBRA!!!
    #random.seed(6) #helt ok
    #random.seed(14)
    if plotRandom:
        targets = {"LSTM": lstmTestTarget, "CNN": cnnTestTarget}
        for key, value in targets.items():
            print(key)
            randomIdx = random.sample(range(0, len(lstmTestTarget)), 3)
            print(randomIdx)
            for j in range(len(randomIdx)):
                plotPred(value[randomIdx[j]], key, str(j+1))
        

    

    



#A function to load models and plot best and worst, and their losses
if __name__ == "__main__":
    lstmString = "LSTM_13.384791__NH1_64__NH2_64__DROP1_0.3__DROP2_0.3__EPOCHS_1400__BS_64__LR_0.001__PAT_300__DAY_False__WEEK_True"
    cnnString = "CNN_14.282786__NH1_64__NH2_256__NH3_512__FC_64__DROP1_0__DROP2_0__DROP3_0__EPOCHS_1400__BS_64__LR_0.001__PAT_300__DAY_False__WEEK_True"
    mlpString = "MLP_15.640279__NH1_1024__NH2_512__NH3_512__DROP1_0.5__DROP2_0.3__DROP3_0.3__EPOCHS_1200__BS_64__LR_0.001__PAT_300__DAY_False__WEEK_True"

    lstmResult = pickle.load(open("./week/" + lstmString + ".pkl", "rb"))
    cnnResult = pickle.load(open("./week/" + cnnString + ".pkl", "rb"))
    mlpResult = pickle.load(open("./week/" + mlpString + ".pkl", "rb"))

    plotResults(lstmResult["dataDict"]["test"][1], cnnResult["dataDict"]["test"][1], mlpResult["dataDict"]["test"][1],plotAllLosses=False, plotBestWorst=False, plotRandom=True)