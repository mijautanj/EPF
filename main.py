import pickle
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 20})

def MAE(p_real, p_pred):
    return np.mean(np.abs(p_real - p_pred))


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


def plotWorstBest(dataDict, targetName, modelName, minIndeces, maxIndeces):
    startString = ["", "", "Third ", "Fourth "]
    for i in range (len(minIndeces)):
        goodPred = dataDict["test"][1][minIndeces[i]]
        badPred = dataDict["test"][1][maxIndeces[i]]
        titleGood = startString[i] + "Best" 
        titleBad = startString[i] + "Worst" 
        

        testErrorGood = MAE(dataDict["test"][1][minIndeces[i]]["Target"], dataDict["test"][1][minIndeces[i]]["Pred-Unscaled"])
        testErrorBad = MAE(dataDict["test"][1][maxIndeces[i]]["Target"], dataDict["test"][1][maxIndeces[i]]["Pred-Unscaled"])
        directStrGood = modelName+"_pred_good" + str(i+1) + "_MAE_" + str(np.round(testErrorGood,3))
        directStrBad = modelName+"_pred_bad" + str(i+1) + "_MAE_" + str(np.round(testErrorBad,3))

        plotPred(goodPred, targetName, titleGood, directStrGood + str(i+1))
        plotPred(badPred, targetName, titleBad, directStrBad + str(i+1))


#A function to load models and plot best and worst
if __name__ == "__main__":
    modelstring = "MLP_15.783148__NH1_1024__NH2_1024__NH3_512__DROP1_0.3__DROP2_0.3__DROP3_0.3__EPOCHS_1500__BS_64__LR_0.001__PAT_300__DAY_False__WEEK_True"
    
    file = open(modelstring + ".pkl", "rb")
    output = pickle.load(file)
   # print("FINAL VALLOSS", output["finalValLoss"])
    #print("TEST sMAPE", output["errors"]["test_smape"] )
    print("ERROR", output["errors"] )
    plotWorstBest(output["dataDict"], output["targetName"], output["modelName"], 
        output["indicesMin"], output["indicesMax"])