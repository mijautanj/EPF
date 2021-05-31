import pickle
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

def plotPred(df,targetName, title, directStr):
    #Plot actual values of week-df
    #color='#174D7F'
    ax = df[['Target', 'DateTime']].plot(x='DateTime', legend="observed", color='#174D7F',figsize=(16,5.5))
    #Plot predicted values of week-df
    df.plot(x='DateTime', y='Pred-Unscaled', ax=ax, color="r", style='--')

    ax.set_xlabel('Date')
    ax.set_ylabel('SEK/MWh')
    ax.legend(['Observed values','Prediction'])

    plt.title(title + " predicted 10-days ahead for " + targetName)
    plt.tight_layout() 
    plt.savefig("./savedModels/" + directStr +'.png')


def plotWorstBest(dataDict, targetName, modelName, minIndeces, maxIndeces):
    startString = ["", "Second ", "Third ", "Fourth "]
    for i in range (len(minIndeces)):
        goodPred = dataDict["test"][1][minIndeces[i]]
        badPred = dataDict["test"][1][maxIndeces[i]]
        titleGood = startString[i] + "Best" 
        titleBad = startString[i] + "Worst" 
        plotPred(goodPred, targetName, titleGood, modelName+"_pred_good" + str(i+1))
        plotPred(badPred, targetName, titleBad, modelName+"_pred_bad" + str(i+1))



#A function to load models and plot best and worst
if __name__ == "__main__":
    modelstring = "LSTM_0.015022__NH1_64__NH2_32__EPOCHS_800__BS_64__LR_0.0001__PAT_300__DAY_False__WEEK_True"
    file = open("./savedModels/" + modelstring + ".pkl", "rb")
    output = pickle.load(file)
    print("FINAL VALLOSS", output["finalValLoss"])
    print("TEST sMAPE", output["errors"]["test_smape"] )
    plotWorstBest(output["dataDict"], output["targetName"], output["modelName"], 
        output["indicesMin"], output["indicesMax"])