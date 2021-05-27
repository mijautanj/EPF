
from helpFunc.metrics import *
import matplotlib.pyplot as plt


def plotLossFunction(model, modelName):
    loss = model.history["loss"]
    val_loss = model.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(modelName + " training and validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('../plots/' + modelName +'_loss.png')
    #plt.show()


def plotPred(df,targetName, title, directStr):
    #Plot actual values of week-df
    ax = df[['Target', 'DateTime']].plot(x='DateTime', legend="observed", color='#174D7F',figsize=(16,5.5))
    #Plot predicted values of week-df
    df.plot(x='DateTime', y='Pred-Unscaled', ax=ax, color="r")

    ax.set_xlabel('Date')
    ax.set_ylabel('SEK/MWh')
    ax.legend(['Observed values','Prediction'])

    plt.title(title + " predicted 10-days ahead for " + targetName)
    plt.tight_layout() 
    plt.savefig('../plots/' + directStr +'.png')
    #plt.show()


def plotWorstBest(dataDict, targetName, modelName, minIndeces, maxIndeces):
    startString = ["", "Second ", "Third ", "Fourth "]
    for i in range (len(minIndeces)):
        goodPred = dataDict["test"][1][minIndeces[i]]
        badPred = dataDict["test"][1][maxIndeces[i]]
        titleGood = startString[i] + "Best" 
        titleBad = startString[i] + "Worst" 
        plotPred(goodPred, targetName, titleGood, modelName+"_pred_good" + str(i+1))
        plotPred(badPred, targetName, titleBad, modelName+"_pred_bad" + str(i+1))
    
    

def plotAllPred(dataDict, targetName, modelName):
    for i, j in enumerate(dataDict["test"][1]):
        plotPred(j, targetName, " EXAMPLE: "+str(i+1), "/massplots/" + str(i+1) + " " + modelName+"_pred_example")
    