
from helpFunc.metrics import *
import matplotlib.pyplot as plt


def minMaxLoss(self, Yp_test, Y_test):
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


def plotPred(df,targetName, title, directStr):
    #Plot actual values of week-df
    ax = df[['Target', 'DateTime']].plot(x='DateTime', legend="observed", color='#174D7F',figsize=(16,5.5))
    #Plot predicted values of week-df
    df.plot(x='DateTime', y='Pred-Unscaled', ax=ax, color="r")

    ax.set_xlabel('Date')
    ax.set_ylabel('SEK/MWh')
    ax.legend(['Observed values','Prediction'])

    plt.title(title + "for price area " + targetName)
    plt.tight_layout() 
    plt.savefig('../plots' + directStr +'.png')
    plt.show()