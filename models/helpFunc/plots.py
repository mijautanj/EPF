
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