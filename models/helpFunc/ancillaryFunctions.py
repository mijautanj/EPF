from helpFunc.metrics import *
import pickle

def evaluate(ytest):
    test_smape = []
    test_mae = []
    test_mape = []
    test_rmse = []
    for i in ytest: #For output in validation set 
        test_smape.append(sMAPE(i["Target"],i["Pred-Unscaled"]))
        test_mae.append(MAE(i["Target"],i["Pred-Unscaled"]))
        test_mape.append(MAPE(i["Target"],i["Pred-Unscaled"]))
        test_rmse.append(RMSE(i["Target"],i["Pred-Unscaled"]))


    errors = {
    "test_mae": np.mean(test_mae),
    "test_smape": np.mean(test_smape),
    "test_rmse": np.mean(test_rmse),
    "test_mape": np.mean(test_mape)
    }

    return test_mae, errors


def minMaxLoss(testMAE, k=2):
    #testMAE = [5,4,3,2,7,6,4,9,2,3]
    A = np.array(testMAE)
    idxMin = np.argpartition(A, k)
    idxMax = np.argpartition(A, -k)

    return idxMin[:k], idxMax[-k:]


def saveModel(dataDict, targetName, modelName, indicesMin, indicesMax, errors, finalValLoss, paramDict):
    modelstring = modelName + "_" + str(round(finalValLoss, 6))
    for key, value in paramDict.items():
        modelstring += "__" + str(key) + "_" + str(value)

    saveDict = {"dataDict": dataDict,
            "targetName": targetName,
            "modelName": modelName,
            "indicesMin": indicesMin,
            "indicesMax": indicesMax,
            "errors": errors,
            "finalValLoss": finalValLoss,
            "paramDict":  paramDict}
    
    file = open("../savedModels/" + modelstring + ".pkl", "wb")
    pickle.dump(saveDict, file)
    file.close()
