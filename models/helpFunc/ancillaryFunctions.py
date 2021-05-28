from helpFunc.metrics import *

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