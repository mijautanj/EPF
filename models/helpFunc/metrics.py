import numpy as np 

def MAE(p_real, p_pred):
    return np.mean(np.abs(p_real - p_pred))

def sMAPE(p_real, p_pred):
    return np.mean(np.abs(p_real - p_pred) / ((np.abs(p_real) + np.abs(p_pred)) / 2)) * 100

def MAPE(p_real, p_pred):
    return np.mean(np.abs(p_real - p_pred) / np.abs(p_real) ) * 100

def RMSE(p_real, p_pred):
    return np.sqrt(np.mean((p_real - p_pred)**2))