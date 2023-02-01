import numpy as np
from sklearn import metrics
def mape(y_true, y_pred):
    return np.mean(np.abs(y_pred-y_true)/y_true)*100

def rmse(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_pred,y_true))

