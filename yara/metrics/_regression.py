import numpy as np

def mean_squared_error(y_pred, y_actual):
    return np.mean((y_actual - y_pred)**2)

def mean_absolute_error(y_pred, y_actual):
    return np.mean(np.abs(y_actual - y_pred))