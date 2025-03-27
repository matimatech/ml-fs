import numpy as np

def Gini(y):
    K, counts = np.unique(y, return_counts=True)
    unique_counts = dict(zip(K, counts))
    N_m = len(y)

    # Calculate the proportion of class k observations in node m
    p_m = {}
    for k in K:
        p_m[k] = unique_counts[k] / N_m
    
    # calculate the node impurity
    node_impurity = 0
    for k in K:
        node_impurity += p_m[k] * (1-p_m[k])

    return node_impurity

def Log_Loss(y):

    # Extract class
    K, counts = np.unique(y, return_counts=True)
    unique_counts = dict(zip(K, counts))
    N_m = len(y)

    # Calculate the proportion of class k observations in node m
    p_m = {}
    for k in K:
        p_m[k] = unique_counts[k] / N_m

    # Find the majority class in node m
    ind_max = np.argmax(counts)
    class_max = K[ind_max]

    # calculate the node impurity
    node_impurity = 1 - p_m[class_max]

    return node_impurity

def Entropy(y):
    K, counts = np.unique(y, return_counts=True)
    unique_counts = dict(zip(K, counts))
    N_m = len(y)

    # calculate the proportion of class k observations in node m
    p_m = {}
    for k in K:
        p_m[k] = unique_counts[k] / N_m

    # calculate the node impurity
    node_impurity = 0
    for k in K:
        node_impurity += p_m[k] * np.log(p_m[k])

    return node_impurity

# Regression impurity
def MSE(y):
    N_m = len(y)

    # Calculate the bset prediction (c) --> Eq. (9.11)
    c_m = (1/N_m) * np.sum(y)

    # Calculate the node impurity (variance)
    node_impurity = 0
    for i in range(N_m):
        node_impurity += (y[i] - c_m)**2

    node_impurity *= (1/N_m)

    return node_impurity

def MAE(y):
    N_m = len(y)
    c_m = np.median(y)

    node_impurity = 0
    for i in range(N_m):
        node_impurity += np.abs(y[i] - c_m)

    node_impurity *= (1/N_m)

    return node_impurity

class Tree:
    def __init__(self):







