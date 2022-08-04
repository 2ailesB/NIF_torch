from matplotlib.pyplot import axis
import torch

def std_normalize(data):
    means = data.mean(0)
    stds = data.std(0)
    if (stds == 0).nonzero().sum() >= 1:
        stds[(stds == 0).nonzero()] = 1.0
        
    data -= means
    data /= stds

    return data, means, stds

def minimax_normalize(data):
    mins = data.min(0)
    maxs = data.max(0)
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    
    data -= mins
    data /= (maxs-mins)

    return data, means, stds