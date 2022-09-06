from matplotlib.pyplot import axis
import torch

def std_normalize(data):
    means = data.mean(0)
    stds = data.std(0)
    if (stds == 0).nonzero().sum() >= 1:
        stds[(stds == 0).nonzero()] = 1.0
        
    data -= means
    data /= stds
    # means : tensor([1.0821e-01, 1.4988e-02, 5.6368e-05])
    # stds : tensor([0.0029, 0.0139, 0.0093])
 
    return data, means, stds

def minimax_normalize(data):
    mins, _ = data.min(0)
    maxs, _ = data.max(0)
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    
    data -= mins
    data /= (maxs-mins)

    return data, means, stds

def nif_normalize(data, din, dout):
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    for i in range(din):
        mean[i] = 0.5*(torch.min(data[:,i])+torch.max(data[:,i]))
        std[i] = 0.5*(-torch.min(data[:,i])+torch.max(data[:,i]))

    # also we normalize the output target to make sure the maximal is most 1
    for j in range(din, din+dout):
        std[j] = torch.max(torch.abs(data[:,j]))

    normalized_data = (data - mean)/std
    return normalized_data, mean, std