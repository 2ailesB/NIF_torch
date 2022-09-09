import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from utils.preprocessing import minimax_normalize, std_normalize, nif_normalize

class NavierStockes(Dataset) : #héritage de classe Dataset de Pytorch
    def __init__(self, path, start=0, end=191775, normalize='standard', train=False):
        super().__init__()
        # TODO : add end = len(dataset)
        data = np.load(path + '/cylinderflow.npz')['data']
        datax = data[:, [0, 1, 2]] # parameter (t) = 0, shape (x, y) = (1, 2)
        datay = data[:, [3, 4]]

        self.datax = torch.tensor(datax).float()#.view(size) permet de modifier la shape et d'utiliser le même espace de stockage
        self.datay = torch.tensor(datay).float() #labels
        self.nsamples = self.datax.shape[0]

        self.mode = 'train'*train + 'val'*(1-train)

        self.sample_weight = data[start:end, -1:]

        if normalize == 'standard':
            self.datax, datax_means, datax_stds = std_normalize(self.datax)
            self.datay, datay_means, datay_stds = std_normalize(self.datay)
            self.datax = self.datax[start:end, :]
            self.datay = self.datay[start:end, :]
            self.means = torch.cat((datax_means, datay_means), dim=0)
            self.stds = torch.cat((datax_stds, datay_stds), dim=0)
        elif normalize == 'minmax':
            self.datax, datax_means, datax_stds = minimax_normalize(self.datax)
            self.datay, datay_means, datay_stds = minimax_normalize(self.datay)
            self.datax = self.datax[start:end, :]
            self.datay = self.datay[start:end, :]
            self.means = torch.cat((datax_means, datay_means), dim=0)
            self.stds = torch.cat((datax_stds, datay_stds), dim=0)
        elif normalize=='nif':
            self.data, self.means, self.stds = nif_normalize(torch.cat((self.datax, self.datay), dim=1), 3, 2)
            self.datax = self.data[start:end, 0:3]
            self.datay = self.data[start:end, 3:5]
        else :
            raise NotImplementedError('Normalization not implemented') #TODO : rescalinf test data le même que train data

        # print("self.means :", self.means)
        # print("self.stds :", self.stds)

    def __getitem__(self ,index ):
        """retourne un couple (exemple,label) correspondant à l’index"""

        return self.datax[index, :], self.datay[index, :]
        
    def __len__(self):
        """renvoie la taille du jeu de donnees"""
        return self.datax.shape[0]

    def nif_normalize(data):
        mean, _ = data.mean(axis=0)
        std, _ = data.std(axis=0)
        print("normalize nif :")

        for i in range(3):
            mean[i] = 0.5*(np.min(data[:,i])+np.max(data[:,i]))
            std[i] = 0.5*(-np.min(data[:,i])+np.max(data[:,i]))

        # also we normalize the output target to make sure the maximal is most 1
        for j in range(3, 5):
            std[j] = np.max(np.abs(data[:,j]))

        normalized_data = (data - mean)/std
        return normalized_data, mean, std

