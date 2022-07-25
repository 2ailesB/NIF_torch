import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from utils.preprocessing import minimax_normalize, std_normalize

class Wave_1d(Dataset) : #héritage de classe Dataset de Pytorch
    def __init__(self, path, start=0, end=2000, normalize='standard', train=False):
        super().__init__()

        data = np.load(path + '/traveling_wave.npz')['data']
        datax = data[start:end, [0, 1]] # TODO
        datay = data[start:end, [2]]

        self.datax = torch.tensor(datax).float()#.view(size) permet de modifier la shape et d'utiliser le même espace de stockage
        self.datay = torch.tensor(datay) #labels
        self.nsamples = self.datax.shape[0]

        self.mode = 'train'*train + 'val'*(1-train)

        if normalize == 'standard':
            self.datax = std_normalize(self.datax)
            self.datay = std_normalize(self.datay)
        elif normalize == 'minmax':
            self.datax = minimax_normalize(self.datax)
            self.datay = minimax_normalize(self.datay)
        else :
            raise NotImplementedError('Normalization not implemented')

    def __getitem__(self ,index ):
        """retourne un couple (exemple,label) correspondant à l’index"""

        return self.datax[index, :], self.datay[index, :]
        
    def __len__(self):
        """renvoie la taille du jeu de donnees"""
        return self.datax.shape[0]
