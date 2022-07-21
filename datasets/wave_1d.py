import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.preprocessing import minimax_normalize, std_normalize

class Wave_1d(Dataset) : #héritage de classe Dataset de Pytorch
    def __init__(self, path, normalize='standard'):
        super().__init__()

        data = np.load(path + '/traveling_wave.npz')['data']
        datax = data[:, [0, 1]] # TODO
        datay = data[:, [2]]

        self.datax = torch.tensor(datax).float()#.view(size) permet de modifier la shape et d'utiliser le même espace de stockage
        # self.datax /= torch.max(self.datax) #normalisation entre 0 et 1
        self.datay = torch.tensor(datay) #labels

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