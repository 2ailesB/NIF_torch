import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Wave_1d(Dataset) : #héritage de classe Dataset de Pytorch
    def __init__(self, path):
        super().__init__()

        data_w = np.load(path + '/ns_train.npz')['u']
        datax = [] # TODO
        datay = []

        self.datax = torch.tensor(datax).view(datax.shape[0],-1).float()#.view(size) permet de modifier la shape et d'utiliser le même espace de stockage
        self.datax/=torch.max(self.datax) #normalisation entre 0 et 1
        self.datay = torch.tensor(datay).view(datay.shape[0],-1) #labels
    def __getitem__(self ,index ):
        """retourne un couple (exemple,label) correspondant à l’index"""
        return self.datax[index], self.datay[index]
    def __len__(self):
        """renvoie la taille du jeu de donnees"""
        return self.datax.shape[0]