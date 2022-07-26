import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init import init_weights_truncNorm

class MLP(nn.Module):
    def __init__(self, in_features, out_features, layers=None, activation='gelu', dropout_rate=0.0, **kwargs):
        """
        Usual MLP module
        :param in_features: number of input features
        :param out_features: number of output features
        :param layers: list of hidden layer dimensions
        :param activation: activation function
        :param dropout_rate: dropout rate
        """
        super(MLP, self).__init__()
        self.layers = layers if layers is not None else []
        self.model = nn.ModuleList([
            nn.Sequential(nn.Linear(lp, lnext), nn.Dropout(dropout_rate))
            for lp, lnext in zip([in_features] + self.layers, self.layers + [out_features])
            ])
            # bias_initializer=initializers.TruncatedNormal(stddev=0.1)
        self.model.apply(init_weights_truncNorm)
        self.act = torch.nn.ReLU() if activation == 'relu' else torch.nn.Tanh() if activation == 'tanh' else torch.nn.Sigmoid() if activation == 'sigmoid' else torch.nn.GELU() if activation == 'gelu' else torch.nn.SiLU() if activation == 'swish' else ValueError

    def forward(self, x):
        for idx, layer in enumerate(self.model):
            x = layer(x)
            if idx != len(self.model) - 1:
                x = self.act(x)
        return x

class MLP_parametrized(nn.Module):
    def __init__(self, in_features, out_features, layers=None, activation='gelu', dropout_rate=0.0, **kwargs):
        """
        Usual MLP module
        :param in_features: number of input features
        :param out_features: number of output features
        :param layers: list of hidden layer dimensions
        :param activation: activation function
        :param dropout_rate: dropout rate
        """
        super(MLP_parametrized, self).__init__()
        self.layers = layers if layers is not None else []
        self.nlayers = len(layers)
        self.dim_in = in_features
        self.dim_out = out_features
        
        self.act = torch.nn.ReLU() if activation == 'relu' else torch.nn.Tanh() if activation == 'tanh' else torch.nn.Sigmoid() if activation == 'sigmoid' else torch.nn.GELU() if activation == 'gelu' else torch.nn.SiLU() if activation == 'swish' else ValueError

    def forward(self, x, out_hnet):
        cpt = 0
        b_size = out_hnet.shape[0]
        # print("self.layers :", self.layers)
        for idx, (lp, lnext) in enumerate(zip([self.dim_in] + self.layers, self.layers + [self.dim_out])): # out_hnet : torch.Size([512, 1951])
            din = lp
            dout = lnext
            # print("din, dout :", din, dout)
            W = out_hnet[:, cpt:cpt + din * dout].reshape(b_size, dout, din) # torch.Size([512, 1951])
            cpt += din * dout
            b = out_hnet[:, cpt:cpt + dout] # torch.Size([512, 1951])
            cpt += dout
            x = torch.einsum('bi, bji -> bj', x, W) + b
            # print("x.shape :", x.shape)
            # print("idx :", idx)
            # print("self.nlayers :", self.nlayers)
            # print("cpt :", cpt)
            if idx != self.nlayers: # stop at last layer between layer[n-1] and dim_out
                x = self.act(x)
            if idx == self.nlayers:
                return x
