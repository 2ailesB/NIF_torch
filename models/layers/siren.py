import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SIREN(nn.Module):
    def __init__(self, in_features, out_features, bias = True, is_first = False, omega_0 = 30, **kwargs):
        """
        Paper: Implicit Neural Representation with Periodic Activ ation Function (SIREN
        :param in_features: number of input features
        :param out_features: number of output features
        :param bias: add a bias or not to the linear transformation
        :param is_first: first layer
        :param omega_0: pulsation of the sine activation
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class SIREN_parametrized(nn.Module):
    def __init__(self, in_features, out_features, layers=None, activation='gelu', dropout_rate=0.0, **kwargs):
        """
        Usual SIREN module
        :param in_features: number of input features
        :param out_features: number of output features
        :param layers: list of hidden layer dimensions
        :param activation: activation function
        :param dropout_rate: dropout rate
        """
        super(SIREN_parametrized, self).__init__()
        self.layers = layers if layers is not None else []
        self.nlayers = len(layers)
        self.dim_in = in_features
        self.dim_out = out_features
        
        self.act = torch.sin() # TODO : init weight ? 

    def forward(self, x, out_hnet):
        cpt = 0
        W = out_hnet[0:self.dim_in * self.layers[0]]
        cpt = self.dim_in * self.layers[0]
        b = out_hnet[cpt : cpt + self.layers[0]] # TODO : if first layer/ last layer
        cpt += self.layers[0]
        for idx, layer in enumerate(self.layers):
            W = out_hnet[cpt:cpt + self.layers[idx] * self.layers[idx + 1]]
            cpt += self.layers[idx] * self.layers[idx + 1]
            b = out_hnet[cpt:cpt + self.layers[idx + 1]]
            cpt += self.layers[idx + 1]
            x = F.linear(x, W, b)
            if idx != self.nlayers - 1:
                x = self.act(x)
        return x