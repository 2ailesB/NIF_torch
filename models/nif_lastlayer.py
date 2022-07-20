import torch
import torch.nn as nn
import torch.nn.functional as F
from Core.training import PytorchNIF

from layers.mlp import MLP
from layers.siren import SIREN
from Core.training import PytorchNIF

class NIF_lastlayer(PytorchNIF):
    def __init__(self, cfg_shape_net, cfg_parameter_net):
        super().__init__()
        self.cfg_shape_net = cfg_shape_net
        self.cfg_parameter_net = cfg_parameter_net

        # compute the number of weights to be computed by parameter net
        nweights = self.cfg_shape_net['input_dim'] * self.cfg_shape_net['units'] + \
                self.cfg_shape_net['units'] * self.cfg_shape_net['units'] * self.cfg_shape_net['nlayers'] + \
                self.cfg_shape_net['units'] * self.cfg_shape_net['output_dim']
        nbias = self.cfg_shape_net['units'] + \
                self.cfg_shape_net['units'] * self.cfg_shape_net['nlayers'] + \
                self.cfg_shape_net['output_dim']

        self.cfg_hnet = dict()
        self.cfg_hnet['dim_out'] = nweights + nbias
        self.cfg_hnet['dim_in'] = self.cfg_parameter_net['latent_dim'] 
        self.hnet = MLP(self.cfg_hnet) #TODO

        # TODO
        if self.cfg_parameter_net[''] =='mlp':
            self.parameter_net = MLP(self.cfg_parameter_net[''])
        else :
            self.parameter_net = SIREN(self.cfg_parameter_net[''])
        if self.cfg_shape_net[''] =='mlp':
            self.shape_net = MLP(self.cfg_shape_net[''])
        else :
            self.shape_net = SIREN(self.cfg_shape_net[''])

    def forward(self, x):
        x_parameter = x[:self.cfg_parameter_net['input_dim']]
        x_shape = x[self.cfg_shape_net['input_dim']]

        y1 = self.parameter_net(x_parameter)
        y2 = self.shape_net(x_shape)

        yhat = y1 @ y2

        return yhat
