import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.mlp import MLP, MLP_parametrized
from layers.siren import SIREN, SIREN_parametrized

class NIF(nn.Module):
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
        # in_features, out_features, layers=None, activation='gelu'
        self.cfg_hnet = dict()
        self.cfg_hnet['dim_out'] = nweights + nbias
        self.cfg_hnet['dim_in'] = self.cfg_parameter_net['latent_dim'] 
        self.hnet = MLP(self.cfg_hnet['dim_in'], self.cfg_hnet['dim_out']) #TODO

        # TODO
        if self.cfg_parameter_net[''] =='mlp':
            self.parameter_net = MLP(self.cfg_parameter_net['input_dim'], self.cfg_parameter_net['output_dim'], self.cfg_parameter_net['layers'], self.cfg_parameter_net['activation'])
        else :
            self.parameter_net = SIREN(self.cfg_parameter_net['input_dim'], self.cfg_parameter_net['output_dim'], self.cfg_parameter_net['layers'], self.cfg_parameter_net['activation'])
        if self.cfg_shape_net[''] =='mlp':
            self.shape_net = MLP_parametrized(self.cfg_shape_net['input_dim'], self.cfg_shape_net['output_dim'], self.cfg_shape_net['layers'], self.cfg_shape_net['activation'])
        else :
            self.shape_net = SIREN_parametrized(self.cfg_shape_net['input_dim'], self.cfg_shape_net['output_dim'], self.cfg_shape_net['layers'], self.cfg_shape_net['activation'])

    def forward(self, x):
        x_parameter = x[:self.cfg_parameter_net['input_dim']]
        x_shape = x[-self.cfg_shape_net['input_dim']:]

        y1 = self.parameter_net(x_parameter)
        rom = self.hnet(y1)
        yhat = self.shape_net(x_shape, rom)

        return yhat
