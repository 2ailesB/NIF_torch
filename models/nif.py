import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.mlp import MLP, MLP_parametrized
from models.layers.siren import SIREN, SIREN_parametrized
from Core.training import PytorchNIF

class NIF(PytorchNIF):
    def __init__(self, cfg_parameter_net, cfg_shape_net):
        super().__init__()
        self.cfg_shape_net = cfg_shape_net
        self.cfg_parameter_net = cfg_parameter_net
        self.input_shape = cfg_parameter_net['input_dim'] + cfg_shape_net['input_dim']

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
        
        if self.cfg_parameter_net['type'] =='mlp':
            self.parameter_net = MLP(self.cfg_parameter_net['input_dim'], self.cfg_parameter_net['latent_dim'], self.cfg_parameter_net['layers'], self.cfg_parameter_net['activation'])
        elif self.cfg_parameter_net['type'] == 'siren' :
            self.parameter_net = SIREN(self.cfg_parameter_net['input_dim'], self.cfg_parameter_net['latent_dim'], self.cfg_parameter_net['layers'], self.cfg_parameter_net['activation'])
        else :
            raise NotImplementedError('NIF not implemented for this kind of layer, please use mlp or siren')
        if self.cfg_shape_net['type'] =='mlp':
            self.shape_net = MLP_parametrized(self.cfg_shape_net['input_dim'], self.cfg_shape_net['output_dim'], self.cfg_shape_net['layers'], self.cfg_shape_net['activation'])
        elif self.cfg_shape_net['type']=='siren':
            self.shape_net = SIREN_parametrized(self.cfg_shape_net['input_dim'], self.cfg_shape_net['output_dim'], self.cfg_shape_net['layers'], self.cfg_shape_net['activation'])
        else :
            raise NotImplementedError('NIF not implemented for this kind of layer, please use mlp or siren')

    def forward(self, x):
        x_parameter = x[:, :self.cfg_parameter_net['input_dim']]
        x_shape = x[:, -self.cfg_shape_net['input_dim']:] # [bxatch-size, input dim]

        y1 = self.parameter_net(x_parameter) # [bxatch-size, input dim]
        # print("y1.shape :", y1.shape)
        rom = self.hnet(y1) # [bxatch-size, latent dim]
        # print("rom.shape :", rom.shape)
        yhat = self.shape_net(x_shape, rom) # [bxatch-size, hnet out]
        # print("yhat.shape :", yhat.shape)

        return yhat
