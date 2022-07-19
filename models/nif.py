import torch
import torch.nn as nn

from layers.mlp import MLP
from layers.siren import SIREN

class NIF(nn.Module):
    def __init__(self, cfg_shape_net, cfg_parameter_net):
        super().__init__()
        self.cfg_shape_net = cfg_shape_net
        self.cfg_parameter_net = cfg_parameter_net

        nweights = self.cfg_shape_net['input_dim'] * self.cfg_shape_net['units'] + \
                self.cfg_shape_net['units'] * self.cfg_shape_net['units'] * self.cfg_shape_net['nlayers'] + \
                self.cfg_shape_net['units'] * self.cfg_shape_net['output_dim']
        nbias = self.cfg_shape_net['units'] + \
                self.cfg_shape_net['units'] * self.cfg_shape_net['nlayers'] + \
                self.cfg_shape_net['output_dim']

        self.cfg_hnet['dim_out'] = nweights + nbias
        self.cfg_hnet['dim_in'] = self.cfg_parameter_net['latent_dim']

        if self.cfg_parameter_net[''] =='mlp':
            self.parameter_net = MLP(self.cfg_parameter_net[''])
        else :
            self.parameter_net = SIREN(self.cfg_parameter_net[''])
        if self.cfg_shape_net[''] =='mlp':
            self.shape_net = MLP(self.cfg_shape_net[''])
        else :
            self.shape_net = SIREN(self.cfg_shape_net[''])
    def forward(self):
        return None