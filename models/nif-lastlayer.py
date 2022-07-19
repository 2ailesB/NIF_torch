import torch
import torch.nn as nn

from models.layers.mlp import MLP
from models.layers.siren import SIREN

class NIF_lastlayer(nn.Module):
    def __init__(self, cfg_shape_net, cfg_parameter_net):
        super().__init__()
        self.cfg_shape_net = cfg_shape_net
        self.cfg_parameter_net = cfg_parameter_net

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