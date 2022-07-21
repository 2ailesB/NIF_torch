import torch
import torch.nn as nn

from models.nif import NIF
from Core.training import PytorchNIF

class simple_NIF(NIF):
    def __init__(self, cfg_parameter_net, cfg_shape_net):
        super().__init__()
        cfg_shape_net['layers'] = [cfg_shape_net['units']]*cfg_shape_net['nlayers']
        cfg_parameter_net['layers'] = [cfg_parameter_net['units']]*cfg_parameter_net['nlayers']
        
        self.model = NIF(cfg_parameter_net, cfg_shape_net)

    def forward(self, x):
        return self.model(x)
