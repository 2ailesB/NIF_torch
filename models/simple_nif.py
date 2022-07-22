import torch
import torch.nn as nn

from models.nif import NIF
from Core.training import PytorchNIF

class simple_NIF(NIF):
    def __init__(self, cfg_parameter_net, cfg_shape_net):
        cfg_shape_net['layers'] = [cfg_shape_net['input_dim']] + [cfg_shape_net['units']]*(cfg_shape_net['nlayers'] + 1) + [cfg_shape_net['output_dim']]
        cfg_shape_net['type'] = 'mlp'
        cfg_parameter_net['layers'] = [cfg_parameter_net['units']]*(cfg_parameter_net['nlayers'])
        cfg_parameter_net['type'] = 'mlp'
        
        super().__init__(cfg_parameter_net, cfg_shape_net)
        self.model = NIF(cfg_parameter_net, cfg_shape_net)

    def forward(self, x):
        return self.model(x)
