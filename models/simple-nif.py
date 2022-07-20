import torch
import torch.nn as nn

from nif import NIF

class simple_NIF(nn.Module):
    def __init__(self, cfg_parameter_net, cfg_shape_net):
        super().__init__()

        self.model = NIF(cfg_parameter_net, cfg_shape_net)

    def forward(self, x):
        return self.model(x)
