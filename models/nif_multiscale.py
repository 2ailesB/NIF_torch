import torch
import torch.nn as nn

from models.nif import NIF
from Core.training import PytorchNIF
from models.nif import NIF

class NIF_multiscale(NIF):
    def __init__(self, cfg_parameter_net, cfg_shape_net):
        super().__init__()

        self.model = NIF(cfg_parameter_net, cfg_shape_net)

    def forward(self, x):
        return self.model(x)
