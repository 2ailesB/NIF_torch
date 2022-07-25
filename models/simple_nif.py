import torch
import torch.nn as nn
from Core.training import PytorchNIF

from models.nif import NIF

class simple_NIF(PytorchNIF):
    def __init__(self, cfg, criterion='mse', logger=None, opt='adam', device='cpu', ckpt_save_path=None, tag='', visual=None):
        super().__init__(criterion=criterion, logger=logger, opt=opt, device=device, ckpt_save_path=ckpt_save_path, tag=tag, visual=visual)

        self.cfg_data = cfg['data_cfg']
        self.cfg_training = cfg['training_cfg']
        self.cfg_parameter_net = cfg['cfg_parameter_net']
        self.cfg_shape_net = cfg['cfg_shape_net']

        self.cfg_shape_net['layers'] =[self.cfg_shape_net['units']]*(self.cfg_shape_net['nlayers'] + 1)
        self.cfg_shape_net['type'] = 'mlp'
        self.cfg_parameter_net['layers'] = [self.cfg_parameter_net['units']]*(self.cfg_parameter_net['nlayers'] + 1)
        self.cfg_parameter_net['type'] = 'mlp'
        
        self.model = NIF(self.cfg_parameter_net, self.cfg_shape_net)
        self.input_shape = self.cfg_parameter_net['input_dim'] + self.cfg_shape_net['input_dim']

    def forward(self, x):
        pass
