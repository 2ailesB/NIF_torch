import torch
import torch.nn as nn

def init_weights_truncNorm(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.trunc_normal_(m.weight, std=0.1)
        torch.nn.init.trunc_normal_(m.bias, std=0.1)