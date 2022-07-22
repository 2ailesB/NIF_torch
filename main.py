import os
import json
import torch
import logging
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib2 as Path
import time
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from datasets.wave_1d import Wave_1d
from models.nif_lastlayer import NIF_lastlayer
from models.nif_multiscale import NIF_multiscale
from models.simple_nif import simple_NIF
from utils.yaml import yaml2dict, dict2yaml

def main(path):

    cfg = yaml2dict(path)

    logging.basicConfig(level=logging.INFO)
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = f"logs/{cfg['model']}-{cfg['dataset']}-{start_time}"
    writer = SummaryWriter(save_path)

    dict2yaml(save_path + '/cfg.yaml', cfg)

    # cfg['logger']            = writer
    # cfg['ckpt_save_path']    = None

    path = '../NIF_expe/datasets/data'
    dataset = Wave_1d(path, cfg['data_cfg']['normalize'])
    print(dataset)

    torch.manual_seed(cfg['training_cfg']['seed'])
    cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', cfg['device'])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg['training_cfg']['batch_size'], shuffle=True, num_workers=1)

    tic = time.time()
    model = simple_NIF(cfg['cfg_parameter_net'], cfg['cfg_shape_net']) if cfg['model']=='nif_simple' else NIF_lastlayer(cfg['cfg_parameter_net'], cfg['cfg_shape_net']) if cfg['model']=='nif_lastlayer' else NIF_multiscale(cfg['cfg_parameter_net'], cfg['cfg_shape_net']) if cfg['model']=='nif_multiscale' else ValueError
    model.fit(dataloader, n_epochs=cfg['training_cfg']['nepoch'], lr=cfg['training_cfg']['lr_init'], save_images_freq=cfg['training_cfg']['print_figure_epoch'])

    cfg['training_time'] = time.time() - tic

    print("cfg, save_path :", cfg, save_path)
    dict2yaml(save_path + '/cfg.yaml', cfg)

    return True

if __name__ == "__main__":
    cfg_path = 'config/nif_1dwave.yaml'
    print(main(cfg_path))