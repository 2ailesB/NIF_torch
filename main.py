import os
import json
import torch
import logging
import datetime
import matplotlib.pyplot as plt
import numpy as np
# import pathlib2 as Path
import time
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from datasets.wave_1d import Wave_1d
from datasets.wave_hf1d import Wave_1dhf
from models.nif_lastlayer import NIF_lastlayer
from models.nif_multiscale import multiscale_NIF
from models.simple_nif import simple_NIF
from utils.utils import count_params
from utils.visual import visual_1dwave
from utils.yaml import yaml2dict, dict2yaml
from utils.scheduler import nif_scheduler

def main(path):

    cfg = yaml2dict(path)

    logging.basicConfig(level=logging.INFO)
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = f"logs/{cfg['dataset']}/{cfg['model']}/{cfg['model']}-{cfg['dataset']}-{start_time}"
    writer = SummaryWriter(save_path)

    dict2yaml(save_path + '/cfg.yaml', cfg)

    # cfg['logger']            = writer
    # cfg['ckpt_save_path']    = None

    path = '../NIF_expe/datasets/data'
    dtrain = Wave_1d(path, 0, 1600, normalize=cfg['data_cfg']['normalize'])
    dtest = Wave_1d(path, 1600, 2000, normalize=cfg['data_cfg']['normalize'])
    dtrain = Wave_1dhf(path, 0, 1600, normalize=cfg['data_cfg']['normalize'])
    dtest = Wave_1dhf(path, 1600, 2000, normalize=cfg['data_cfg']['normalize'])
    print(dtrain)

    torch.manual_seed(cfg['training_cfg']['seed'])
    cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', cfg['device'])

    dataloader_train = torch.utils.data.DataLoader(dtrain, batch_size=cfg['training_cfg']['batch_size'], shuffle=True, num_workers=1)
    dataloader_test = torch.utils.data.DataLoader(dtest, batch_size=cfg['training_cfg']['batch_size'], shuffle=True, num_workers=1)

    tic = time.time()
    model = simple_NIF(cfg, logger=writer, device = cfg['device'], ckpt_save_path=save_path, visual=visual_1dwave) if cfg['model'] == 'nif_simple' else multiscale_NIF(cfg, logger=writer, device=cfg['device'], ckpt_save_path=save_path, visual=visual_1dwave) if cfg['model'] == 'nif_multiscale' else NotImplementedError('This model has not been implemented')
    cfg['training_cfg']['nb_params'] = count_params(model)
    print("model :", model)
    
    model.fit(dataloader_train, n_epochs=cfg['training_cfg']['nepoch'], lr=cfg['training_cfg']['lr_init'],
              scheduler = nif_scheduler, validation_data=dataloader_test, verbose=100,
              save_images_freq=cfg['training_cfg']['print_figure_epoch'], vistrain=dtrain[:], vistest=dtest[:])

    cfg['training_time'] = time.time() - tic

    print("cfg : ", cfg)
    print("save_path : ", save_path)
    dict2yaml(save_path + '/cfg.yaml', cfg)

    return True

if __name__ == "__main__":
    cfg_path = 'config/nif_1dwave.yaml'
    cfg_path = 'config/nifmultiscale_1dhfwave.yaml'
    print(main(cfg_path))
