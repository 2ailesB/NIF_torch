import os
import json
import torch
import logging
import datetime
import numpy as np
import time
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from models.nif_lastlayer import NIF_lastlayer
from models.nif_multiscale import NIF_multiscale
from models.simple_nif import simple_NIF
from utils.yaml import yaml2dict, dict2yaml

# TODO : data preprocessing
def main(path):

    cfg = yaml2dict(path)

    logging.basicConfig(level=logging.INFO)
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = f"logs/{cfg['model']}-{cfg['dataset']}-{start_time}"
    writer = SummaryWriter(save_path)

    dict2yaml(save_path + 'cfg.yaml', cfg)

    cfg['logger']            = writer
    cfg['ckpt_save_path']    = None

    path = os.path.abspath("../data") # TODO chage data
    dataset = dset.MNIST(root=path,
                         download=True,
                         transform=transforms.Compose([
                             transforms.Resize(cfg['img_size']),
                             transforms.CenterCrop(cfg['img_size']),
                             transforms.ToTensor(),
                             transforms.Normalize(0.5, 0.5),
                         ]))

    print(dataset)
    torch.manual_seed(cfg['seed'])
    cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', cfg['device'])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=1)

    tic = time.time()
    model = simple_NIF() if cfg['model']=='nif' else NIF_lastlayer() if cfg['model']=='nif_lastlayer' else NIF_multiscale() if cfg['model']=='nif_multiscale' else ValueError
    model.fit(dataloader, n_epochs=cfg['training_cfg']['nepoch'], lr=cfg['training_cfg']['lr_init'], save_images_freq=cfg['training_cfg']['freq_'])

    cfg['training_time'] = time.time() - tic

    dict2yaml(save_path + 'cfg.yaml', cfg)

    return True

if __name__ == "__main__":
    print(main())