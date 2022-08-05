import datetime
import os

import torch
import torch.nn as nn

from torchvision.utils import make_grid
from utils.progress_bar import print_progress_bar

class PytorchNIF(nn.Module):
    def __init__(self, criterion='mse', logger=None, opt='adam', device='cpu', ckpt_save_path=None, tag='', visual=None):
        super().__init__()
        self.opt_type = opt if opt in ['sgd', 'adam'] else ValueError
        self.opt = None
        self.log = logger
        self.device = device
        self.ckpt_save_path = ckpt_save_path
        self.state = {}
        self.criterion = nn.MSELoss(reduction='mean') if criterion == 'mse' else nn.BCELoss(reduction='mean') if criterion == 'bce' else criterion
        self.visual_func = visual

        self.best_criterion = {'train_loss': 10**10, 'val_loss': 10**10, 'mse':10*10}
        self.best_model = None
        self.best_epoch = 0

        # /!\ the overriding class must implement a discriminator and a generator extending nn.Module
        self.input_shape = None
        self.model = None
        
        # useful stuff that can be needed for during fit
        self.start_time = None
        self.verbose    = None
        self.n_epochs   = None
        self.n          = None
        self.tag        = tag
        self.scheduler  = None

        self.save_images_freq = None

    def _train_epoch(self, dataloader):

        epoch_loss  = 0

        for idx, batch in enumerate(dataloader):

            batch_x, batch_y = batch
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_size = batch_x.size(0)

            # Compute the loss the for the discriminator with real images
            self.model.zero_grad()
            yhat = self.model(batch_x) # .view(-1)
            loss = self.criterion(yhat, batch_y)
            loss.backward()
            self.opt.step()
            
            if self.scheduler is not None:
                self.opt.param_groups[0]['lr'] = self.scheduler(self.n, self.lr)

            # Update running losses
            epoch_loss += loss.item()
            if self.verbose == 1:
                print_progress_bar(idx, len(dataloader))

        return epoch_loss / len(dataloader)

    def _validate(self, dataloader):
        epoch_loss = 0

        for idx, batch in enumerate(dataloader):
            batch_x, batch_y = batch
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_size = batch_x.size(0)

            # Compute the loss the for the discriminator with real images
            self.model.zero_grad()
            yhat = self.model(batch_x) # .view(-1)
            loss = self.criterion(yhat, batch_y)

            # Update running losses
            epoch_loss += loss.item()
            if self.verbose == 1:
                print_progress_bar(idx, len(dataloader))

        return loss / len(dataloader)

    def fit(self, dataloader, n_epochs, lr, scheduler=None, validation_data=None, verbose=1, save_images_freq=None, save_criterion='mse', ckpt=None, save_model_freq=None, betas=(0.0, 0.99),  vistrain=None, vistest=None, **kwargs):
        assert self.model is not None, 'Model does not seem to have a model, assign the model to the self.model attribute'
        assert self.input_shape is not None, 'Could not find the input shape, please specify this attribute before fitting the model'

        if self.opt_type == 'sgd':
            self.opt = torch.optim.SGD(params=self.model.parameters(), lr=lr)
        elif self.opt_type == 'adam':
            self.opt = torch.optim.Adam(params=self.model.parameters(), lr=lr, betas=betas)
        else:
            raise ValueError('Unknown optimizer')

        start_time = datetime.datetime.now()
        self.start_time = start_time

        start_epoch = 0
        self.verbose = verbose
        self.scheduler = scheduler
        self.lr = lr

        if ckpt:
            state = torch.load(ckpt)
            start_epoch = state['epoch']
            self.load_state_dict(state['state_dict'])
            for g in self.opt.param_groups:
                g['lr'] = state['lr']['lr']

        self.n_epochs = n_epochs
        for n in range(start_epoch, n_epochs):
            self.n = n
            self.train()
            t_loss = self._train_epoch(dataloader)
            v_loss = 0
            if validation_data is not None:
                self.eval()
                with torch.no_grad():
                    v_loss = self._validate(validation_data)

            epoch_result = {'train_loss': t_loss, 'val_loss': v_loss, save_criterion:t_loss}
            if self.log:
                for k, v in epoch_result.items():
                    self.log.add_scalar(k, v, n)

            if epoch_result[save_criterion] <= self.best_criterion[save_criterion]:
                self.best_criterion = epoch_result
                self.__save_state(n)

            if n % verbose == 0:
                print('Epoch {:3d} training loss: {:1.4f} | Validation loss: {:1.4f} | Best epoch {:3d}'.format(
                    n, t_loss, v_loss, self.best_epoch))
            with torch.no_grad():
                if save_images_freq is not None and n % save_images_freq == 0 and self.visual_func is not None: # TODO visual logs
                    self.visual_func(self.model, vistrain[0].to(self.device), vistrain[1].to(self.device), self.ckpt_save_path, 'train')
                    self.visual_func(self.model, vistest[0].to(self.device), vistest[1].to(self.device), self.ckpt_save_path, 'test')

            if save_model_freq is not None and n % save_model_freq == 0 :
                assert self.ckpt_save_path is not None, 'Need a path to save models'
                self.save({'lr': lr}, n)

        print(f'Training completed in {str(datetime.datetime.now() - start_time).split(".")[0]}')

    def save(self, lr, n):
        self.state['lr'] = lr
        self.state['epoch'] = n
        self.state['state_dict'] = self.state_dict()
        if not os.path.exists(self.ckpt_save_path):
            os.mkdir(self.ckpt_save_path)
        torch.save(self.state, os.path.join(self.ckpt_save_path, f'ckpt_{self.tag}{self.start_time}_epoch{n}.ckpt'))

    def load(self, ckpt_path):
        state = torch.load(ckpt_path)
        self.load_state_dict(state['state_dict'])

    def __save_state(self, n):
        self.best_epoch = n
        self.best_model = self.state_dict()

    def __load_saved_state(self):
        if self.best_model is None:
            raise ValueError('No saved model available')
        self.load_state_dict(self.best_model)