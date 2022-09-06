import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.mlp import MLP, MLP_parametrized, MLP4SIREN
from models.layers.siren import SIREN, SIRENs, SIREN_parametrized

class NIF(nn.Module):
    """
        Main class that defines the general NIF class and hypernetworks
        Must be implemented through a class
    """
    def __init__(self, cfg_parameter_net, cfg_shape_net):
        super(NIF, self).__init__()
        self.cfg_shape_net = cfg_shape_net
        self.cfg_parameter_net = cfg_parameter_net
        self.input_shape = cfg_parameter_net['input_dim'] + cfg_shape_net['input_dim']

        # compute the number of weights to be computed by parameter net
        nweights = self.cfg_shape_net['input_dim'] * self.cfg_shape_net['units'] + \
                self.cfg_shape_net['units'] * self.cfg_shape_net['units'] * self.cfg_shape_net['nlayers'] + \
                self.cfg_shape_net['units'] * self.cfg_shape_net['output_dim']
        nbias = self.cfg_shape_net['units'] + \
                self.cfg_shape_net['units'] * self.cfg_shape_net['nlayers'] + \
                self.cfg_shape_net['output_dim']
        # in_features, out_features, layers=None, activation='gelu'
        self.cfg_hnet = dict()
        self.cfg_hnet['dim_out'] = nweights + nbias
        self.cfg_hnet['dim_in'] = self.cfg_parameter_net['latent_dim'] 
        self.hnet = MLP(self.cfg_hnet['dim_in'], self.cfg_hnet['dim_out']) #TODO
        
        if self.cfg_parameter_net['type'] =='mlp':
            self.parameter_net = MLP(self.cfg_parameter_net['input_dim'], self.cfg_parameter_net['latent_dim'], self.cfg_parameter_net['layers'], self.cfg_parameter_net['activation'])
        elif self.cfg_parameter_net['type'] == 'siren' :
            self.parameter_net = SIRENs(self.cfg_parameter_net['input_dim'], self.cfg_parameter_net['latent_dim'], self.cfg_parameter_net['layers'], omega_0=self.cfg_parameter_net['omega_0'])
        else :
            raise NotImplementedError('NIF not implemented for this kind of layer, please use mlp or siren')
        if self.cfg_shape_net['type'] =='mlp':
            self.shape_net = MLP_parametrized(self.cfg_shape_net['input_dim'], self.cfg_shape_net['output_dim'], self.cfg_shape_net['layers'], self.cfg_shape_net['activation'])
        elif self.cfg_shape_net['type']=='siren':
            self.hnet = MLP4SIREN(self.cfg_hnet['dim_in'], self.cfg_hnet['dim_out'], self.cfg_shape_net) #TODO
            self.shape_net = SIREN_parametrized(self.cfg_shape_net['input_dim'], self.cfg_shape_net['output_dim'], self.cfg_shape_net['layers'], omega_0=self.cfg_shape_net['omega_0'])
        else :
            raise NotImplementedError('NIF not implemented for this kind of layer, please use mlp or siren')

        

    def forward(self, x):
        x_parameter = x[:, :self.cfg_parameter_net['input_dim']]
        x_shape = x[:, -self.cfg_shape_net['input_dim']:] # [bxatch-size, input dim]
        y1 = self.parameter_net(x_parameter) # [bxatch-size, input dim]
        # print("y1.shape :", y1.shape)
        sweights = self.hnet(y1) # [bxatch-size, latent dim]
        # print("sweights.shape :", sweights.shape)
        yhat = self.shape_net(x_shape, sweights) # [bxatch-size, hnet out]
        # print("yhat.shape :", yhat.shape)

        return yhat

class NIF_DO(nn.Module):
    """
        Main class that defines the general NIF class and hypernetworks
        Must be implemented through a class
    """
    def __init__(self, cfg_parameter_net, cfg_shape_net):
        super(NIF_DO, self).__init__()
        self.cfg_shape_net = cfg_shape_net
        self.cfg_parameter_net = cfg_parameter_net
        self.input_shape = cfg_parameter_net['input_dim'] + cfg_shape_net['input_dim']

        # in_features, out_features, layers=None, activation='gelu'
        self.cfg_hnet = dict()
        self.cfg_hnet['dim_out'] = self.cfg_parameter_net['latent_dim'] 
        self.cfg_hnet['dim_in'] = self.cfg_parameter_net['latent_dim'] 
        self.hnet = MLP(self.cfg_hnet['dim_in'], self.cfg_hnet['dim_out']) #TODO
        
        if self.cfg_parameter_net['type'] =='mlp':
            self.parameter_net = MLP(self.cfg_parameter_net['input_dim'], self.cfg_parameter_net['latent_dim'], self.cfg_parameter_net['layers'], self.cfg_parameter_net['activation'])
        elif self.cfg_parameter_net['type'] == 'siren' :
            self.parameter_net = SIREN(self.cfg_parameter_net['input_dim'], self.cfg_parameter_net['latent_dim'], self.cfg_parameter_net['layers'], omega_0=self.cfg_parameter_net['omega_0'])
        else :
            raise NotImplementedError('NIF not implemented for this kind of layer, please use mlp or siren')
        if self.cfg_shape_net['type'] =='mlp':
            self.shape_net = MLP(self.cfg_shape_net['input_dim'], self.cfg_parameter_net['latent_dim']*self.cfg_shape_net['output_dim'], self.cfg_shape_net['layers'], self.cfg_shape_net['activation'])
        elif self.cfg_shape_net['type'] == 'siren' :
            self.hnet = MLP4SIREN(self.cfg_hnet['dim_in'], self.cfg_hnet['dim_out'], self.cfg_shape_net) #TODO
            self.shape_net = SIRENs(self.cfg_shape_net['input_dim'], self.cfg_parameter_net['latent_dim']*self.cfg_shape_net['output_dim'], self.cfg_shape_net['layers'], omega_0=self.cfg_shape_net['omega_0'])
        else :
            raise NotImplementedError('NIF not implemented for this kind of layer, please use mlp or siren')

        self.last_layer_bias = nn.Parameter(torch.zeros((self.cfg_shape_net['output_dim']))) # TODO check
        torch.nn.init.trunc_normal_(self.last_layer_bias, std=0.1)

        

    def forward(self, x):
        x_parameter = x[:, :self.cfg_parameter_net['input_dim']]
        x_shape = x[:, -self.cfg_shape_net['input_dim']:] # [bxatch-size, input dim]
        y1 = self.parameter_net(x_parameter) # [bxatch-size, latent dim]
        # print("y1.shape :", y1.shape)
        llweights = self.hnet(y1) # [bxatch-size, latent dim]
        # print("llweights.shape :", llweights.shape)
        llval = self.shape_net(x_shape)
        # print("llval.shape :", llval.shape)
        llval = llval.reshape((llval.shape[0], self.cfg_shape_net['output_dim'], self.cfg_parameter_net['latent_dim'])) # [bxatch-size, outputdim, latent dim]
        # print("llval.shape :", llval.shape)
        # print("llweights.shape :", llweights.shape)
        yhat = torch.einsum('bij,bj->bi', llval, llweights)
        # print("yhat.shape :", yhat.shape)
        yhat += self.last_layer_bias
        # print("yhat.shape :", yhat.shape)

        return yhat

