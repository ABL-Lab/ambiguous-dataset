import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from datetime import datetime
import yaml
import h5py
import copy
from itertools import chain
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
device='cuda'

### MNIST Conditional VAE ###
class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden=500, latent_dim=4, n_categories=10):
        super(Encoder, self).__init__()
        self.l1 = nn.Linear(input_dim+n_categories, hidden)
        self.l2 = nn.Linear(hidden, latent_dim)

    def forward(self, x, c):
        x = torch.cat([x, c], 1)
        o = torch.tanh(self.l1(x))
        encoding = self.l2(o)
        return encoding, o

    def sample(self, variational_params, device):
        q_mu, q_logsigma = variational_params[:, :2], variational_params[:, 2:]
        q_mu, q_logsigma = q_mu.view(-1, 2), q_logsigma.view(-1, 2)
        z = torch.randn_like(q_mu, device=device)*torch.exp(q_logsigma) + q_mu 
        return z, q_mu, q_logsigma


class Decoder(nn.Module):
    def __init__(self, output_dim=784, hidden=500, z_dim=2, n_categories=10):
        super(Decoder, self).__init__()
        self.l1 = nn.Linear(z_dim+n_categories, hidden)
        self.l2 = nn.Linear(hidden, output_dim)

    def forward(self, x1, z, c):
        z = torch.cat([z, c], 1)
        o = torch.tanh(self.l1(z))
        decoded = self.l2(o)
        return decoded


# ## EMNIST CVAE ###

class EMNIST_CVAE(pl.LightningModule):
    def __init__(self, latent_dim, enc_layer_sizes, dec_layer_sizes, n_classes=26, conditional=False):
        super().__init__()
        self.encoder = EMNIST_Encoder(latent_dim, enc_layer_sizes, n_classes, conditional)
        self.decoder = EMNIST_Decoder(latent_dim, dec_layer_sizes, n_classes, conditional)
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        
    def loss(self, x, rec, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        rec_error = F.binary_cross_entropy(rec, x, reduction='sum')
        return (KLD + rec_error) / x.size(0) 
    
    def training_step(self, batch, batch_idx):
        x, t = batch
        _,c,h,w = x.shape
        x, t = x.to(device), t.to(device) - 1
        c = torch.zeros(x.size(0),self.n_classes).to(device)
        c[range(x.size(0)), t] = 1
        mu, logvar = self.encoder(x.view(-1, 784), c)
        z = self.encoder.sample(mu, logvar)
        rec = self.decoder(z, c).view(-1, 1, 28, 28)
        loss = self.loss(x, rec, mu, logvar)
        batch_dictionary={'loss':loss, 'mu': mu, 'targets': t}
        return batch_dictionary      
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, t = batch
            _,c,h,w = x.shape
            x, t = x.to(device), t.to(device) - 1
            c = torch.zeros(x.size(0),self.n_classes).to(device)
            c[range(x.size(0)), t] = 1
            mu, logvar = self.encoder(x.view(-1, 784), c)
            z = self.encoder.sample(mu, logvar)
            rec = self.decoder(z, c).view(-1, 1, 28, 28)
            loss = self.loss(x, rec, mu, logvar)
        batch_dictionary={'loss':loss, 'mu': mu, 'targets': t}
        return batch_dictionary
    
    def embedding_figure_adder(self, outputs):
        mu = torch.cat([x['mu'] for x in outputs[-100:]], dim=0)
        targets = torch.cat([x['targets'] for x in outputs[-100:]], dim=0).cpu().detach().numpy()
        embed = TSNE(2).fit_transform(mu.cpu().detach().numpy())
        fig = plt.figure()
        sns.set(rc={'figure.figsize':(6,6)})
        sns.scatterplot(x=embed[:,0], y=embed[:,1], hue=targets, palette='deep', legend='full')
        self.logger.experiment.add_figure('Viz/Embedding', fig, self.current_epoch)
        
    def generate_imgs(self, outputs):
        targets = outputs[-2]['targets']
        n = targets.size(0)
        c = torch.zeros(n,self.n_classes).to(device)
        c[range(n), targets] = 1
        z = torch.randn(n, self.latent_dim).to(device)
        rec = self.decoder(z, c).view(-1, 1, 28, 28)
        grid = make_grid(rec, nrow=8)
        self.logger.experiment.add_image('Viz/Reconstruction', grid, self.current_epoch)
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)
        epoch_dictionary = {'loss': avg_loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('Loss/Val', avg_loss, self.current_epoch)
        epoch_dictionary = {'loss': avg_loss}
        self.embedding_figure_adder(outputs)
        self.generate_imgs(outputs)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': self.encoder.parameters(), 'lr': 1e-3}, 
                                  {'params':self.decoder.parameters(), 'lr': 1e-3}])
        return optimizer

class EMNIST_Encoder(nn.Module):
    def __init__(self, latent_dim, layer_sizes, n_classes, conditional=False):
        super(EMNIST_Encoder, self).__init__()
        self.MLP = nn.Sequential()
        
        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += n_classes
            
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
        self.fc_mu = nn.Linear(layer_sizes[-1], latent_dim)
        self.fc_logvar = nn.Linear(layer_sizes[-1], latent_dim)
        
        self.device = device
        
    def forward(self, x, c=None):
        if self.conditional:
            x = torch.cat([x, c], 1)       
        h = self.MLP(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def sample(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.device == 'cuda':
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

class EMNIST_Decoder(nn.Module):
    def __init__(self, latent_dim, layer_sizes, n_classes, conditional=False):
        super(EMNIST_Decoder, self).__init__()
        
        self.MLP = nn.Sequential()
        self.conditional = conditional
        if self.conditional:
            input_size = latent_dim + n_classes
        else:
            input_size = latent_dim

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())
    def forward(self, z, c):
        if self.conditional:
            z = torch.cat([z, c], 1)
        rec = self.MLP(z)
        return rec   
