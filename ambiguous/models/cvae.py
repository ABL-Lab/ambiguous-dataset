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
import torch.nn.functional as F
device='cuda'
    
class CatConv(nn.Module):
    def __init__(self, conv1, conv2):
        super(CatConv, self).__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        
    def forward(self, xy):
        x, y = xy
        out1 = self.conv1(x)
        out2 = self.conv2(y)
        return torch.cat([out1, out2], 1)

class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)        

    
class Conv_CVAE(nn.Module):
    def __init__(self,
                 latent_dim,
                 hidden_dims=None,
                 in_channels=1,
                 n_cls=10,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        h_dim = hidden_dims[0]
        conv0_2 = nn.Sequential(
                                nn.Conv2d(n_cls, out_channels=h_dim//2, 
                                          kernel_size=3, stride=2, padding=1), 
                                nn.BatchNorm2d(h_dim//2), 
                                nn.LeakyReLU()
                            )
        conv0_1 = nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim//2, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim//2), nn.LeakyReLU())
        conv0 = CatConv(conv0_1, conv0_2)
        modules.append(conv0)
        in_channels = h_dim
        
        for h_dim in hidden_dims[1:]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        
        self.decoder_input = nn.Linear(latent_dim, latent_dim) #hidden_dims[-1])

        hidden_dims.reverse()
        # Build Decoder
        modules = []

        deconv0_1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dims[1]//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[1]//2),
            nn.LeakyReLU()
        )
        deconv0_2 = nn.Sequential(
            nn.ConvTranspose2d(n_cls, hidden_dims[1]//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[1]//2),
            nn.LeakyReLU()
        )
        deconv0 = CatConv(deconv0_1, deconv0_2)
        modules.append(deconv0)
        for i in range(1,len(hidden_dims) - 1):
            if i==2:
                output_padding=0
            else:
                output_padding=1
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=output_padding),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
        

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, zy):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        z,y = zy
        z = self.decoder_input(z)
        z = z.view(-1, self.latent_dim, 1, 1) # 
        result = self.decoder((z,y))
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        x, y_enc, y_dec = input
        mu, log_var = self.encode((x,y_enc))
        z = self.reparameterize(mu, log_var)
        return  [self.decode((z,y_dec)), mu, log_var]

    def loss_function(self,
                      *args):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        rec = args[0]
        x = args[1]
        mu = args[2]
        logvar = args[3]
        
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        rec_loss = F.mse_loss(rec, x, reduction='sum')
        loss = rec_loss + kld_loss
        return {'loss': loss, 'rec':rec_loss.detach(), 'kld':kld_loss.detach()}

    def sample(self,
               num_samples,
              **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim).to(device)
        with torch.no_grad():
            samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class ConvolutionalVAE(nn.Module):
    def __init__(self, latent_dim=20, n_cls=10, img_size=28, n_convs=None, in_ch=[1,32,64,128], conditional=False,
                 kernel_size=3, stride=2, padding=1, h_dim=2048, relu=False, req_flatten_size=None, last_layer='linear'):
        super(ConvolutionalVAE, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.n_cls = n_cls
        self.conditional = conditional
        self.in_ch = in_ch
        self.n_convs = len(in_ch)-1 if in_ch is not None else n_convs
        self.kernel_size = [kernel_size]*self.n_convs
        self.stride = [stride]*self.n_convs
        self.padding = [padding]*self.n_convs
        self.num_layers = self.n_convs
        self.last_layer = last_layer
        self.h_dim = h_dim
        self.activation = nn.ReLU() if relu else nn.LeakyReLU()
        self.calculate_final_dim()
        if req_flatten_size is not None:
            assert self.flatten_size == req_flatten_size, "Flatten size is not equal to required flatten size"

        modules = []

        for i in range(self.num_layers):
            if i == 0 and self.conditional:
                conv0_1 = nn.Sequential(nn.Conv2d(in_ch[i], in_ch[i+1]//2, self.kernel_size[i], stride=self.stride[i], padding=self.padding[i]), nn.BatchNorm2d(in_ch[i+1]//2), self.activation)
                conv0_2 = nn.Sequential(nn.Conv2d(n_cls, in_ch[i+1]//2, self.kernel_size[i], stride=self.stride[i], padding=self.padding[i]), nn.BatchNorm2d(in_ch[i+1]//2), self.activation)
                conv = CatConv(conv0_1, conv0_2)
            else:
                conv = nn.Sequential(nn.Conv2d(in_ch[i], in_ch[i+1], self.kernel_size[i], stride=self.stride[i], padding=self.padding[i]), nn.BatchNorm2d(in_ch[i+1]), self.activation)
            modules.append(conv)

        modules.append(
                        nn.Sequential(
                            nn.Conv2d(in_ch[-1], self.h_dim, kernel_size=self.flatten_size, stride=self.flatten_size, padding=0),
                            nn.BatchNorm2d(self.h_dim),
                            self.activation
                        )
                    )
        
        if self.last_layer == 'linear':
            modules.append(nn.Flatten())
        
        self.encoder = nn.Sequential(*modules)

        # readin modulates at this output
        
        self.fc_mu = nn.Linear(self.h_dim, latent_dim) if self.last_layer=='linear' else nn.Sequential(nn.Conv2d(self.h_dim, latent_dim, kernel_size=1), nn.Flatten())
        self.fc_logvar = nn.Linear(self.h_dim, latent_dim) if self.last_layer=='linear' else nn.Sequential(nn.Conv2d(self.h_dim, latent_dim, kernel_size=1), nn.Flatten())

        self.fc_mu2 = nn.Sequential(
                                    nn.Linear(latent_dim, in_ch[-1]*self.flatten_size**2), 
                                    View(shape=(-1, in_ch[-1], self.flatten_size, self.flatten_size)), 
                                    self.activation
                                )
        modules = []
        for i in range(self.num_layers):
            last_pad = img_size//(2**i) % 2
            scale_factor = img_size//(2**i) / (img_size//(2**(i+1)) + last_pad)
            act_fn = nn.Identity() if i == self.num_layers - 1 else self.activation
            if i == 0 and self.conditional:
                upconv0 = nn.Sequential( 
                                        nn.Upsample(scale_factor=scale_factor, mode='bilinear') , 
                                        nn.Conv2d(in_ch[-1], in_ch[-2-i]//2, kernel_size=(1, 1), stride=1, padding=0), 
                                        nn.BatchNorm2d(in_ch[-2-i]//2), 
                                        act_fn
                                    )
                upconv1 = nn.Sequential( 
                                        nn.Upsample(scale_factor=scale_factor, mode='bilinear') , 
                                        nn.Conv2d(n_cls, in_ch[-2-i]//2, kernel_size=(1, 1), stride=1, padding=0), 
                                        nn.BatchNorm2d(in_ch[-2-i]//2), 
                                        act_fn
                                    )
                upconv = CatConv(upconv0, upconv1)
            else:
                upconv = nn.Sequential( 
                                        nn.Upsample(scale_factor=scale_factor, mode='bilinear') , 
                                        nn.Conv2d(in_ch[-1-i], in_ch[-2-i], kernel_size=(1, 1), stride=1, padding=0), 
                                        nn.BatchNorm2d(in_ch[-2-i]), 
                                        act_fn
                                    )
            modules.append(upconv)
        self.decoder = nn.Sequential(*modules)

    @staticmethod
    def output_size(img_size, kernel_size, stride, padding):
        return (img_size - kernel_size + 2*padding)//stride + 1

    def calculate_final_dim(self):
        flatten_size = self.img_size
        for i in range(self.num_layers):
            flatten_size = self.output_size(flatten_size, self.kernel_size[i], self.stride[i], self.padding[i])
        self.flatten_size = flatten_size

    def reparamaterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, input, out_h=False):
        if self.conditional:
            x, y_enc, y_dec = input
            h = self.encoder((x, y_enc))
        else:
            x = input
            h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparamaterize(mu, logvar)
        z = self.fc_mu2(z)
        if self.conditional:
            rec = self.decoder((z, y_dec))
        else:
            rec = self.decoder(z)
        if out_h:
            return rec, x, mu, logvar, h
        return rec, x, mu, logvar

    def loss_function(self,rec,x,mu,logvar):
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        rec_loss = F.mse_loss(rec, x, reduction='sum')
        loss = rec_loss + kld_loss
        return {'loss': loss, 'rec':rec_loss.detach(), 'kld':kld_loss.detach()}
