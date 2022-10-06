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
        x, t = x.to(device), t.to(device)
        c = torch.zeros(x.size(0),self.n_classes).to(device)
        c[range(x.size(0)), t] = 1
        mu, logvar = self.encoder(x.view(-1, 784), c)
        z = self.encoder.sample(mu, logvar)
        rec = self.decoder(z, c).view(-1, 1, 28, 28)
        loss = self.loss(x, rec, mu, logvar)
        batch_dictionary={'loss':loss, 'mu': mu, 'targets': t}
        self.logger.log_metrics({'Loss/TrainStream': loss.item()}, step=self.global_step)
        return batch_dictionary      
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, t = batch
            _,c,h,w = x.shape
            x, t = x.to(device), t.to(device)
            c = torch.zeros(x.size(0),self.n_classes).to(device)
            c[range(x.size(0)), t] = 1
            mu, logvar = self.encoder(x.view(-1, 784), c)
            z = self.encoder.sample(mu, logvar)
            rec = self.decoder(z, c).view(-1, 1, 28, 28)
            loss = self.loss(x, rec, mu, logvar)
        batch_dictionary={'loss':loss, 'mu': mu, 'targets': t}
        self.logger.log_metrics({'Loss/ValStream': loss.item()}, step=self.global_step)
        return batch_dictionary
    
    def embedding_figure_adder(self, outputs):
        mu = torch.cat([x['mu'] for x in outputs[-100:]], dim=0)
        targets = torch.cat([x['targets'] for x in outputs[-100:]], dim=0).cpu().detach().numpy()
        embed = TSNE(2).fit_transform(mu.cpu().detach().numpy())
        fig = plt.figure()
        sns.set(rc={'figure.figsize':(12,12)})
        sns.scatterplot(x=embed[:,0], y=embed[:,1], hue=targets, palette='deep', legend='full')
        fig.savefig("embedding.png")
        plt.close(fig)
#         self.logger.experiment.add_figure('Viz/Embedding', fig, self.current_epoch)
        self.logger.log_image(key=f'Viz/Embedding', images=['embedding.png'])
        
    def generate_imgs(self, outputs):
        targets = outputs[-2]['targets']
        n = targets.size(0)
        c = torch.zeros(n,self.n_classes).to(device)
        c[range(n), targets] = 1
        z = torch.randn(n, self.latent_dim).to(device)
        rec = self.decoder(z, c).view(-1, 1, 28, 28)
        torchvision.utils.save_image(rec, "reconstruction.png", nrow=8)
        #self.logger.experiment.add_image('Viz/Reconstruction', grid, self.current_epoch)
        self.logger.log_image(key=f'Viz/Reconstruction', images=["reconstruction.png"])
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        #self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)
        self.logger.log_metrics({'Loss/Train': avg_loss.item()}, step=self.current_epoch)
        epoch_dictionary = {'loss': avg_loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
#         self.logger.experiment.add_scalar('Loss/Val', avg_loss, self.current_epoch)
        self.logger.log_metrics({'Loss/Val': avg_loss.item()}, step=self.current_epoch)
        epoch_dictionary = {'loss': avg_loss}
        self.embedding_figure_adder(outputs)
        self.generate_imgs(outputs)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': self.encoder.parameters(), 'lr': 1e-3}, 
                                  {'params':self.decoder.parameters(), 'lr': 1e-3}])
        return optimizer
    
    
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
    def __init__(self, latent_dim=20, n_cls=10, img_size=28, in_ch=[1,32,64,128],conditional=False,
                 kernel_size=3, stride=2, padding=1):
        super(ConvolutionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.n_cls = n_cls
        self.conditional = conditional
        self.in_ch = in_ch
        kernel_size = [kernel_size]*len(in_ch)
        stride = [stride]*len(in_ch)
        padding = [padding]*len(in_ch)
        num_layers = len(in_ch)-1
        modules = []
        for i in range(num_layers):
            if i == 0 and self.conditional:
                conv0_1 = nn.Sequential(nn.Conv2d(in_ch[i], in_ch[i+1]//2, kernel_size[i], stride=stride[i], padding=padding[i]), nn.BatchNorm2d(in_ch[i+1]//2), nn.LeakyReLU())
                conv0_2 = nn.Sequential(nn.Conv2d(n_cls, in_ch[i+1]//2, kernel_size[i], stride=stride[i], padding=padding[i]), nn.BatchNorm2d(in_ch[i+1]//2), nn.LeakyReLU())
                conv = CatConv(conv0_1, conv0_2)
            else:
                conv = nn.Sequential(nn.Conv2d(in_ch[i], in_ch[i+1], kernel_size[i], stride=stride[i], padding=padding[i]), nn.BatchNorm2d(in_ch[i+1]), nn.LeakyReLU())
            modules.append(conv)
        modules.append(nn.Flatten())
        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(128*4*4, latent_dim)
        self.fc_logvar = nn.Linear(128*4*4, latent_dim)

        self.fc_mu2 = nn.Sequential(
                                    nn.Linear(latent_dim, in_ch[-1]*4*4), 
                                    View(shape=(-1, in_ch[-1], 4, 4)), 
                                    nn.LeakyReLU()
                                )
        modules = []
        for i in range(num_layers):
            last_pad = img_size//(2**i) % 2
            scale_factor = img_size//(2**i) / (img_size//(2**(i+1)) + last_pad)
            act_fn = nn.Sigmoid() if i == num_layers - 1 else nn.LeakyReLU()
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

    def reparamaterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, input):
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
        return rec, x, mu, logvar

    def loss_function(self,rec,x,mu,logvar):
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        rec_loss = F.mse_loss(rec, x, reduction='sum')
        loss = rec_loss + kld_loss
        return {'loss': loss, 'rec':rec_loss.detach(), 'kld':kld_loss.detach()}

####################################
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

    
class EMNIST_EncoderV1(nn.Module):
    def __init__(self, latent_dim, layer_sizes):
        super(EMNIST_EncoderV1, self).__init__()
        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
        self.fc_mu = nn.Linear(layer_sizes[-1], latent_dim)
        self.fc_logvar = nn.Linear(layer_sizes[-1], latent_dim)
        
        self.device = device
        
    def forward(self, x):     
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
    
    