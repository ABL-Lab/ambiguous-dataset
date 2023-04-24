import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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

class AmbiguousGenerator:      

    def init_prior(self, n, num, c1, c2):
        prior = torch.zeros(n, self.n_classes, device=self.device)
        idxs = range(n)
        prior[idxs, c1] = num
        prior[idxs, c2] = 1-num
        return prior
    
    def generate_ambiguous(self):
        """Generate ambiguous stimuli between class 1 and class 2.
        Ambiguity=0 looks like class 1, while 1 looks like class 2
        Ambiguity: tensor with shape (N, 1). N = num samples. N <= batch size"""
        raise notImplementedError

    def plot_images(self, imgs, h=28):
        img = imgs.view(-1,h)
        fig=plt.figure(figsize=(imgs.size(0)*2,imgs.size(0)*2))
        plt.imshow(img.cpu().detach().numpy(), cmap='gray')
        plt.axis('off')
        return   
    
    def eval_cls_interpolation(self, net, c1, c2, interpolation, n_eval=256, label_offset=0):
        pct_c1 = torch.zeros_like(interpolation)
        pct_c2 = torch.zeros_like(pct_c1)
        pct_other = torch.zeros_like(pct_c2)
        for i, blend in enumerate(interpolation):
            _, rec, c1, c2 = self.generate_ambiguous(n_eval, blend, c1, c2)
            pred = net(rec)
            pct_c1[i] = torch.sum(pred==c1+label_offset)/len(pred)
            pct_c2[i] = torch.sum(pred==c2+label_offset)/len(pred)
            pct_other[i] = 1-pct_c1[i]-pct_c2[i]
        return pct_c1, pct_c2, pct_other


class MNISTGenerator(AmbiguousGenerator):
    def __init__(self, encoder, decoder, dataloader, device, n_classes=10):
        super(AmbiguousGenerator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.n_classes = n_classes
        self.data_loader = dataloader        
        
    def generate_ambiguous(self, N, blend=0.5, c1=None, c2=None, h=28):
        x, t = next(iter(self.data_loader)) 
        x = x.view(-1, h*h).to(self.device)
        x = x[:N]
        c1 = torch.zeros((N,), dtype=torch.long).to(self.device) + c1
        c2 = torch.zeros((N,), dtype=torch.long).to(self.device) + c2
        prior = self.init_prior(N, blend, c1, c2)
        q_params, x1 = self.encoder(x, prior)
        z, q_mu, q_ls = self.encoder.sample(q_params, device=self.device)
        rec = transform_logit(self.decoder(x1, z, prior)).view(-1,1,h,h)
        return x, rec, c1, c2

def transform_logit(y):
    return torch.exp(y)/(1+torch.exp(y))     

class EMNISTGenerator(AmbiguousGenerator):
    def __init__(self, cvae, n_latent, n_classes=26):
        super(EMNISTGenerator, self).__init__()
        self.cvae = cvae
        self.device = device
        self.n_classes = n_classes
        self.n_latent = n_latent
        
    def generate_ambiguous(self, n_samples, blend, c1, c2, h=28):
        c = torch.zeros(n_samples, self.n_classes, dtype=torch.long).to(device)
        a = torch.zeros(n_samples, dtype=torch.long).to(device) + c1
        b = torch.zeros(n_samples, dtype=torch.long).to(device) + c2
        c[range(n_samples), a.long()] = float(blend)
        c[range(n_samples), b.long()] = float(1-blend)
        z = torch.randn(n_samples, self.n_latent).to(device)
        rec = self.cvae.decoder(z, c).view(-1, 1, 28, 28)
        return 0,rec,c1,c2
    
    

def psychometric_curves(gen, net, n_classes=10, label_offset=0):
    interpolation = torch.arange(0,1.01,0.05)
    pairs = [[(x,y) for y in range(x+1, n_classes)] for x in range(n_classes)]
    pairs = list(chain(*pairs))
    fig,ax = plt.subplots(len(pairs), figsize=(10,80))
    plt.tight_layout()
    [axi.xaxis.set_visible(False) for axi in ax.ravel()]
    p_c1 = torch.zeros((len(pairs), interpolation.size(0)))
    p_c2, p_other = torch.zeros_like(p_c1), torch.zeros_like(p_c1)
    for i,(a,b) in enumerate(pairs):
        p_c1[i], p_c2[i], p_other[i] = gen.eval_cls_interpolation(net, a,b, interpolation, label_offset=label_offset)
        ax[i].plot(p_c1[i]); ax[i].plot(p_c2[i]); ax[i].plot(p_other[i])
        ax[i].set_title(f"{a}-{b} interpolation")
        ax[0].legend(['class 1', 'class 2', 'other'])
    fig.savefig("results/psychometric.pdf")
    
    fig2 = plt.figure(figsize=(20,5))
    mass_other = p_other.mean(dim=1)
    xy = sorted(list(zip(mass_other, pairs)))
    y,x = [p[0] for p in xy], [f"{p[1][0]}-{p[1][1]}" for p in xy]
    plt.bar(x,y)
    plt.title("Probability of other class")
    plt.axhline(np.mean(y), color='r')
    fig2.savefig("results/ambiguous_pairs.pdf")
    return
