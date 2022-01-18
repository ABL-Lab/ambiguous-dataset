import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import *
from torchvision.utils import save_image
from tqdm import tqdm
from datetime import datetime
import yaml
from project.data_utils import *
from project.models.ambiguous_generator import *
import project.models.cvae
from project.models.cvae import Encoder, Decoder, EMNIST_CVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AmbiguousDataset(Dataset):
    def __init__(self, generator, pure_pairs, transform=None, target_transform=None, 
                 n=60000, n_classes=10, blend=0.5):
        self.transform = transform
        self.target_transform = target_transform
        self.n = n
        self.generator = generator
        self.blend = blend
        self.n_classes = n_classes
        self.pure_pairs = pure_pairs
        self.ambiguity = np.array([blend]*len(pure_pairs)) #fix
        
    def __len__(self):
        return self.n

    def __getitem__(self, idx, p=2,C=3, H=32, W=32, n_classes=10):
        images,labels=torch.zeros((p,C,H,W)).to(device), torch.zeros((2,self.n_classes)).to(device)
        idx = sample_pair(len(self.pure_pairs))
        t, ambi = self.pure_pairs[idx], self.ambiguity[idx]
        t, ambi = torch.from_numpy(t).to(device).long(), torch.from_numpy(ambi).to(device).float()
        c1,c2 = t[:,0],t[:,1]
        images = load_batch(self.generator, t, c1, c2, self.transform, ambi, no_label=True)
        labels[list(range(p)), c1] = self.blend
        labels[list(range(p)), c2] = 1-self.blend
        if self.target_transform:
            labels = self.target_transform(labels)
        image,label = images[0],labels[0]
        return image, label
    
    def set_blend(self, blend):
        self.blend = blend

MNIST_pairs = np.array([(3,8),(8,3),(3,5),(5,3),(5,8),(8,5),(0,6),(6,0)])

def MNIST_fly(root, blend, pairs=MNIST_pairs):
    """
    root: data directory
    blend: ambiguity level (min 0, max 1)
    pairs: ambiguous class pairs, by default = MNIST_PAIRS
    """
    with open('project/save_dict.yaml','r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    n_classes = params['n_classes']
    img_path=params['img']
    encoder = torch.load(params['enc']).to(device)
    decoder = torch.load(params['dec']).to(device)
    encoder.eval(),decoder.eval()
    dataset = datasets.MNIST(root=root, download=True, train=True,
                             transform=transforms.Compose([transforms.ToTensor()]))
    generator = MNISTGenerator(encoder, decoder, DataLoader(dataset, batch_size=2, shuffle=True),
                               n_classes=n_classes, device=device)
    dataset = AmbiguousDataset(generator, pairs, n_classes=n_classes, blend=blend)
    return dataset



idx = lambda x: 'abcdefghijklmnopqrstuvwxyz'.index(x)
EMNIST_PAIRS = np.array([(idx('c'), idx('o')), (idx('c'), idx('e'))])

def EMNIST_fly(root, blend, pairs=EMNIST_PAIRS):
    """
    root: data directory
    blend: ambiguity level (min 0, max 1)
    pairs: ambiguous class pairs, by default = EMNIST_PAIRS
    """
    with open('project/emnist_params.yaml','r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    n_classes = 26
    latent_dim = 4
    enc_layers = [28*28, 512]
    dec_layers = [512, 28*28]
    ckpt_path = params['ckpt_path']
    img_path = params['img_path']
    model = EMNIST_CVAE(latent_dim, enc_layers, dec_layers, n_classes=26, conditional=True).to(device)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    encoder,decoder = model.encoder,model.decoder
    dataset = datasets.EMNIST(root=root, download=True, train=True, 
                              split='letters', transform=transforms.Compose([transforms.ToTensor()]))
    generator = EMNISTGenerator(encoder, decoder, DataLoader(dataset, batch_size=2, shuffle=True),
                                n_classes=n_classes, device=device)
    dataset = AmbiguousDataset(generator, pairs, blend=blend, n_classes=n_classes)  
    return dataset

  
    
def test_dataloader(data_loader, img_path):
    x,t = next(iter(data_loader))
    save_image(x,img_path,nrow=8)
    return
