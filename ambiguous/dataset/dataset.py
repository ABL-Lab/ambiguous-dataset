import os
import importlib
import matplotlib.pyplot as plt
import torchvision
from torch import nn, optim
from datetime import datetime
from ambiguous.data_utils import *
from ambiguous.models.ambiguous_generator import *
import ambiguous.models.cvae
from ambiguous.models.cvae import Encoder, Decoder, EMNIST_CVAE
from torchvision.utils import save_image
import torch
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import yaml
import glob
import subprocess
import urllib.request
from torch.utils.data import *
from torch.utils.data.dataset import Dataset  # For custom datasets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MNIST_PAIRS = np.array([(3,8),(8,3),(3,5),(5,3),(5,8),(8,5),(0,6),(6,0),(4,9),(9,4),(6,8),(8,6),(5,6),(6,5),(1,7),(7,1)])
idx = lambda x: 'abcdefghijklmnopqrstuvwxyz'.index(x)
EMNIST_PAIRS = np.array([(idx('c'), idx('o')), (idx('c'), idx('e')),
                        (idx('i'), idx('l')), (idx('k'), idx('x')),
                        (idx('n'), idx('m')), (idx('v'), idx('w')),
                        (idx('i'), idx('j')), (idx('u'), idx('v')),
                        (idx('h'), idx('n')), (idx('h'), idx('b'))])


class DatasetFromNPY(Dataset):
    def __init__(self, root, download=False, train=True, transform=None):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        # if download:
        #     if not os.path.isfile(root+'/aMNIST.zip'):
        #         print("Downloading dataset...")
        #         _ = subprocess.run(['sh', 'download_amnist.sh'])
        #     else:
        #         print("Using downloaded dataset...")
        if train:
            self.image_list = glob.glob(root+'/train/*')
        else:
            self.image_list = glob.glob(root+'/test/*')
        # Calculate len
        self.data_len = len(self.image_list)
        self.transform = transform

    def __getitem__(self, index):
        single_image_path = self.image_list[index]
        # Open image
        im_as_np = np.load(single_image_path).astype(np.float64)/255.
        im_as_ten = torch.from_numpy(im_as_np)
        if self.transform:
            im_as_ten = self.transform(im_as_ten)
        # Get label(class) of the image based on the file name
        start_idx = single_image_path.rfind('_c') + 2
        end_idx = single_image_path.rfind('.npy')
        split_idx = single_image_path[start_idx:end_idx].rfind('_')
        label1 = int(single_image_path[start_idx:start_idx+split_idx])
        label2 = int(single_image_path[start_idx+split_idx+1:end_idx])
        label = [label1, label2]
        return (im_as_ten, label)

    def __len__(self):
        return self.data_len


def save_dataset_to_file(dataset_name, og_root, new_root, blend, batch_size=100, n_train=60000, n_test=10000):
    if not os.path.isdir(new_root):
        os.makedirs(new_root+'/train/')
        os.makedirs(new_root+'/test/')
    if dataset_name == 'MNIST':
        dataset_train = aMNIST_fly(og_root, blend=blend, train=True)
        dataset_test = aMNIST_fly(og_root, blend=blend, train=False)
        pairs = MNIST_PAIRS
    elif dataset_name == 'EMNIST':
        dataset_train = aEMNIST_fly(og_root, blend=blend, train=True)
        dataset_test = aEMNIST_fly(og_root, blend=blend, train=False)
        pairs = EMNIST_PAIRS
    trainLoader = DataLoader(dataset_train, batch_size=100, num_workers=0, shuffle=True)
    testLoader = DataLoader(dataset_test, batch_size=100, num_workers=0)
    for i in tqdm(range(n_train//batch_size)):
        x, t = next(iter(trainLoader))
        x_, t_ = next(iter(testLoader))
        x, t = (255*x).cpu().detach().numpy().astype(np.uint8), t.cpu().detach().numpy()
        x_, t_ = (255*x_).cpu().detach().numpy().astype(np.uint8), t_.cpu().detach().numpy()
        for j in range(batch_size):
            idx = i*batch_size+j
            cl = np.where(t[j] == 0.5)[0]
            np.save(new_root+f'/train/tr_{idx}_c{cl[0]}_{cl[1]}.npy', x[j])
            if i < n_test//batch_size:
                cl = np.where(t_[j] == 0.5)[0]
                np.save(new_root+f'/test/test_{idx}_c{cl[0]}_{cl[1]}.npy', x_[j])
    return


class AmbiguousDatasetFly(Dataset):
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
        images,labels = torch.zeros((p,C,H,W)).to(device), torch.zeros((2,self.n_classes)).to(device)
        idx = sample_pair(len(self.pure_pairs))
        t, ambi = self.pure_pairs[idx], self.ambiguity[idx]
        t, ambi = torch.from_numpy(t).to(device).long(), torch.from_numpy(ambi).to(device).float()
        c1,c2 = t[:,0],t[:,1]
        images = load_batch(self.generator, t, c1, c2, self.transform, ambi, no_label=True)
        labels[list(range(p)), c1] = self.blend
        labels[list(range(p)), c2] = 1-self.blend
        if self.target_transform:
            labels = self.target_transform(labels)
        image, label = images[0],labels[0]
        return image, label
    
    def set_blend(self, blend):
        self.blend = blend


def aMNIST_fly(root, blend, pairs=MNIST_PAIRS, train=True):
    """
    root: data directory
    blend: ambiguity level (min 0, max 1)
    pairs: ambiguous class pairs, by default = MNIST_PAIRS
    """
    with open('/home/nislah/ambiguous-dataset/ambiguous/save_dict.yaml','r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    n_classes = params['n_classes']
    img_path = params['img']
    encoder = torch.load(params['enc']).to(device)
    decoder = torch.load(params['dec']).to(device)
    encoder.eval(), decoder.eval()
    dataset = datasets.MNIST(root=root, download=True, train=train,
                             transform=transforms.Compose([transforms.ToTensor()]))
    generator = MNISTGenerator(encoder, decoder, DataLoader(dataset, batch_size=2, shuffle=True),
                               n_classes=n_classes, device=device)
    dataset = AmbiguousDatasetFly(generator, pairs, n_classes=n_classes, blend=blend)
    return dataset


def aEMNIST_fly(root, blend, pairs=EMNIST_PAIRS, train=True):
    """
    root: data directory
    blend: ambiguity level (min 0, max 1)
    pairs: ambiguous class pairs, by default = EMNIST_PAIRS
    """
    with open('/home/nislah/ambiguous-dataset/ambiguous/emnist_params.yaml','r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    n_classes = 26
    latent_dim = 4
    enc_layers = [28*28, 512]
    dec_layers = [512, 28*28]
    ckpt_path = params['ckpt_path']
    model = EMNIST_CVAE(latent_dim, enc_layers, dec_layers, n_classes=26, conditional=True).to(device)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    encoder, decoder = model.encoder, model.decoder
    dataset = datasets.EMNIST(root=root, download=True, train=train, 
                              split='letters', transform=transforms.Compose([transforms.ToTensor()]))
    generator = EMNISTGenerator(encoder, decoder, DataLoader(dataset, batch_size=2, shuffle=True),
                                n_classes=n_classes, device=device)
    dataset = AmbiguousDatasetFly(generator, pairs, blend=blend, n_classes=n_classes)  
    return dataset


def save_examples(data_loader, output_file):
    x, t = next(iter(data_loader))
    save_image(x, output_file, nrow=8)
    return