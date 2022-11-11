import os
import importlib
import matplotlib.pyplot as plt
import torchvision
from torch import nn, optim
from datetime import datetime
from ambiguous.data_utils import *
# from ambiguous.adversarial import *
from ambiguous.models.ambiguous_generator import *
import ambiguous.models.cvae
from ambiguous.models.cvae import *
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
        if split_idx == -1: # If only one class (unambiguous)
            label1 = int(single_image_path[start_idx:end_idx])
            label = [label1]
        else: # If two classes (ambiguous)
            label1 = int(single_image_path[start_idx:start_idx+split_idx])
            label2 = int(single_image_path[start_idx+split_idx+1:end_idx])
            label = [label1, label2]
        return (im_as_ten, label)

    def __len__(self):
        return self.data_len

class DatasetTriplet(Dataset):
    def __init__(self, root, download=False, split='train', transform=None):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        Args:
            folder_path (string): path to image folder
        """
        self.image_list = sorted(glob.glob(root+f'/{split}/*image.npy'))
        self.label_list = sorted(glob.glob(root+f'/{split}/*label.npy'))
        # Calculate len
        self.data_len = len(self.image_list)
        self.transform = transform

    def __getitem__(self, index, img_size=28):
        single_image_path = self.image_list[index]
        im_as_np = np.load(single_image_path).astype(np.float64)/255.
        im_as_ten = torch.from_numpy(im_as_np)
        clean1 = im_as_ten[:, :, :img_size]
        amb = im_as_ten[:, :, img_size:2*img_size]
        clean2 = im_as_ten[:, :, 2*img_size:3*img_size] 
        label = torch.from_numpy(np.load(self.label_list[index]))
        return (clean1, amb, clean2), label

    def iterlabels(self, numpy=True):
        for index in range(len(self)):
            if numpy:
                label = np.load(self.label_list[index])
            else:
                label = torch.from_numpy(np.load(self.label_list[index]))
            yield label

    def __len__(self):
        return self.data_len


def partition_datasetV2(dataset, n_cls):
    newdataset = [[] for _ in range(n_cls)]
    for (im, label) in dataset:
        newdataset[label[0]].append((im, label))
    return newdataset


class SequenceDataset(Dataset):
    def __init__(self, root, download=False, train=True, transform=None, n_cls=10, ambiguous=False, 
                 cache=False,cache_dir=None, include_irrelevant=False):
        """
        A dataset where the input is a sequence of images that should add up to a target image
        """
        self.triplet_dataset = DatasetTriplet(root, download, train, transform)
        self.partitioned_datasets = partition_datasetV2(self.triplet_dataset, n_cls)
        self.ambiguous = ambiguous
        self.cache = cache
        self.cache_dir = cache_dir
        self.split = 'train' if train else 'test'
        self.data_len = sum([len(dataset) for dataset in self.partitioned_datasets])
        self.include_irrelevant = include_irrelevant

    def __getitem__(self, index, img_size=28):
        """
        return the sequence as a tuple and the target as a tensor
        """
        if self.cache:
            (clean1, _, clean2), label = self.triplet_dataset[index]
            target = (label[0] + label[1]) % 10
            cleansum1, sum_label = self.sample(target)
            if self.include_irrelevant and self.ambiguous:
                ambsum1, pair_label = self.sample(target, ambiguous=True) # 2nd label in the pair
                ambsum2, _ = self.sample(pair_label, ambiguous=True)
                cleansum2, _ = self.sample(pair_label)
                clean3, _ = self.sample((pair_label-label[0]) % 10) # 0/6 .. 2 0-2=-2 % 10 = 8
                img_seq = torch.stack([clean1, clean2, cleansum1, ambsum1, clean3, ambsum2, cleansum2]) # clean1 + clean2 = cleansum1 (ambsum1). clean1 + clean3 = cleansum2 (ambsum2). irrelevant = clean2 + clean3
            else:
                img_seq = torch.stack([clean1, clean2, cleansum1])
            torch.save(img_seq, f'{self.cache_dir}/{self.split}/img_seq_{index}.pt')
            torch.save(sum_label, f'{self.cache_dir}/{self.split}/sum_label_{index}.pt')
        else:
            img_seq = torch.load(f'{self.cache_dir}/{self.split}/img_seq_{index}.pt')
            sum_label = torch.load(f'{self.cache_dir}/{self.split}/sum_label_{index}.pt')
        return img_seq, sum_label

    def __len__(self):
        return self.data_len

    def sample(self, target, ambiguous=False):
        """
        sample n images from the dataset
        """
        dataset = self.partitioned_datasets[target]
        idx = torch.randint(0, len(dataset), (1,))
        label = torch.where(dataset[idx][1] == target)[0]
        other = torch.where(dataset[idx][1] != target)[0]
        if ambiguous:
            label = 1
        else:
            label *= 2 # 0 or 1 -
        return dataset[idx][0][label], dataset[idx][1][other]

def save_dataset_to_file(dataset_name, og_root, new_root, blend, pairs=None, batch_size=100, n_train=60000, n_test=10000):
    os.makedirs(new_root+'/train/')
    os.makedirs(new_root+'/test/')
    if dataset_name == 'MNIST':
        if pairs is None:
            pairs = MNIST_PAIRS
        dataset_train = aMNIST_fly(og_root, pairs=pairs, blend=blend, train=True)
        dataset_test = aMNIST_fly(og_root, pairs=pairs, blend=blend, train=False)
    elif dataset_name == 'EMNIST':
        if pairs is None:
            pairs = EMNIST_PAIRS
        dataset_train = aEMNIST_fly(og_root, pairs=pairs, blend=blend, train=True)
        dataset_test = aEMNIST_fly(og_root, pairs=pairs, blend=blend, train=False)
    trainLoader = DataLoader(dataset_train, batch_size=100, num_workers=0, shuffle=True)
    testLoader = DataLoader(dataset_test, batch_size=100, num_workers=0)
    for i in tqdm(range(n_train//batch_size)):
        x, t = next(iter(trainLoader))
        x_, t_ = next(iter(testLoader))
        x, t = (255*x).cpu().detach().numpy().astype(np.uint8), t.cpu().detach().numpy()
        x_, t_ = (255*x_).cpu().detach().numpy().astype(np.uint8), t_.cpu().detach().numpy()
        for j in range(batch_size):
            idx = i*batch_size+j
            cl = np.where(t[j] > 0)[0]
            if len(cl)>1:
                np.save(new_root+f'/train/tr_{idx}_c{cl[0]}_{cl[1]}.npy', x[j])
            elif len(cl)==1:
                np.save(new_root+f'/train/tr_{idx}_c{cl[0]}.npy', x[j])
            if i < n_test//batch_size:
                cl = np.where(t_[j] > 0)[0]
                if len(cl)>1:
                    np.save(new_root+f'/test/test_{idx}_c{cl[0]}_{cl[1]}.npy', x_[j])
                elif len(cl)==1:
                    np.save(new_root+f'/test/test_{idx}_c{cl[0]}.npy', x_[j])
    return


def to_numpy_uint8(x):
    """x: tensor"""
    return (255*x).cpu().detach().numpy().astype(np.uint8)


def to_numpy(t):
    return t.cpu().detach().numpy()


# def save_EMNIST_adversarial(vae, readout, dataloader, dataset_name, og_root, new_root, batch_size, h=28, train=True):
#     """
#     vae: emnist cvae
#     readout: emnist readout
#     dataloader: emnist dataloader
#     dataset_name: name of newdataset
#     og_root: emnist location
#     new_root: newdataset location
#     """
#     train = 'train' if train else 'test'
#     os.makedirs(new_root+f'/{train}', exist_ok=True)
    
#     for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader)):
#         inputs, targets = inputs.cuda(), targets.cuda()
#         even = torch.arange(0,len(targets),2)
#         odd = torch.arange(1,len(targets),2)
#         t_even = torch.index_select(t, 0, even).to(device)
#         t_odd = torch.index_select(t, 0, odd).to(device) 
#         attacked_inputs_even, attacked_targets_even = generate_adversarial(vae, readout, inputs[even], t_even, t_odd)
#         attacked_inputs_odd, attacked_targets_odd = generate_adversarial(vae, readout, inputs[odd], t_odd, t_even)
#         t_even, t_odd = to_numpy(t_even), to_numpy(t_odd)
#         attacked_targets_even, attacked_targets_odd = to_numpy(attacked_targets_even), to_numpy(attacked_targets_odd)
        
# #       if you want the best for sure
# #         _, attacked_mu, _ = vae(attacked_inputs)
# #         attack_out = readout(attacked_mu)
# #         pred = torch.argmax(attack_out, 1)
# #         topk=F.softmax(attack_out).topk(2)[0]
# #         best_adv = topk[:,1]>0.3
# #         best_adv_inputs = attacked_inputs[best_adv]
# #         best_adv_targets = attacked_targets[best_adv]
# #         num_adv = len(best_adv)
#         print(attacked_inputs_even.size(0),attacked_inputs_odd.size(0))
#         combined_inputs = torch.zeros(1, h*2, h*2).to(device)
#         for i in range(batch_size//2):
#             idx = batch_idx*batch_size+i
#             combined_inputs[:, 0:h, 0:h] = inputs[even][i]
#             combined_inputs[:, 0:h, h:2*h] = inputs[odd][i]
#             combined_inputs[:, h:2*h, 0:h] = attacked_inputs_even[i]
#             combined_inputs[:, h:2*h, h:2*h] = attacked_inputs_odd[i]
#             combined_inputs = to_numpy_uint8(combined_inputs)
#             t = np.array([ [t_even[i], t_odd[i]], [attacked_targets_even[i], attacked_targets_odd[i]] ])
#             t_flat = t.flatten()
#             img_loc = new_root+f'/{train}/img_{idx}.npy'
#             tgt_loc = new_root+f'/{train}/tgt_{idx}.npy'
#             np.save(img_loc, combined_inputs)
#             np.save(tgt_loc, t)
            
#     return



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
    with open('../save_dict.yaml','r') as file:
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
    with open('../emnist_params.yaml','r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    n_classes = 26
    latent_dim = 4
    enc_layers = [28*28, 1024, 1024]
    dec_layers = [1024, 1024, 28*28]
    ckpt_path = params['ckpt_path']
    model = EMNIST_CVAE(latent_dim, enc_layers, dec_layers, n_classes=26, conditional=True).to(device)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    encoder, decoder = model.encoder, model.decoder
    dataset = datasets.EMNIST(root=root, download=True, train=train, 
                              split='byclass', transform=transforms.Compose([transforms.ToTensor()]))
    new_dataset = partition_dataset(dataset, range(10, 36))
    generator = EMNISTGenerator(encoder, decoder, DataLoader(new_dataset, batch_size=2, shuffle=True),
                                n_classes=n_classes, device=device)
    dataset = AmbiguousDatasetFly(generator, pairs, blend=blend, n_classes=n_classes)  
    return dataset


def partition_dataset(dataset, t):
    newdataset = copy.copy(dataset)
    newdataset.data = [im for im, label in zip(newdataset.data, newdataset.targets) if label in t]
    newdataset.targets = [label - min(t) for label in newdataset.targets if label in t]
    return newdataset


def save_examples(data_loader, output_file):
    x, t = next(iter(data_loader))
    save_image(x, output_file, nrow=8)
    return
