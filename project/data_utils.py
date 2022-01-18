import torch
from torch.utils.data import *
from torchvision import transforms, datasets
import numpy as np
from copy import deepcopy
from tqdm import tqdm
device='cuda'

class AmbiguousDataset(Dataset):
    def __init__(self, generator, pure_pairs, transform=None, target_transform=None, n=60000, n_classes=10, blend=0.5):
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

def sample_pair(n):
    idx = list(np.random.choice(n,2))
    return idx

def load_batch(generator, t, c1, c2, transform=None, amb=0.5, N=200, bs=2, test=False, no_label=False):
    _,x,_,_ = generator.generate_ambiguous(bs,c1=c1,c2=c2,blend=amb)
    if transform:
        x = transform(x)
    if no_label:
        return x
    y = torch.zeros(bs, 10).to(device)
    y_eprime = deepcopy(y).to(device)
    idx = list(range(bs))
    y[idx, t] = 1
    eprime = sample_expectation(c1,c2, bs)
    if test:
        eprime = [c1 if targ==c2 else c2 for targ in t]
    y_eprime[idx, eprime] = 1
    return x, y, y_eprime

def map_true_ambiguity(vae, readout, generator, pairs, n_sample, transform, true_amb=0.5):
    """ Find the mapping between generator ambiguity and readout ambiguity"""
    mapping = torch.zeros(len(pairs)).to(device)-1
    for i,(a,b) in enumerate(pairs):
        for r in np.arange(0.1,1,0.1):
            _, img, _, _ = generator.generate_ambiguous(n_sample, r, r, a,b)
            img=transform(img.to(device))
            _, mu, _ = vae(img)
            pred = torch.argmax(readout(mu), 1)
            p_a, p_b = torch.mean((pred==a).float()), torch.mean((pred==b).float())
            if abs(p_a - true_amb) < 0.1 and abs (p_b - true_amb) < 0.1: # if abs(p_a - p_b) < 0.2 and p_a > 0.35:
                mapping[i] = r
        print((a,b),mapping[i])
    return mapping

def pure_pair(a,b, pure_pairs):
    return (a,b) in pure_pairs

def retain_pure_pairs(t):
    c1, c2 = t[list(range(0,n,2))], t[list(range(1,n,2))]
    pure_idxs = []
    pair_idxs = []
    for i in range(len(c1)):
        if pure_pair(c1[i],c2[i], pure_pairs):
            pure_idxs.append(i)
            pair_idxs.append(pure_pairs.index((c1[i],c2[i])))
    c1,c2 = c1[pure_idxs], c2[pure_idxs]
    return c1,c2, pair_idxs

def sample_expectation(a,b, bs=2):
    L = [x for x in range(10) if x != a and x != b]
    idx = torch.randint(0, len(L), (1,))
    return L[idx]

def generate_targets(t, a_weight=None):
    if a_weight is None:
        return t
    t_w = deepcopy(t).to(device)
    p = torch.FloatTensor(t.size(0)).uniform_(0,1).to(device)
    t_w[p > a_weight] = t[1] # bs = 2
    t_w[p <= a_weight] = t[0]
    return t_w


# def save_dataset(vae, readout, encoder, decoder, save_file, N=60000, C=3, H=32, W=32, n_classes=10, bs=2, n_sample=2000):
#     """ images: tensor of size (N, C, H, W)"""
#     tform = transforms.Compose([transforms.ToTensor()])
#     tf = transforms.Compose([transforms.Resize((32,32))]) 
#     dataset = datasets.MNIST(root='/share/datasets', download=True, train=False, transform=tform)
#     temp_generator = MNISTGenerator(encoder, decoder, DataLoader(dataset, batch_size=n_sample, shuffle=True), device=device)
#     #ambiguity = map_true_ambiguity(vae, readout, temp_generator, pure_pairs, n_sample, tf)
#     generator = MNISTGenerator(encoder, decoder, DataLoader(dataset, batch_size=bs, shuffle=True), device=device)
#     images,labels=torch.zeros((N,C,H,W)).to(device), torch.zeros((N,n_classes)).to(device)

#     for i in tqdm(range(0,N,bs)):
#         idx = sample_pair(len(pure_pairs))
#         t, ambi = pure_pairs[idx], ambiguity[idx]
#         t,ambi = torch.from_numpy(t).to(device).long(), torch.from_numpy(ambi).to(device).float()
#         c1,c2 = t[:,0],t[:,1]
#         x = load_batch(generator, t, c1, c2, tf, ambi, no_label=True)
#         images[i:i+bs] = x
#         labels[list(range(i,i+bs)), c1] = 0.5
#         labels[list(range(i,i+bs)), c2] = 0.5
#     torch.save(images, save_file+"_images.pt")
#     torch.save(labels, save_file+"_labels.pt")
#     return images, labels
