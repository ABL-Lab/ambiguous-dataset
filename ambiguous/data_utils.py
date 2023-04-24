import torch
from torchvision import transforms, datasets
import numpy as np
from copy import deepcopy
from tqdm import tqdm
device='cuda'

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

def map_target_ambiguity(model, generator, pairs, n_sample, transform, target=0.5):
    """ Find the mapping between generator blend and classifier ambiguity (target)"""
    mapping = torch.zeros(len(pairs)).to(device)-1
    for i,(a,b) in enumerate(pairs):
        for blend in np.arange(0.1,1,0.1):
            _, img, _, _ = generator.generate_ambiguous(n_sample, c1=a,c2=b, blend=blend)
            img=transform(img.to(device))
            pred = torch.argmax(model(img), 1)
            p_a, p_b = (pred==a).float().mean(), (pred==b).float().mean()
            if abs(p_a - target) < 0.1 and abs (p_b - target) < 0.1:
                mapping[i] = blend
    return mapping