from torch import nn
import torch
import torch.nn.functional as F

device='cuda'
class Readout(nn.Module):
    def __init__(self, latent_dim=10, h=64, n_classes=10):
        super(Readout, self).__init__()
        self.readout = nn.Sequential(nn.Linear(latent_dim, h),
                                     nn.ReLU(),
                                     nn.Linear(h, h),
                                     nn.ReLU(),
                                     nn.Linear(h, n_classes)
                                    )
        
    def forward(self, x):
        return self.readout(x)
    

def loss_readout(readout, vae, images, targets, criterion=nn.CrossEntropyLoss()):
    images, targets = images.to(device), targets.to(device)
    with torch.no_grad():
        _, mu, _ = vae(images)
    pred = readout(mu)
    loss = criterion(pred, targets)
    return loss, pred

def loss_readout_OG(readout, vae, images, targets, criterion=nn.CrossEntropyLoss()):
    images, targets = images.to(device), targets.to(device)
    with torch.no_grad():
        _, mu, _, _ = vae(images)
    pred = readout(mu)
    loss = criterion(pred, targets)
    return loss, pred


def loss_readout_conv(readout, vae, images, targets, criterion=nn.CrossEntropyLoss()):
    images, targets = images.to(device), targets.to(device)
    with torch.no_grad():
        _, _, mu, _ = vae(images)
    pred = readout(mu)
    loss = criterion(pred, targets)
    return loss, pred