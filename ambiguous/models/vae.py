from torch import nn
import torch
import torch.nn.functional as F
    
class MLPVAE(nn.Module):
    def __init__(self, input_img_size=32, hidden1_size=256, hidden2_size=128, latent_dim=2):
        super(MLPVAE, self).__init__()
        self.img_size = input_img_size
        self.encoder = nn.Sequential(
            nn.Linear(input_img_size**2,hidden1_size),
            nn.BatchNorm1d(hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size,hidden2_size),
            nn.BatchNorm1d(hidden2_size),
            nn.ReLU(),
            )
        # apical basal combination
        self.fc1_mu = nn.Linear(hidden2_size, latent_dim) # -> mu
        self.fc1_logvar = nn.Linear(hidden2_size, latent_dim) # -> sigma
        self.fc2 = nn.Linear(latent_dim, hidden2_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden2_size, hidden1_size),
            nn.BatchNorm1d(hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size, input_img_size**2)
        )
                
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
    
    def forward(self, x):
        x = self.encoder(x.view(-1, self.img_size**2))
        mu = self.fc1_mu(x)
        logvar = self.fc1_logvar(x)
        z = self.reparameterize(mu, logvar)
        rec = self.decoder(self.fc2(z))
        rec = rec.view(-1, 1, self.img_size, self.img_size)
        return rec, mu, logvar, x
    
    def combine_topdown(self, x, readin, latent, expectation, intermediate=False, apical_mask=None, alpha=0, interaction='dendritic'):
        bottomup = self.encoder(x.view(-1, self.img_size**2))
        topdown = readin(latent, expectation)
        if apical_mask is not None:
            topdown = topdown*apical_mask.float()
        if interaction == 'dendritic':
            combined = topdown_bottomup_dendritic(bottomup, topdown)
        elif interaction == 'dendritic-warmup':
            combined = topdown_bottomup_dendritic_warmup(bottomup, topdown, alpha=alpha)
        elif interaction == 'add':
            combined = topdown_bottomup_add(bottomup, topdown)
        elif interaction == 'mult':
            combined = topdown_bottomup_mult(bottomup, topdown)
        else:
            print("Not supported interaction.")
            return
        mu = self.fc1_mu(combined) # AB @ W_mu. vs B @ W_mu. combined should correlate to unambiguous basal
        if intermediate:
            return (mu, topdown, bottomup)
        return mu


def topdown_bottomup_dendritic(r_bottomup, r_topdown):
    return r_bottomup * (1 + F.relu(r_topdown))

def topdown_bottomup_dendritic_warmup(r_bottomup, r_topdown, alpha=0):
    return r_bottomup * (1 + F.relu(r_topdown)) + alpha*r_topdown

def topdown_bottomup_dendritic_clamp(r_bottomup, r_topdown):
    return r_bottomup * (1 + torch.clamp(F.relu(r_topdown), max=1))

def topdown_bottomup_add(r_bottomup, r_topdown):
    return r_bottomup + r_topdown

def topdown_bottomup_mult(r_bottomup, r_topdown):
    return r_bottomup * r_topdown

def topdown_bottomup_add_relu(r_bottomup, r_topdown):
    return r_bottomup + F.relu(r_topdown)

def topdown_bottomup_mult_relu(r_bottomup, r_topdown):
    return r_bottomup * F.relu(r_topdown)