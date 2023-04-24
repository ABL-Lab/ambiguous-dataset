from torch import nn
import torch
import torch.nn.functional as F


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


class MLPVAE(nn.Module):
    """
    VAE with an MLP backbone. This architecture is supported in this class but not used for the paper.
    """
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