import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import time
import torchvision
import seaborn as sns
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch
from tqdm import tqdm
from datetime import datetime
import yaml
import h5py
from copy import deepcopy
from project.models.cvae import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Seed for reproducibility
torch.manual_seed(42)
print(device)
# Create the working directory from timestamp and model name
model_name = 'cvae_emnist' # invertible network
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
workingDir = f'results/{model_name}_{timestamp}'
os.mkdir(workingDir)
configDict = {
'workingDir': workingDir,
'timestamp' : timestamp,
'model': model_name, # Model
'lr': 1e-3, # Learning rate
'batch_size': 64,
'n_epochs': 50,
"latent_dim": 4 # latent dimension for sampling
}

with open(f'{workingDir}/config.yaml', 'w') as file:
    documents = yaml.dump(configDict, file)
with open(f'{workingDir}/config.yaml','r') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
# Import parameters and construct model and optimizers
lr=params['lr']
batch_size=params['batch_size']
n_epochs=params['n_epochs']
latent_dim=params['latent_dim'] # latent dimension for sampling
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.EMNIST(root='/share/datasets', download=False, split='letters', train=True, transform=transform)
train_set, val_set = torch.utils.data.random_split(dataset, [100000, 24800])
test_set = datasets.EMNIST(root='/share/datasets', download=False, split='letters', train=False, transform=transform)

# Dataloaders
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=12, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=12, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=12, shuffle=False)
tb_logger = TensorBoardLogger('tb_logs', name=model_name)
enc_layers = [28*28, 512]
dec_layers = [512, 28*28]


model = EMNIST_CVAE(latent_dim, enc_layers, dec_layers, n_classes=26, conditional=True).to(device)
print(model)
trainer = pl.Trainer(
                     gpus=1,
                     logger=tb_logger,
                     max_epochs=n_epochs,
#                      callbacks=[
#                          pl.callbacks.early_stopping.EarlyStopping(
#                              monitor='Val/Loss',
#                              mode='min',
#                              patience=5
#                          )
#                      ],
                    auto_lr_find=True)
trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
