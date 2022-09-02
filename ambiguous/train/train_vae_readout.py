import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py
from ambiguous.models.vae import MLPVAE
from ambiguous.models.readout import *
from ambiguous.dataset.dataset import partition_dataset
import argparse
from sklearn.manifold import TSNE

def plot_reconstruction(plot_path, model, dataloader, device='cuda'):
    images, labels = next(iter(dataloader))
    rec,_,_,_ = model(images.to(device))
    fig=plt.figure(figsize=[12,12])
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(images[i].cpu().permute(1,2,0))
        plt.axis('off')
    plt.figure(figsize=[12,12])
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(rec[i].detach().cpu().permute(1,2,0))
        plt.axis('off')
    fig.savefig(plot_path)
    plt.close(fig)

def latent_space_viz(plot_path, model, data_loader, N=1000, device='cuda'):
    batch = torch.cat([data_loader.dataset[x][0] for x in range(N)], 0)
    labels = torch.Tensor([data_loader.dataset[x][1] for x in range(N)])
    _, mu, _, _  = model(batch.view(-1,1,model.img_size,model.img_size).to(device))
    mu_2d = TSNE(2).fit_transform(mu.cpu().detach().numpy())
    fig=plt.figure()
    sns.set(rc={'figure.figsize':(10,8)})
    sns.scatterplot(x=mu_2d[:,0], y=mu_2d[:,1], hue=labels, palette='deep', legend='full')
    fig.savefig(plot_path)
    plt.close(fig)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--num_epochs_readout', type=int, default=20)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--readout_h_dim', type=int, default=512)
    parser.add_argument('--n_cls', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--vae_path', type=str, default='')
    parser.add_argument('--latent_plot_path', type=str, default='')
    parser.add_argument('--recon_plot_path', type=str, default='')
    parser.add_argument('--readout_path', type=str, default='')
    parser.add_argument('--train_vae', default=False, action='store_true')
    parser.add_argument('--plot_vae', default=False, action='store_true')
    parser.add_argument('--train_readout', default=False, action='store_true')

    args = parser.parse_args()
    data_path = args.data_path
    vae_path = args.vae_path
    latent_plot_path = args.latent_plot_path
    recon_plot_path = args.recon_plot_path
    readout_path = args.readout_path
    train_vae = args.train_vae
    plot_vae = args.plot_vae
    train_readout = args.train_readout
    batch_size = args.batch_size # 64
    img_size=args.img_size # 28
    lr=args.lr # 1e-3
    n_cls = args.n_cls
    num_epochs=args.num_epochs # 25
    num_epochs_readout=args.num_epochs_readout
    latent_dim = args.latent_dim # 10
    readout_h_dim = args.readout_h_dim
    device = args.device # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(device)
    root=data_path

    torch.cuda.empty_cache()
    torch.cuda.init()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.MNIST(root=root, download=True, train=True, transform=transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [round(0.8*len(dataset)), round(0.2*len(dataset))])
    test_set = datasets.MNIST(root=root, download=True, train=False, transform=transform)
    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    def loss_function(rec, x, mu, logvar, criterion):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        rec_error = criterion(rec, x)
        return KLD + rec_error

    if train_vae:
        model = MLPVAE(latent_dim = latent_dim, input_img_size=img_size).to(device)
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss(reduction='sum')
        n_train = len(train_loader)
        n_val = len(val_loader)
        for i in tqdm(range(num_epochs)):
            running_loss = 0
            for _, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                rec, mu, logvar, _ = model(images)
                loss = loss_function(rec, images, mu, logvar, criterion)
                running_loss += loss/batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            val_loss = 0
            for _, (images, labels) in enumerate(val_loader):
                with torch.no_grad():
                    images = images.to(device)
                    labels = labels.to(device)
                    rec, mu, logvar, _ = model(images)
                    loss = loss_function(rec, images, mu, logvar, criterion)
                    val_loss += loss/batch_size
            print(f"Epoch: {i+1} \t Train Loss: {running_loss/n_train:.2f} \t Val Loss: {val_loss/n_val:.2f}")

            torch.save(model.state_dict(), vae_path)
    else:
        model = MLPVAE(latent_dim = latent_dim, input_img_size=img_size).to(device)
        model.load_state_dict(torch.load(vae_path))
        model.eval()
        print(model)

    if plot_vae:
        latent_space_viz(latent_plot_path, model, val_loader, 2000)
        plot_reconstruction(recon_plot_path, model, val_loader)

    if train_readout:
        readout = Readout(latent_dim=latent_dim, h=readout_h_dim, n_classes=n_cls).to(device)
        print(readout)
        readout_opt = optim.Adam(readout.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(reduction='sum')
        for i in tqdm(range(num_epochs_readout)):
            train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0
            for _, (images, labels) in enumerate(train_loader):
                images=images.to(device)
                labels=labels.to(device)
                loss, pred = loss_readout_OG(readout, model, images, labels, criterion)
                readout_opt.zero_grad()
                loss.backward()
                readout_opt.step()
                train_loss += loss/batch_size
                train_acc += (torch.argmax(pred,1)==labels).float().sum()
                
            readout.eval()
            for _, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                loss, pred = loss_readout_OG(readout, model, images, labels, criterion)
                val_loss += loss/batch_size
                val_acc += (torch.argmax(pred,1)==labels).float().sum()
            
            val_acc = val_acc/(batch_size*len(val_loader))
            train_acc /= (batch_size*len(train_loader))
            print(f"Epoch:{i+1} \t Train Loss:{train_loss:.3f} \t Val Loss:{val_loss:.3f} \t Train Acc:{train_acc:.3f} \t Val Acc:{val_acc:.3f}")
            torch.save(readout.state_dict(), readout_path)
            readout.eval()

    else:
        readout = Readout(latent_dim=latent_dim, h=readout_h_dim, n_classes=n_cls).to(device)
        print(readout)
        readout.load_state_dict(torch.load(readout_path))
        criterion = nn.CrossEntropyLoss(reduction='sum')
        readout.eval()

if __name__ == '__main__':
    main()