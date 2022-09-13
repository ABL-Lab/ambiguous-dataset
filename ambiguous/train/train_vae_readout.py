import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import h5py
from ambiguous.models.vae import MLPVAE
from ambiguous.models.cvae import Conv_CVAE
from ambiguous.models.readout import *
from ambiguous.dataset.dataset import partition_dataset
import argparse
from sklearn.manifold import TSNE
import mlflow
from mlflow import log_metrics, log_metric, log_artifact, log_artifacts, log_params

def plot_reconstruction(plot_path, model, dataloader, device='cuda'):
    images, labels = next(iter(dataloader))
    rec,_,_,_ = model(images.to(device))
    fig=plt.figure(figsize=[12,12])
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(images[i].cpu().detach().permute(1,2,0).numpy())
        plt.axis('off')
    plt.figure(figsize=[12,12])
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(rec[i].detach().cpu().detach().permute(1,2,0).numpy())
        plt.axis('off')
    fig.savefig(plot_path)
    plt.close(fig)

def latent_space_viz(plot_path, model, dataloader, N=1000, device='cuda'):
    batch = torch.cat([dataloader.dataset[x][0] for x in range(N)], 0)
    labels = torch.Tensor([dataloader.dataset[x][1] for x in range(N)])
    _, mu, _, _  = model(batch.view(-1,1,model.img_size,model.img_size).to(device))
    mu_2d = TSNE(2).fit_transform(mu.cpu().detach().numpy())
    fig=plt.figure()
    sns.set(rc={'figure.figsize':(10,8)})
    sns.scatterplot(x=mu_2d[:,0], y=mu_2d[:,1], hue=labels, palette='deep', legend='full')
    fig.savefig(plot_path)
    plt.close(fig)


class RecDataset(Dataset):
    def __init__(self, ccvae, n, n_cls=10, latent_dim=10, transform=None, device='cuda'):
        self.ccvae = ccvae
        self.n = n
        self.n_cls=n_cls
        self.latent_dim=latent_dim
        self.transform = transform
        self.device=device
        self.onehot = torch.zeros(n_cls, n_cls).scatter_(1, torch.LongTensor(range(n_cls)).view(n_cls,1), 1).view(n_cls, n_cls, 1, 1)

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        z = torch.randn(1, self.latent_dim).to(self.device)
        t = (torch.rand(1, 1) * self.n_cls).type(torch.LongTensor).squeeze(1)
        y = self.onehot[t].to(self.device)
        x = self.ccvae.decode((z, y))
        if self.transform is not None:
            x = self.transform(x)
        return x.squeeze(0), t.squeeze(0)


    def next_iter(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        t = (torch.rand(batch_size, 1) * self.n_cls).type(torch.LongTensor).squeeze(1)
        y = self.onehot[t].to(self.device)
        x = self.ccvae.decode((z, y))
        if self.transform is not None:
            x = self.transform(x)
        return x.squeeze(0), t.squeeze(0)

    

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
    parser.add_argument('--train_size', type=int, default=60000)
    parser.add_argument('--test_size', type=int, default=10000)
    parser.add_argument('--ccvae_path', type=str, default='')
    parser.add_argument('--vae_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--latent_plot_path', type=str, default='')
    parser.add_argument('--recon_plot_path', type=str, default='')
    parser.add_argument('--readout_path', type=str, default='')
    parser.add_argument('--train_vae', default=False, action='store_true')
    parser.add_argument('--plot_vae', default=False, action='store_true')
    parser.add_argument('--train_readout', default=False, action='store_true')

    args = parser.parse_args()
    ccvae_path = args.ccvae_path
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
    train_size = args.train_size
    test_size = args.test_size
    readout_h_dim = args.readout_h_dim
    device = args.device # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = args.data_path
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        mlflow.create_experiment('vae_readout')
    except Exception as e:
        print(e)
    experiment = mlflow.set_experiment('vae_readout')
    run_id = torch.randint(0, 1000, (1,)).item()
    mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f'run_{run_id}')
    print(vars(args))
    log_params(vars(args))


    ckpt = torch.load(ccvae_path)
    ccvae = Conv_CVAE(latent_dim = latent_dim,n_cls=n_cls).to(device)
    ccvae.load_state_dict(ckpt); ccvae.eval()
    print("Loaded ccvae checkpoint.")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # dataset = RecDataset(ccvae, train_size, n_cls, latent_dim)
    # train_set, val_set = torch.utils.data.random_split(dataset, [round(0.8*len(dataset)), round(0.2*len(dataset))])
    # test_set = RecDataset(ccvae, test_size, n_cls, latent_dim)
    # print("instantiated custom dataset")
    dataset = datasets.MNIST(root=root, download=True, train=True, transform=transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [round(0.8*len(dataset)), round(0.2*len(dataset))])
    test_set = datasets.MNIST(root=root, download=True, train=False, transform=transform)
    print("Loaded MNIST")
    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    fill = torch.zeros([n_cls, n_cls, img_size, img_size]).to(device)
    for i in range(n_cls):
        fill[i, i, :, :] = 1
    onehot = torch.zeros(n_cls, n_cls).to(device)
    onehot = onehot.scatter_(1, torch.LongTensor(range(n_cls)).view(n_cls,1).to(device), 1).view(n_cls, n_cls, 1, 1)

    def reconstruct(ccvae, images, labels, device=device):
        y_ = labels # (torch.rand(images.size(0), 1) * n_cls).type(torch.LongTensor).squeeze().to(device)
        y = onehot[y_]
        labels_fill_ = fill[labels]
        rec_x, _, _ = ccvae((images, labels_fill_, y))
        return rec_x

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
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                # reconstruct from ccvae
                rec_x = reconstruct(ccvae, images, labels)
                rec, mu, logvar, _ = model(rec_x)
                loss = loss_function(rec, images, mu, logvar, criterion) # or use rec_x for ground truth?
                running_loss += loss/batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                log_metric("train/loss", loss.item()/batch_size)
                
            val_loss = 0
            for batch_idx, (images, labels) in enumerate(val_loader):
                with torch.no_grad():
                    images, labels = images.to(device), labels.to(device)
                    rec_x = reconstruct(ccvae, images, labels)
                    rec, mu, logvar, _ = model(rec_x)                    
                    loss = loss_function(rec, images, mu, logvar, criterion)
                    val_loss += loss/batch_size
                    log_metric("valid/loss", loss.item()/batch_size)

            print(f"Epoch: {i+1} \t Train Loss: {running_loss/n_train:.2f} \t Val Loss: {val_loss/n_val:.2f}")
            log_metric('train/epoch_loss', running_loss.item()/n_train, step=i)
            log_metric('val/epoch_loss', val_loss.item()/n_val, step=i)
            torch.save(model.state_dict(), vae_path)
    else:
        model = MLPVAE(latent_dim = latent_dim, input_img_size=img_size).to(device)
        model.load_state_dict(torch.load(vae_path))
        model.eval()
        print(model)

    if plot_vae:
        latent_space_viz(latent_plot_path, model, test_loader, 2000)
        plot_reconstruction(recon_plot_path, model, test_loader)

    if train_readout:
        readout = Readout(latent_dim=latent_dim, h=readout_h_dim, n_classes=n_cls).to(device)
        print(readout)
        readout_opt = optim.Adam(readout.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(reduction='sum')
        for i in tqdm(range(num_epochs_readout)):
            train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0
            for _, (images, labels) in enumerate(train_loader):
                images, labels=images.to(device), labels.to(device)
                rec_x = reconstruct(ccvae, images, labels)
                loss, pred = loss_readout_OG(readout, model, rec_x, labels, criterion)
                readout_opt.zero_grad()
                loss.backward()
                readout_opt.step()
                train_loss += loss/batch_size
                train_acc += (torch.argmax(pred,1)==labels).float().sum()
                
            readout.eval()
            for _, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                rec_x = reconstruct(ccvae, images, labels)
                loss, pred = loss_readout_OG(readout, model, rec_x, labels, criterion)
                val_loss += loss/batch_size
                val_acc += (torch.argmax(pred,1)==labels).float().sum()
            
            val_acc /= (batch_size*len(val_loader))
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

    mlflow.end_run()

if __name__ == '__main__':
    main()