from distutils.archive_util import make_tarball
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
import ambiguous.models.cvae
from ambiguous.models.cvae import Conv_CVAE, VAE, Readout
from ambiguous.dataset.dataset import partition_dataset
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--vae_latent_dim', type=int, default=10)
    parser.add_argument('--readout_h_dim', type=int, default=512)
    parser.add_argument('--n_cls', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_path', type=str, default='')
    parser.add_argument('--valid_path', type=str, default='')
    parser.add_argument('--test_path', type=str, default='')
    parser.add_argument('--train_size', type=int, default=60000)
    parser.add_argument('--valid_size', type=int, default=10000)
    parser.add_argument('--test_size', type=int, default=10000)
    parser.add_argument('--n_iterations', type=int, default=2000)
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--mix_high', type=float, default=0.3)
    parser.add_argument('--mix_low', type=float, default=0.7)
    parser.add_argument('--version', type=str, default='V1')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--vae_path', type=str, default='')
    parser.add_argument('--readout_path', type=str, default='')
    parser.add_argument('--train_cvae', default=False, action='store_true')
    parser.add_argument('--generate_amnist', default=False, action='store_true')
    parser.add_argument('--make_train', default=False, action='store_true')
    parser.add_argument('--make_valid', default=False, action='store_true')
    parser.add_argument('--make_test', default=False, action='store_true')


    args = parser.parse_args()
    save_freq = args.save_freq # 20
    version = args.version # 'V2'
    threshold = args.threshold # 0.35
    mix_low = args.mix_low # 0.3
    mix_high = args.mix_high #  0.7
    n_iterations = args.n_iterations # 2000
    make_train = args.make_train
    make_valid = args.make_valid 
    make_test = args.make_test 
    train_size = args.train_size 
    valid_size = args.valid_size 
    test_size = args.test_size
    train_path = args.train_path
    valid_path = args.valid_path 
    test_path = args.test_path
    data_path = args.data_path
    model_path = args.path
    train_cvae = args.train_cvae
    vae_path = args.vae_path
    readout_path = args.readout_path
    generate_amnist = args.generate_amnist
    batch_size = args.batch_size # 64
    n_cls=args.n_cls # 10
    img_size=args.img_size # 28
    lr=args.lr # 1e-3
    vae_latent_dim=args.vae_latent_dim
    readout_h_dim = args.readout_h_dim
    num_epochs=args.num_epochs # 25
    latent_dim = args.latent_dim # 10
    device = args.device # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(device)
    root=data_path

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.MNIST(root=root, download=True, train=True, transform=transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [round(0.8*len(dataset)), round(0.2*len(dataset))])
    test_set = datasets.MNIST(root=root, download=True, train=False, transform=transform)
    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=2)

    onehot = torch.zeros(n_cls, n_cls).to(device)
    onehot = onehot.scatter_(1, torch.LongTensor(range(n_cls)).view(n_cls,1).to(device), 1).view(n_cls, n_cls, 1, 1)
    fill = torch.zeros([n_cls, n_cls, img_size, img_size]).to(device)
    for i in range(n_cls):
        fill[i, i, :, :] = 1
    if train_cvae:
        model = Conv_CVAE(
        latent_dim = latent_dim,
            n_cls=n_cls
        ).to(device)
        print(model)
        print("num parameters:", sum([x.numel() for x in model.parameters() if x.requires_grad]))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss(reduction='sum')
        n_train = len(train_loader)
        n_val = len(val_loader)
        for i in tqdm(range(num_epochs)):
            running_loss = 0
            for idx, (images, labels) in tqdm(enumerate(train_loader)):
                images = images.to(device)
                labels = labels.to(device)
                labels_fill_ = fill[labels]
                y_ = (torch.rand(images.size(0), 1) * n_cls).type(torch.LongTensor).squeeze().to(device)
                y_label_ = onehot[y_]
                rec, mu, logvar = model((images, labels_fill_, y_label_))
                loss_dict = model.loss_function(rec, images, mu, logvar)
                running_loss += loss_dict['loss'].item()/batch_size
                optimizer.zero_grad()
                loss_dict['loss'].backward()
                optimizer.step()
                # wandb.log(loss_dict)
            val_loss = 0
            for _, (images, labels) in tqdm(enumerate(val_loader)):
                with torch.no_grad():
                    images = images.to(device)
                    labels = labels.to(device)
                    labels_fill_ = fill[labels]
                    y_ = (torch.rand(images.size(0), 1) * n_cls).type(torch.LongTensor).squeeze()
                    y_label_ = onehot[y_]
                    rec, mu, logvar = model((images, labels_fill_, y_label_))
                    torchvision.utils.save_image(rec, "reconstruction_valid.pdf")
                    loss_dict = model.loss_function(rec, images, mu, logvar)
                    val_loss += loss_dict['loss'].item()/batch_size
                    loss_dict = {'loss_val':loss_dict['loss'], 'kld_val':loss_dict['kld'], 'rec_val':loss_dict['rec']}
                    # wandb.log(loss_dict)
            torch.save(model.state_dict(), model_path)
            print(f"Epoch: {i+1} \t Train Loss: {running_loss:.2f} \t Val Loss: {val_loss:.2f}")
        
    else:
        ckpt = torch.load(model_path)
        model = Conv_CVAE(latent_dim = latent_dim,n_cls=n_cls).to(device)
        model.load_state_dict(ckpt)
        
    model.eval()
    images, labels = next(iter(val_loader))
    with torch.no_grad():
        images = images.to(device)
        labels = labels.to(device)
        labels_fill_ = fill[labels]
        y_ = (torch.rand(images.size(0), 1) * n_cls).type(torch.LongTensor).squeeze()
        y_label_ = onehot[y_]
        rec, mu, logvar = model((images, labels_fill_, y_label_))
    torchvision.utils.save_image(rec, model_path+"/reconstructions.png")

    if generate_amnist:
        vae = MLPVAE(latent_dim=vae_latent_dim, input_img_size=img_size).to(device)
        vae.load_state_dict(torch.load(vae_path))
        readout = Readout(latent_dim=vae_latent_dim, h=readout_h_dim, n_classes=n_cls).to(device)
        readout.load_state_dict(torch.load(readout_path))

        onehot = torch.zeros(n_cls, n_cls).to(device)
        onehot = onehot.scatter_(1, torch.LongTensor(range(n_cls)).view(n_cls,1).to(device), 1).view(n_cls, n_cls, 1, 1)

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(valid_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        count=0
        sizes = {}
        dpaths = {}
        if make_train:
            sizes['train'] = train_size
            dpaths['train'] = train_path
        if make_valid:
            sizes['valid'] = valid_size
            dpaths['valid'] = valid_path
        if make_test:
            sizes['test'] = test_size
            dpaths['test'] = test_path
        for dset in sizes:
            for i in tqdm(range(n_iterations)):
                z = torch.rand(sizes[dset], latent_dim).to(device)
                y_ = (torch.rand(sizes[dset], 1) * n_cls).type(torch.LongTensor).squeeze().to(device)
                y1_label_ = onehot[y_]
                y2_ = (torch.rand(sizes[dset], 1) * n_cls).type(torch.LongTensor).squeeze().to(device)
                y2_label_ = onehot[y2_]

                w1 = torch.FloatTensor(1).uniform_(mix_low, mix_high).to(device)
                w2 = 1 - w1

                y_label_ = (w1*y1_label_ + w2*y2_label_)/2 # use range of 0.3 to 0.7 for mixing
                rec = model.decode((z, y_label_))
                _, _, mu, _ = vae(rec)
                pred = torch.softmax(readout(mu),dim=1)
                top2_idx = pred.topk(2)[0][:, 1]>threshold # use > 0.3
                amb = rec[top2_idx]
                clean_1 = model.decode((z[top2_idx], y1_label_[top2_idx]))
                clean_2 = model.decode((z[top2_idx], y2_label_[top2_idx]))
                labels = y_label_[top2_idx]

                h=28
                for j, (img_c1, img_amb, img_c2, label) in enumerate(zip(clean_1, amb, clean_2, labels)):
                    np1 = (255*img_c1).cpu().detach().numpy().astype(np.uint8)
                    np_amb = (255*img_amb).cpu().detach().numpy().astype(np.uint8)
                    np2 = (255*img_c2).cpu().detach().numpy().astype(np.uint8)
                    np_img = np.zeros((1, h, h*3)).astype(np.uint8)
                    np_img[:, :, :h] = np1
                    np_img[:, :, h:2*h] = np_amb
                    np_img[:, :, 2*h:3*h] = np2
                    np_label = label.cpu().detach().numpy().astype(np.uint8)

                    np.save(f'{dpaths[dset]}/{j+count}_image.npy', np_img)
                    np.save(f'{dpaths[dset]}/{j+count}_label.npy', np_label)
                
                count += clean_1.size(0)
                if (i+1) % save_freq == 0:
                    print(count)
                if count >= sizes[dset]:
                    c=0
                    break
        torchvision.utils.save_image(amb, "good_ambig.pdf")



if __name__ == '__main__':
    main()