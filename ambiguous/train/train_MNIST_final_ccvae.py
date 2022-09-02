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
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--train_cvae', default=False, action='store_true')
    parser.add_argument('--vae_path', type=str, default='')
    parser.add_argument('--readout_path', type=str, default='')
    parser.add_argument('--generate_amnist', default=False, action='store_true')


    args = parser.parse_args()
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
            torch.save(model.state_dict(), model_path+'/conv_cvae_mnist.pth')
            print(f"Epoch: {i+1} \t Train Loss: {running_loss:.2f} \t Val Loss: {val_loss:.2f}")
        
    else:
        ckpt = torch.load(model_path+'/conv_cvae_mnist.pth')
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
        vae = VAE().to(device)
        vae.load_state_dict(torch.load(vae_path))
        readout = Readout().to(device)
        readout.load_state_dict(torch.load(readout_path))

        size=1000
        train_size = 60000
        valid_size=10000
        test_size=10000
        save_freq = 20
        version = 'V2'
        threshold = 0.35
        mix_low = 0.3
        mix_high = 0.7
        n_iterations = 2000
        train_path = f'EMNIST_conv_dataset_{version}/train'
        valid_path = f'EMNIST_conv_dataset_{version}/valid'
        test_path = f'EMNIST_conv_dataset_{version}/test'
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(valid_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        count=0
        flag = 'test'

        size = 0
        if flag == 'train':
            size = train_size
        elif flag == 'valid':
            size = valid_size
        elif flag == 'test':
            size = test_size




        for i in tqdm(range(n_iterations)):
            z = torch.rand(size, latent_dim).to(device)
            y_ = (torch.rand(size, 1) * n_cls).type(torch.LongTensor).squeeze().to(device)
            y1_label_ = onehot[y_]
            y2_ = (torch.rand(size, 1) * n_cls).type(torch.LongTensor).squeeze().to(device)
            y2_label_ = onehot[y2_]

            w1 = torch.FloatTensor(1).uniform_(mix_low, mix_high).to(device)
            w2 = 1 - w1

            y_label_ = (w1*y1_label_ + w2*y2_label_)/2 # use range of 0.3 to 0.7 for mixing
            rec = ccvae.decode((z, y_label_))
            _, _, mu, _ = model(rec)
            pred = torch.softmax(readout(mu),dim=1)
            top2_idx = pred.topk(2)[0][:, 1]>threshold # use > 0.3
            amb = rec[top2_idx]
            clean_1 = ccvae.decode((z[top2_idx], y1_label_[top2_idx]))
            clean_2 = ccvae.decode((z[top2_idx], y2_label_[top2_idx]))
            labels = y_label_[top2_idx]

            torchvision.utils.save_image(amb, "good_ambig.pdf")
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
                if flag == 'train':
                    imgpath = train_path
                elif flag == 'valid':
                    imgpath = valid_path
                elif flag == 'test':
                    imgpath = test_path
                img_loc  = f'{imgpath}/{j+count}_image.npy'
                label_loc = f'{imgpath}/{j+count}_label.npy'
                np.save(img_loc, np_img)
                np.save(label_loc, np_label)

            count += clean_1.size(0)
            if (i+1) % save_freq == 0:
                print(count)
            if count >= size:
                break



if __name__ == '__main__':
    main()