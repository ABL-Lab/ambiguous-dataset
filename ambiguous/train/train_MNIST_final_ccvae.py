from distutils.archive_util import make_tarball
import os
import importlib
from pickle import TRUE
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
from ambiguous.models.cvae import ConvolutionalVAE
from ambiguous.models.vae import MLPVAE
from ambiguous.models.readout import Readout
from ambiguous.dataset.dataset import partition_dataset
import argparse
import glob
import mlflow
from mlflow import log_metrics, log_metric, log_artifact, log_artifacts, log_params

BATCH_SIZE_AMNIST = 512
T_CLEAN = 0.6 # clean image threshold, ie. model > 75% confident in top-1 prediction

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
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--train_path', type=str, default='')
    parser.add_argument('--valid_path', type=str, default='')
    parser.add_argument('--test_path', type=str, default='')
    parser.add_argument('--train_size', type=int, default=60000)
    parser.add_argument('--valid_size', type=int, default=10000)
    parser.add_argument('--test_size', type=int, default=10000)
    parser.add_argument('--n_iterations', type=int, default=2000)
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--mix_low', type=float, default=0.3)
    parser.add_argument('--mix_high', type=float, default=0.7)
    parser.add_argument('--threshold', type=float, default=0.35)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--version', type=str, default='V1')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--vae_path', type=str, default='')
    parser.add_argument('--readout_path', type=str, default='')
    parser.add_argument('--recon_path', type=str, default='')
    parser.add_argument('--train_cvae', default=False, action='store_true')
    parser.add_argument('--conditional', default=False, action='store_true')
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
    temperature = args.temperature
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
    dataset = args.dataset
    model_path = args.path
    train_cvae = args.train_cvae
    recon_path = args.recon_path
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
    device = args.device
    conditional = args.conditional
    seed = args.seed

    try:
        mlflow.create_experiment('ccvae')
    except Exception as e:
        print(e)
    experiment = mlflow.set_experiment('ccvae')
    run_id = torch.randint(0, 1000, (1,)).item()
    mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f'run_{run_id}')
    print(vars(args))
    log_params(vars(args))

    # np.random.seed(seed)
    # torch.manual_seed(seed)

    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = datasets.MNIST(root=data_path, download=True, train=True, transform=transform)
        train_set, val_set = torch.utils.data.random_split(dataset, [round(0.8*len(dataset)), round(0.2*len(dataset))])
        test_set = datasets.MNIST(root=data_path, download=True, train=False, transform=transform)
        print("Loaded MNIST")
    elif dataset == 'emnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            lambda x: x.rot90(1,[2,1]).flip(2)
        ])
        dataset = datasets.EMNIST(root=data_path, download=True, split='byclass', train=True, transform=transform)
        new_dataset = partition_dataset(dataset, range(10, 36))
        train_set, val_set = torch.utils.data.random_split(new_dataset, [round(0.8*len(new_dataset)), round(0.2*len(new_dataset))])
        test_set = datasets.EMNIST(root=data_path, download=True, split='byclass', train=False, transform=transform)

    else:
        print("dataset not supported")
    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    fill = torch.zeros([n_cls, n_cls, img_size, img_size]).to(device)
    for i in range(n_cls):
        fill[i, i, :, :] = 1
    onehot = torch.zeros(n_cls, n_cls).to(device)
    onehot = onehot.scatter_(1, torch.LongTensor(range(n_cls)).view(n_cls,1).to(device), 1).view(n_cls, n_cls, 1, 1)
    if conditional:
        onehot = nn.Upsample(scale_factor=4)(onehot)

    def reconstruct(ccvae, images, labels, device=device):
        y_ = labels # (torch.rand(images.size(0), 1) * n_cls).type(torch.LongTensor).squeeze().to(device)
        y = onehot[y_]
        labels_fill_ = fill[labels]
        rec_x, _, _, _ = ccvae((images, labels_fill_, y))
        return rec_x, labels_fill_, y

    def reconstruct_amb(model, images1, y1_label_, y2_label_, label_fill1, label_fill2):
        y_label_ = y1_label_ + y2_label_
        label_fill = label_fill1 + label_fill2
        amb, _, _, _ = model((images1, label_fill, y_label_))
        return amb

    def get_mu(vae, clean_1, amb, clean_2):
        _, _, mu_clean_1, _ = vae(clean_1)
        _, _, mu_clean_2, _ = vae(clean_2)
        _, _, mu, _ = vae(amb)
        return mu_clean_1, mu, mu_clean_2

    def get_good_idxs(readout, mu_clean_1, mu, mu_clean_2, y1_label_):
        pred = torch.softmax(readout(mu)/temperature,dim=1)
        pred_clean1 = torch.softmax(readout(mu_clean_1), dim=1)
        pred_clean2 = torch.softmax(readout(mu_clean_2), dim=1)
        max_clean1 = pred_clean1.argmax(1)
        max_clean2 = pred_clean2.argmax(1)
        top2_softprobs_amb = pred.topk(2)[0][:, 1]
        top2_label_amb = pred.topk(2)[1][:, 1]
        bistable = torch.logical_or(pred.argmax(1)==max_clean1, pred.argmax(1)==max_clean2)
        bistable2 = torch.logical_or(top2_label_amb==max_clean1, top2_label_amb==max_clean2)
        good_idxs = torch.logical_and(max_clean1==y1_label_.squeeze().argmax(1)[:,0,0], max_clean1!=max_clean2)
        good_idxs = torch.logical_and(good_idxs, top2_softprobs_amb > threshold)
        good_idxs = torch.logical_and(good_idxs, bistable)
        good_idxs = torch.logical_and(good_idxs, bistable2)
        return good_idxs, max_clean1, max_clean2, pred_clean1, pred_clean2, top2_softprobs_amb

    def make_example(img_c1, img_amb, img_c2, label1, label2, img_size=img_size):
        #TODO: fix problems hereeee
        np1 = (255*torch.sigmoid(img_c1)).cpu().detach().numpy().astype(np.uint8)
        np_amb = (255*torch.sigmoid(img_amb)).cpu().detach().numpy().astype(np.uint8)
        np2 = (255*torch.sigmoid(img_c2)).cpu().detach().numpy().astype(np.uint8)
        fig_path='./np_clean1.pdf'
        fig_path2='./np_clean1_255.pdf'
        fig_path3='./img_c1.pdf'
        torchvision.utils.save_image(img_c1, fig_path3)
        torchvision.utils.save_image(torch.from_numpy(np1/255.), fig_path)
        torchvision.utils.save_image(torch.from_numpy(np1.astype(np.float32)), fig_path2)
        log_artifact(fig_path)
        log_artifact(fig_path2)
        log_artifact(fig_path3)

        np_img = np.zeros((1, img_size, img_size*3)).astype(np.uint8)
        np_img[:, :, :img_size] = np1
        np_img[:, :, img_size:2*img_size] = np_amb
        np_img[:, :, 2*img_size:3*img_size] = np2
        np_label = torch.tensor([label1.item(), label2.item()]).cpu().detach().numpy().astype(np.uint8)
        return np_img, np_label

    def save_images(clean_1, amb, clean_2, labels1, labels2, top2_softprobs_amb):
        for j, (img_c1, img_amb, img_c2, label1, label2, top2_softprob) in enumerate(zip(clean_1, amb, clean_2, labels1, labels2, top2_softprobs_amb)):
            np_img, np_label = make_example(img_c1, img_amb, img_c2, label1, label2)
            top2_softprob = top2_softprob.cpu().detach().numpy()
            clean1_softprob = pred_clean1[good_idxs][j]
            clean2_softprob = pred_clean2[good_idxs][j]
            
            if (i+1) % save_freq == 0:
                print(np_label)
            np.save(f'{dpaths[dset]}/{j+count}_image.npy', np_img)
            np.save(f'{dpaths[dset]}/{j+count}_label.npy', np_label)
        return np_img, np_label,top2_softprob, clean1_softprob, clean2_softprob

    if train_cvae:
        model = ConvolutionalVAE(
        latent_dim = latent_dim,
            n_cls=n_cls,
            conditional=conditional
        ).to(device)
        print(model)
        print("num parameters:", sum([x.numel() for x in model.parameters() if x.requires_grad]))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss(reduction='sum')
        n_train = len(train_loader)
        n_val = len(val_loader)
        for i in tqdm(range(num_epochs)):
            running_loss = 0
            for batch_idx, (images, labels) in tqdm(enumerate(train_loader)):
                images = images.to(device)
                labels = labels.to(device)
                labels_fill_ = fill[labels]
                y_ = labels # (torch.rand(images.size(0), 1) * n_cls).type(torch.LongTensor).squeeze().to(device)
                y_label_ = onehot[y_]
                rec, x, mu, logvar = model((images, labels_fill_, y_label_))
                loss_dict = model.loss_function(rec, images, mu, logvar)
                running_loss += loss_dict['loss'].item()/batch_size
                optimizer.zero_grad()
                loss_dict['loss'].backward()
                optimizer.step()
                log_metric('loss_batch/train',loss_dict['loss'].item()/batch_size)

            val_loss = 0
            for batch_idx, (images, labels) in tqdm(enumerate(val_loader)):
                with torch.no_grad():
                    images = images.to(device)
                    labels = labels.to(device)
                    labels_fill_ = fill[labels]
                    y_ = labels # (torch.rand(images.size(0), 1) * n_cls).type(torch.LongTensor).squeeze()
                    y_label_ = onehot[y_]
                    rec, x, mu, logvar = model((images, labels_fill_, y_label_))
                    torchvision.utils.save_image(rec, "reconstruction_valid.pdf")
                    loss_dict = model.loss_function(rec, images, mu, logvar)
                    val_loss += loss_dict['loss'].item()/batch_size
                    log_metric('loss_batch/val', loss_dict['loss'].item()/batch_size)
            torch.save(model.state_dict(), model_path)
            print(f"Epoch: {i+1} \t Train Loss: {running_loss:.2f} \t Val Loss: {val_loss:.2f}")
            log_metric('loss_epoch/train', running_loss/len(train_loader),step=i)
            log_metric('loss_epoch/val', val_loss/len(val_loader), step=i)

        images, labels = next(iter(val_loader))
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            labels_fill_ = fill[labels]
            y_ = labels # (torch.rand(images.size(0), 1) * n_cls).type(torch.LongTensor).squeeze()
            y_label_ = onehot[y_]
            rec, x, mu, logvar = model((images, labels_fill_, y_label_))
        torchvision.utils.save_image(rec, recon_path)
        
    else:
        ckpt = torch.load(model_path)
        model = ConvolutionalVAE(latent_dim = latent_dim,n_cls=n_cls,conditional=conditional).to(device)
        model.load_state_dict(ckpt)
        print("Loaded ccvae checkpoint.")
        
    model.eval()


    if generate_amnist:
        vae = ConvolutionalVAE(latent_dim = vae_latent_dim, conditional=False).to(device)
        vae.load_state_dict(torch.load(vae_path))
        readout = Readout(latent_dim=vae_latent_dim, h=readout_h_dim, n_classes=n_cls).to(device)
        readout.load_state_dict(torch.load(readout_path))
        vae.eval(); readout.eval()
        print("Loaded vae and readout checkpoints.")

        # onehot = torch.zeros(n_cls, n_cls).to(device)
        # onehot = onehot.scatter_(1, torch.LongTensor(range(n_cls)).view(n_cls,1).to(device), 1).view(n_cls, n_cls, 1, 1)

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(valid_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
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
            print(dset)
            count=0
            i=0
            train_loader1 = DataLoader(train_set, batch_size=BATCH_SIZE_AMNIST,shuffle=True)
            train_loader2 = DataLoader(train_set, batch_size=BATCH_SIZE_AMNIST,shuffle=True)

            while count < sizes[dset]:
                i+=1
                for idx, ((images1, t1), (images2, t2)) in enumerate(zip(train_loader1, train_loader2)):
                    images1,t1,images2,t2 = images1.to(device),t1.to(device),images2.to(device),t2.to(device)
                    clean_1, label_fill1, y1_label_, = reconstruct(model, images1, t1)
                    clean_2, label_fill2, y2_label_ = reconstruct(model, images2, t2)
                    if idx % 10 == 0:
                        torchvision.utils.save_image((clean_1*255).int()/255., f'clean_recon_{idx}.pdf')
                        mlflow.log_artifact(f'clean_recon_{idx}.pdf')
                    amb = reconstruct_amb(model, images1, y1_label_, y2_label_, label_fill1, label_fill2)
                    mu_clean_1, mu, mu_clean_2 = get_mu(vae, clean_1, amb, clean_2)
                    good_idxs, max_clean1, max_clean2, pred_clean1, pred_clean2, top2_softprobs_amb = get_good_idxs(readout, mu_clean_1, mu, mu_clean_2, y1_label_)
                    amb, clean_1, clean_2, labels1, labels2, top2_softprobs_amb = amb[good_idxs], clean_1[good_idxs], clean_2[good_idxs], max_clean1[good_idxs], max_clean2[good_idxs], top2_softprobs_amb[good_idxs]               
                    if clean_1.size(0)>0:
                        np_img, np_label, top2_softprob, clean1_softprob, clean2_softprob = save_images(clean_1, amb, clean_2, labels1, labels2, top2_softprobs_amb)
                    count += clean_1.size(0)
                    step = idx+i*len(train_loader)  
                    log_metric('count', count, step=step)
                    if step % save_freq == 0 and clean_1.size(0) > 0:
                        print(f"iter: {step}, count: {count}")
                        fig=plt.figure()
                        plt.imshow(np_img.transpose(1,2,0)/255.)
                        plt.title((np_label,round(top2_softprob.max().item(),3), round(clean1_softprob.max().item(),3), round(clean2_softprob.max().item(),3)),fontsize=15)
                        fig_path = f'{dpaths[dset]}/{count}.png'
                        fig.savefig(fig_path)
                        log_artifact(fig_path)
                        torchvision.utils.save_image(torch.from_numpy(np_img)/255.,fig_path[:-4]+'.pdf')
                        log_artifact(fig_path[:-4]+'.pdf')

            print("Reached dataset size")
            # torchvision.utils.save_image(torch.from_numpy(np_img), "good_ambig.png")

    mlflow.end_run()


if __name__ == '__main__':
    main()