#!/bin/bash

#SBATCH --job-name=CCVAE
#SBATCH --output=emnist_output.txt
#SBATCH --error=emnist_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=01:00:00

source $HOME/test/bin/activate
module load libffi
# python $HOME/ambiguous-dataset/ambiguous/train/train_MNIST_final_ccvae.py --dataset emnist --n_cls 26 --train_cvae --path $SCRATCH/ccvae_emnist.pth --data_path $HOME/ambiguous-dataset/ambiguous/train/ --recon_path $SCRATCH/recon_ccvae_emnist.png

# python $HOME/ambiguous-dataset/ambiguous/train/train_vae_readout.py --dataset emnist --n_cls 26 --num_epochs 50 --train_vae --plot_vae --train_readout --data_path $HOME/ambiguous-dataset/ambiguous/train/ --vae_path $SCRATCH/mlp_vae_emnist.pth --readout_path $SCRATCH/readout_emnist.pth --ccvae_path $SCRATCH/ccvae_emnist.pth --latent_plot_path $SCRATCH/latent_emnist.png --recon_plot_path $SCRATCH/recon_emnist.png
python $HOME/ambiguous-dataset/ambiguous/train/train_vae_readout.py --dataset emnist --n_cls 26 --data_path $HOME/ambiguous-dataset/ambiguous/train/ --vae_path $SCRATCH/mlp_vae_emnist.pth --readout_path $SCRATCH/readout_emnist.pth --ccvae_path $SCRATCH/ccvae_emnist.pth --latent_plot_path $SCRATCH/latent_emnist.png --recon_plot_path $SCRATCH/recon_emnist.png

# python $HOME/ambiguous-dataset/ambiguous/train/train_MNIST_final_ccvae.py --dataset emnist --n_cls 26 --temperature 1. --path $SCRATCH/ccvae_emnist.pth --vae_path $SCRATCH/mlp_vae_emnist.pth --readout_path $SCRATCH/readout_emnist.pth --data_path $HOME/ambiguous-dataset/ambiguous/train/ --generate_amnist --make_train --make_valid --make_test --train_path $SCRATCH/emnist/train/ --valid_path $SCRATCH/emnist/valid/ --test_path $SCRATCH/emnist/test/ --n_iterations 10000
