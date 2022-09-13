#!/bin/bash

#SBATCH --job-name=CCVAE
#SBATCH --output=amnist_output2.txt
#SBATCH --error=amnist_error2.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=00:40:00

source $HOME/test/bin/activate
module load libffi
mkdir -p $SLURM_TMPDIR/MNIST/raw
cp -r /network/datasets/torchvision/MNIST $SLURM_TMPDIR
# python $HOME/ambiguous-dataset/ambiguous/train/train_MNIST_final_ccvae.py --train_cvae --path $SCRATCH/ccvae_mnistV2.pth --data_path $HOME/ambiguous-dataset/ambiguous/train/ --recon_path $SCRATCH/recon_ccvae_mnistV2.png

# python $HOME/ambiguous-dataset/ambiguous/train/train_vae_readout.py --num_epochs 25 --train_vae --plot_vae --train_readout --data_path $HOME/ambiguous-dataset/ambiguous/train/ --vae_path $SCRATCH/mlp_vae2.pth --readout_path $SCRATCH/readout2.pth --ccvae_path $SCRATCH/ccvae_mnistV2.pth --latent_plot_path $SCRATCH/latent2.png --recon_plot_path $SCRATCH/recon2.png

python $HOME/ambiguous-dataset/ambiguous/train/train_MNIST_final_ccvae.py --temperature 1. --path $SCRATCH/ccvae_mnistV2.pth --vae_path $SCRATCH/mlp_vae2.pth --readout_path $SCRATCH/readout2.pth --data_path $HOME/ambiguous-dataset/ambiguous/train/ --generate_amnist --make_train --make_valid --make_test --train_path $SCRATCH/amnistV2/train/ --valid_path $SCRATCH/amnistV2/valid/ --test_path $SCRATCH/amnistV2/test/ --n_iterations 10000
