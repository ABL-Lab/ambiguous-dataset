#!/bin/bash

#SBATCH --job-name=CCVAE
#SBATCH --output=amnist_output2.txt
#SBATCH --error=amnist_error2.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00

source $HOME/test/bin/activate
module load libffi
mkdir -p $SLURM_TMPDIR/MNIST/raw
cp -r /network/datasets/torchvision/MNIST $SLURM_TMPDIR
# python $HOME/ambiguous-dataset/ambiguous/train/train_MNIST_final_ccvae.py --path $SLURM_TMPDIR --data_path $SLURM_TMPDIR --train_readout --generate_amnist
# cp $SLURM_TMPDIR/{conv_cvae_mnist.pth,reconstructions.png} $SCRATCH

# python $HOME/ambiguous-dataset/ambiguous/train/train_vae_readout.py --num_epochs 25 --plot_vae --train_readout --data_path $SLURM_TMPDIR --vae_path $SCRATCH/mlp_vae2.pth --readout_path $SCRATCH/readout2.pth --ccvae_path $SCRATCH/conv_cvae_mnist.pth --latent_plot_path $SCRATCH/latent2.png --recon_plot_path $SCRATCH/recon2.png
# cp $SLURM_TMPDIR/{mlp_vae2.pth,readout2.pth,latent2.png,recon2.png} $SCRATCH

python $HOME/ambiguous-dataset/ambiguous/train/train_MNIST_final_ccvae.py --temperature 1. --path $SCRATCH/conv_cvae_mnist.pth --vae_path $SCRATCH/mlp_vae2.pth --readout_path $SCRATCH/readout2.pth --data_path $HOME/ambiguous-dataset/ambiguous/train/ --generate_amnist --make_test --train_path $SCRATCH/amnist/train/ --valid_path $SCRATCH/amnist/valid/ --test_path $SCRATCH/amnist/test3/ --n_iterations 10000
# zip -r $SLURM_TMPDIR/amnist.zip $SLURM_TMPDIR/amnist/ 
cp -r $SLURM_TMPDIR/amnist $SCRATCH