#!/bin/bash

#SBATCH --job-name=vae
#SBATCH --output=amnist_readout_output.txt
#SBATCH --error=amnist_readout_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=01:00:00

source $HOME/test/bin/activate
module load libffi
cp -r /network/datasets/mnist/ $SLURM_TMPDIR
# python $HOME/ambiguous-dataset/ambiguous/train/train_MNIST_final_ccvae.py --path $SLURM_TMPDIR --data_path $SLURM_TMPDIR --train_readout --generate_amnist
# cp $SLURM_TMPDIR/{conv_cvae_mnist.pth,reconstructions.png} $SCRATCH

# python $HOME/ambiguous-dataset/ambiguous/train/train_vae_readout.py --plot_vae --train_readout --vae_path $SCRATCH/mlp_vae.pth --readout_path $SLURM_TMPDIR/readout.pth --data_path $SLURM_TMPDIR --latent_plot_path $SLURM_TMPDIR/latent.png --recon_plot_path $SLURM_TMPDIR/recon.png
# cp $SLURM_TMPDIR/{mlp_vae.pth,readout.pth,latent.png,recon.png} $SCRATCH

python $HOME/ambiguous-dataset/ambiguous/train/train_MNIST_final_ccvae.py --path $SCRATCH/conv_cvae_mnist.pth --vae_path $SCRATCH/mlp_vae.pth --readout_path $SCRATCH/readout.pth --data_path $SLURM_TMPDIR --generate_amnist --make_train --make_valid --make_test --train_path $SLURM_TMPDIR/amnist/train/ --valid_path $SLURM_TMPDIR/amnist/valid/ --test_path $SLURM_TMPDIR/amnist/test/ --n_iterations 5000
# zip -r $SLURM_TMPDIR/amnist.zip $SLURM_TMPDIR/amnist/ 
cp -r $SLURM_TMPDIR/amnist $SCRATCH