#!/bin/bash

#SBATCH --job-name=mnist_conv_cvae
#SBATCH --output=mnist_conv_cvae_output.txt
#SBATCH --error=mnist_conv_cvae_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=00:30:00

source $HOME/test/bin/activate
cp /network/datasets/mnist/ $SLURM_TMPDIR
# python $HOME/ambiguous-dataset/ambiguous/train/train_MNIST_final_ccvae.py --path $SLURM_TMPDIR --data_path $SLURM_TMPDIR --train_readout --generate_amnist
# cp $SLURM_TMPDIR/conv_cvae_mnist.pth $SCRATCH
# cp $SLURM_TMPDIR/reconstructions.png $SCRATCH
python $HOME/ambiguous-dataset/ambiguous/train/train_vae_readout.py --vae_path $SLURM_TMPDIR/mlp_vae.pth --readout_path $SLURM_TMPDIR/readout.pth --data_path $SLURM_TMPDIR --train_vae --plot_vae --train_readout
cp $SLURM_TMPDIR/mlp_vae.pth $SCRATCH
cp $SLURM_TMPDIR/readout.pth $SCRATCH
