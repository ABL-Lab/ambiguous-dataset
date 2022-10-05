#!/bin/bash

#SBATCH --job-name=CCVAE
#SBATCH --output=amnist_output3.txt
#SBATCH --error=amnist_error3.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=01:00:00

source $HOME/test/bin/activate
module load libffi
# python $HOME/ambiguous-dataset/ambiguous/train/train_MNIST_final_ccvae.py --train_cvae --path $SCRATCH/ccvae_mnistV2.pth --data_path $HOME/ambiguous-dataset/ambiguous/train/ --recon_path $SCRATCH/recon_ccvae_mnistV2.png

# python $HOME/ambiguous-dataset/ambiguous/train/train_vae_readout.py --num_epochs 25 --train_vae --plot_vae --train_readout --data_path $HOME/ambiguous-dataset/ambiguous/train/ --vae_path $SCRATCH/mlp_vae2.pth --readout_path $SCRATCH/readout2.pth --ccvae_path $SCRATCH/ccvae_mnistV2.pth --latent_plot_path $SCRATCH/latent2.png --recon_plot_path $SCRATCH/recon2.png

# python $HOME/ambiguous-dataset/ambiguous/train/train_MNIST_final_ccvae.py --dataset mnist --train_cvae --conditional --recon_path $SCRATCH/recon_ccvae_mnistV3.png --path $SCRATCH/ccvae_mnistV3.pth --vae_path $SCRATCH/mlp_vae2.pth --readout_path $SCRATCH/readout2.pth --data_path $HOME/ambiguous-dataset/ambiguous/train/ --generate_amnist --make_train --make_valid --make_test --train_path $SCRATCH/amnistV3/train/ --valid_path $SCRATCH/amnistV3/valid/ --test_path $SCRATCH/amnistV3/test/ --n_iterations 10000
python $HOME/ambiguous-dataset/ambiguous/train/train_MNIST_final_ccvae.py --dataset mnist --conditional --path $SCRATCH/ccvae_mnistV3.pth --vae_path $SCRATCH/mlp_vae2.pth --readout_path $SCRATCH/readout2.pth --data_path $HOME/ambiguous-dataset/ambiguous/train/ --generate_amnist --make_train --make_valid --make_test --train_path $SCRATCH/amnistV3/train/ --valid_path $SCRATCH/amnistV3/valid/ --test_path $SCRATCH/amnistV3/test/ --n_iterations 10000
