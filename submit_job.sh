#!/bin/bash
#SBATCH --output=logs/output_%j.log       # Log file
#SBATCH --error=logs/error_%j.log         # Error file
#SBATCH --gres=gpu:A40                    # GPUs required

# Usage:
# sbatch submit_job.sh SDI_SANA_nerf.py --n-opt 50 --n-fid 10 --lr 1e-2
# sbatch submit_job.sh SDI_SANA_nerf.py --use-cg
# sbatch submit_job.sh samp_SANA_nerf.py

# Load Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gaussianimage

cd ~/threestudio   # Replace with the absolute path to your folder

# Display GPU information
nvidia-smi

# Run the Python script with arguments
python "$@"
