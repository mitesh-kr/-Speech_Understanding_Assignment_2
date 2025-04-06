#!/bin/bash
#SBATCH --job-name=test_job            # Job name
#SBATCH --partition=fat                # Partition (check with `sinfo`)
#SBATCH --nodes=1                      # All processes on one node
#SBATCH --ntasks=1                     # One task
#SBATCH --cpus-per-task=4              # 4 CPU cores
#SBATCH --gres=gpu:2                   # Request 1 GPU
#SBATCH --output=training.log                # Log file
export OMP_NUM_THREADS=4
module load openmpi4

mpirun python3 /iitjhome/m23mac004/speechbrain/recipes/WHAMandWHAMR/separation/TRAIN_SEP_CLS.py /iitjhome/m23mac004/speechbrain/recipes/WHAMandWHAMR/separation/SEP_CLS.yaml >> /iitjhome/m23mac004/training.OUT
