#!/bin/bash

#SBATCH --partition=all
#SBATCH --nodelist=hpe162
#SBATCH --nodes=1
#SBATCH --job-name=ias
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=32

#SBATCH --container-image /purestorage/project/hgkim/enroot/images/ias.sqsh
#SBATCH --container-mounts=/purestorage:/purestorage
#SBATCH --container-writable