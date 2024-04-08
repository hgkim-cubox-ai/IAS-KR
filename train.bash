srun -p all --nodelist aten234 --gpus 8 --cpus-per-task=128 --job-name test --pty bash
enroot create --name ir_c11 /purestorage/project/byko/enroot_images/pytorch_cuda11_7.sqsh
enroot start --rw --mount /purestorage:/purestorage ir_c11
pip install wandb
pip install opencv-python-headless
pip install tensorboard
cd /purestorage/project/byko/IR/hwang_code/
python model/train_resnet.py
exit
enroot export --output /purestorage/project/byko/enroot_images/ir_c11.sqsh ir_c11
enroot remove ir_c11


#!/bin/bash

#SBATCH --partition=all
#SBATCH --nodelist=nv176
#SBATCH --nodes=1
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=8
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96

#SBATCH --container-image /purestorage/project/byko/enroot_images/ir_c11.sqsh
#SBATCH --container-mounts=/purestorage/project/byko/IR_refact/AI4-IR:/workspace,/purestorage/ir_liveness:/workspace/results,/purestorage/datasets/IR_dataset:/workspace/data,/purestorage:/purestorage
#SBATCH --container-writable