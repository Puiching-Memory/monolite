# !/bin/bash

export MASTER_ADDR=localhost
export MASTER_PORT=15132
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=1,7
export OMP_NUM_THREADS=4
export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE
#export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
# 默认使用CUDA>12,使用本地ptxas启用
device_count=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')

# python -m torch.distributed.run \
# --nproc_per_node=$device_count --master_port 11451 \
# ../tools/train_val_ddp.py --config kitti_example.yaml

torchrun --standalone --nnodes=1 --nproc_per_node=$device_count ./tools/train_val_ddp.py --config ./kitti.yaml --ddp