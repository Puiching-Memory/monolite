# !/bin/bash

export MASTER_ADDR=localhost
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
device_count=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')

# python -m torch.distributed.run \
# --nproc_per_node=$device_count --master_port 11451 \
# ../tools/train_val_ddp.py --config kitti_example.yaml

#torchrun --standalone --nnodes=1 --nproc_per_node=$device_count ./tools/train_val_ddp.py --config ./kitti.yaml --ddp -e
python ./tools/train_val_ddp.py --config ./kitti_val.yaml -e