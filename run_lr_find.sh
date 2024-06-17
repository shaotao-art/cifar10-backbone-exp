#!/bin/bash

device_idx=6
max_steps=1000
config_file=./configs/config.py
mkdir $log_root

lrs=(1e-3 5e-4 3e-4)
for lr in "${lrs[@]}"; do
    run_name=find-lr-$lr
    CUDA_VISIBLE_DEVICES=$device_idx  python run.py \
                            --config $config_file\
                            --run_name $run_name \
                            --find_lr \
                            --max_steps $max_steps \
                            --lr $lr
done
