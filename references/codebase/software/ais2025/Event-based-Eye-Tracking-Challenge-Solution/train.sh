#!/usr/bin/env bash
# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

batch_size=32
lr=0.001
n_time_bins=4
num_epochs=800
train_length=45
val_length=45
test_length=45
train_stride=10
val_stride=10
test_stride=10
spatial_factor=0.125
map_type=binary
dataset=t_v
temporal_subsample_factor=1

run_name="cnn_trans_${batch_size}_${lr}_len${train_length}_stride_${train_stride}_tsf${temporal_subsample_factor}_sf${spatial_factor}_${dataset}_trans"

python3 train.py --batch_size $batch_size --lr $lr --n_time_bins $n_time_bins --num_epochs $num_epochs \
--train_length $train_length --val_length $val_length --test_length $test_length \
--train_stride $train_stride --val_stride $val_stride --test_stride $test_stride \
--spatial_factor $spatial_factor --map_type $map_type --dataset $dataset --run_name $run_name \
--temporal_subsample_factor $temporal_subsample_factor --device 7

# done
