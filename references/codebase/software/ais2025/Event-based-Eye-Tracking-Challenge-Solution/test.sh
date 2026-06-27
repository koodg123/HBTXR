#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6
test_length=45
test_stride=10
spatial_factor=0.125
map_type=binary
temporal_subsample_factor=1
checkpoint=./ckpt/final.pth

python3 test.py --test_length $test_length --test_stride $test_stride \
--spatial_factor $spatial_factor --map_type $map_type --checkpoint $checkpoint \
--temporal_subsample_factor $temporal_subsample_factor 
