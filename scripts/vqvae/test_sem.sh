#!/bin/bash

python exp_vqvae.py --gpu_id=0 \
--dataset=sem \
--exp_name=baseline_vqvaegen \
--config_file=$trained_vqvae_config_file$ \
--mags=10000x,25000x,50000x,100000x \
--test_file=val_multi_mags.txt \
--stage=test \
--checkpoint_dir=$dir_to_trained_vqvae_checkpoint$ \
--checkpoint=-1 \

