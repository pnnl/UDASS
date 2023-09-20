#!/bin/bash

python exp_vqvae.py --gpu_id=2 \
--dataset=sem \
--exp_name=baseline_vqvaegen \
--train_file=train_multi_mags.txt \
--mags=10000x,25000x,50000x,100000x \
--test_file=val_multi_mags.txt \
--config_file=./configs/vqvae/sem_config.yaml \
--seed=1 \
--style_config_file=$trained_style_model_config$ \
--style_checkpoint_file=$trained_style_model_checkpoint$ \

