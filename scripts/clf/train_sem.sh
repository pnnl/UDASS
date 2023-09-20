#!/bin/bash

python exps.py --gpu_id=3 \
--exp_name=baseline_model \
--dataset=sem \
--config_file=./configs/clf/clf_sem_config.yaml \
--train_dir=train_multi_mags.txt \
--val_dir=val_multi_mags.txt \
--style_config_file=$trained_style_model_config$ \
--style_checkpoint_file=$trained_style_model_checkpoint$ \
