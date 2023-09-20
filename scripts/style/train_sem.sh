#!/bin/bash

python style_manipulation.py --gpu_id=1 \
--dataset=sem \
--exp_name=style_exp \
--train_dir=domainA_dir \
--test_dir=domainB_dir \
--config_file=./configs/style/sem_config.yaml \

