#!/bin/bash

python exps.py --gpu_id=0 \
--exp_name=baseline_model \
--dataset=sem \
--config_file=$trained_clf_model_config$ \
--saved_model_dir=$dir_to_trained_clf_model$ \
--stage=test \
--test_dir=val_multi_mags.txt \
--target_train_dir=train_multi_mags.txt \
--adapt_type=train_s_bn_ep \
--style_config_file=$trained_style_model_config$ \
--style_checkpoint_file=$trained_style_model_checkpoint$ \
