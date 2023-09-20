import numpy as np
import os
import argparse
import sys
import yaml

import torch
from torchvision import datasets

from datasets.style_sem import *
from models.style_model import *
import utils.style_utils as sutils
import utils.utils as utils

def main():

    ## Parse input from command prompt
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default = 0, help = 'GPU#')
    parser.add_argument('--dataset', type=str, default = '', help = 'Dataset')
    parser.add_argument('--train_dir', type=str, default = '', help = 'Root directory of train dataset')
    parser.add_argument('--sub_dir', type=str, default = '', help = 'Sub directory')
    parser.add_argument('--test_dir', type=str, default = '', help = 'Root directory of val dataset')
    parser.add_argument('--config_file', type=str, default = 'sem_config.yaml', help = 'Params file')
    parser.add_argument('--exp_name', type=str, default = 'test', help = 'Name of current experiment')
    parser.add_argument('--stage', type=str, default='train', help = 'Model stage')

    parser.add_argument('--checkpoint_dir', type=str, default = './checkpoints/style/', help = 'Directory of stored checkpoint')
    parser.add_argument('--checkpoint', type=int, default = -1, help = 'Checkpoint to be restored')
    parser.add_argument('--gen_dir', type=str, default = './gen_imgs', help = 'Directory for generated images')
    parser.add_argument('--num_styles', type=int, default = 3, help = 'Number of random styles to be generated')

    params, unparsed = parser.parse_known_args()

    ## Load config file
    config_params = utils.load_config_file(params, 'style')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu_id)

    ## Create logger
    logger_name = './results/logs/style/test_' if params.stage=='test' else './results/logs/style/'
    logger_name += '%s_%s.log' %(params.dataset,params.exp_name)
    logger = utils.create_logger(logger_name)
    sutils.log_params(logger,params)

    style_model = Style_Model(params, config_params)

    ## Create dataset
    dataloader_dict = {}
    if params.stage == 'train' or params.stage == 'viz':
        datatypes = ['train','test']
    else:
        datatypes = ['test']

    new_size = (config_params['input_res'],config_params['input_res']) if 'input_res' in config_params else None
    for dt in datatypes:
        curr_dir = params.train_dir if dt == 'train' else params.test_dir

        if params.dataset == 'sem':
            dataset = SEM_Dataset(logger, curr_dir, dt, n_channel=config_params['input_dim'], new_size=new_size,
                                    subdir=params.sub_dir)
        else:
            print('No compatible dataset!')
            return

        if params.stage == 'viz':
            config_params['batch_size'] = 12

        dataloader_dict[dt] = torch.utils.data.DataLoader(dataset=dataset, num_workers=4, shuffle=True,\
                                batch_size=config_params['batch_size']) \
                                if dt == 'train' or params.stage != 'train'\
                                else dataset


        logger.info("# of {:} files: {:}".format(dt, len(dataset)))
        print("Finished loading data!\n")


    if params.stage == 'train':
        style_model.train(dataloader_dict, logger)
    elif params.stage == 'test':
        style_model.inference(dataloader_dict['test'], params)
    elif params.stage == 'viz':
        style_model.viz_dataset(dataloader_dict, params.dataset)

if __name__ == '__main__':
    main()
