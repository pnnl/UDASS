import numpy as np
import os
import argparse
import sys
import yaml

import torch
from torchvision import datasets

from datasets.vqvae_sem import *
from models.vqvae_model import *
import utils.utils as utils
from utils.clf_utils import *

def main():

    ## Parse input from command prompt
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default = 0, help = 'GPU#')
    parser.add_argument('--dataset', type=str, default = '', help = 'Dataset')
    parser.add_argument('--train_file', type=str, default = '', help = 'Root directory of train dataset')
    parser.add_argument('--test_file', type=str, default = '', help = 'Root directory of val dataset')
    parser.add_argument('--config_file', type=str, default = 'sem_config.yaml', help = 'Params file')
    parser.add_argument('--exp_name', type=str, default = 'test', help = 'Name of current experiment')
    parser.add_argument('--mags', type=str, default = '', help = 'List of magnifications used to train vqvae')
    parser.add_argument('--stage', type=str, default='train', help = 'Model stage')
    parser.add_argument('--seed', type=int, default = 0, help = 'Seed for random split of train/val files')
    parser.add_argument('--style_config_file', type=str, default = '', help='Configs file for style model')
    parser.add_argument('--style_checkpoint_file', type=str, default = '', help='Checkpoint of style model to be restored')
    parser.add_argument('--checkpoint_dir', type=str, default = './checkpoints/vqvae/', help = 'Directory of stored checkpoint')
    parser.add_argument('--checkpoint', type=int, default = -1, help = 'Checkpoint to be restored')


    params, unparsed = parser.parse_known_args()
    ## Add seed info
    params.exp_name = add_seed_name(params.exp_name, params.seed)
    params.train_file = add_seed_name(params.train_file, params.seed)
    params.test_file = add_seed_name(params.test_file, params.seed)
    if params.stage == 'test':
        params.config_file = add_seed_name(params.config_file, params.seed)
        params.checkpoint_dir = add_seed_name(params.checkpoint_dir, params.seed)

    ## Load config file
    config_params = utils.load_config_file(params, 'vqvae')
    ## Load config file for style model if needed
    style_config_params = None
    if params.style_config_file != '':
        style_config_params = get_config(params.style_config_file)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu_id)

    ## Create logger
    logger_name = './results/logs/vqvae/test_' if params.stage=='test' else './results/logs/vqvae/'
    logger_name += '%s_%s.log' %(params.dataset,params.exp_name)
    logger = utils.create_logger(logger_name)
    logger.info('##### Params #####')
    logger.info('Train text file: {:}'.format(params.train_file))
    logger.info('Test text file: {:}'.format(params.test_file))
    logger.info('Config file name: {:}'.format(params.config_file))
    logger.info('Experiment name: {:}'.format(params.exp_name))
    logger.info('Magnifications: {:}'.format(params.mags))
    if params.stage == 'test':
        logger.info('Restore checkpoint: {:}'.format(params.checkpoint))
    logger.info('##########\n')

    vqvae_model = VQVAE_Model(params, config_params, style_config_params)

    ## Create dataset
    dataloader_dict = {}
    datatypes = ['train','test'] if params.stage == 'train' else ['test']

    new_size = (config_params['input_res'],config_params['input_res']) if 'input_res' in config_params else None
    for dt in datatypes:
        curr_file = params.train_file if dt == 'train' else params.test_file

        if params.dataset == 'sem':
            dataset = VQ_Dataset(curr_file, params.mags.split(',') if params.mags != '' else [], \
                                    dt, n_channel=config_params['input_dim'], new_size=new_size)
        else:
            print('No compatible dataset!')
            return

        dataloader_dict[dt] = torch.utils.data.DataLoader(dataset=dataset, num_workers=4, shuffle=dt=='train',\
                                batch_size=config_params['batch_size'] if dt == 'train' else 1)  \
                                if dt == 'train' or params.stage != 'train'\
                                else dataset

        logger.info("# of {:} files: {:}".format(dt, len(dataset)))
        print("Finished loading data!\n")


    if params.stage == 'train':
        vqvae_model.train(dataloader_dict, logger)
    elif params.stage == 'test':
        vqvae_model.inference(dataloader_dict['test'])


if __name__ == '__main__':
    main()
