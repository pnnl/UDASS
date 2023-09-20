import os
import logging
import yaml
import math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.utils as vutils


def log_params(logger, params):
    '''
    Log user params
    '''
    logger.info('##### Params #####')
    logger.info('Dataset: {:}'.format(params.dataset))
    logger.info('Train dataset directory: {:}'.format(params.train_dir))
    logger.info('Config file name: {:}'.format(params.config_file))
    logger.info('Experiment name: {:}'.format(params.exp_name))
    logger.info('Checkpoint directory: {:}'.format(params.checkpoint_dir))
    if params.stage == 'test':
        logger.info('Restore checkpoint: {:}'.format(params.checkpoint))
        logger.info('Directory to store generated image: {:}'.format(params.gen_dir))
        logger.info('# of generated styles: {:}'.format(params.num_styles))

    logger.info('##########\n')


def save_gen_imgs(x, out_name, isNorm=True, isGrayscale=False):
    '''
    Save generated images

    Args:        
        out_name - input's file name
        isNorm - normalized generated image
        isGrayscale - conver generated image to grayscale
    '''
    curr_sample = x
    if isNorm:
        curr_sample = x*0.5 + 0.5

    if isGrayscale:
        curr_sample = torch.mean(curr_sample, 1)
        curr_sample = curr_sample.repeat([1,3,1,1])

    curr_gen_img = curr_sample.squeeze(0).cpu().numpy().transpose(1,2,0)

    curr_img = np.clip(curr_gen_img*255, 0, 255).astype('uint8')
    img = Image.fromarray(curr_img, mode='RGB')

    img.save(out_name[:-4] + '.tif')


def get_model_list(dirname, key, checkpoint=-1):
    '''
    Get file name of trained model based on checkpoint

    Args:
        dirname - directory of trained models
        key - type of model (generator/discriminator) [gen|dis]
        checkpoint - checkpoint to restore
    '''
    ## Get model list for resume
    if os.path.exists(dirname) is False:
        return None

    if checkpoint != -1:
        return os.path.join(dirname, '%s_%08d.pt'%(key,checkpoint))

    else:
        gen_models = []
        for f in os.listdir(dirname):
            if key in f:
                gen_models.append(os.path.join(dirname, f))

        if gen_models is None:
            return None
        else:
            gen_models.sort()
            return gen_models[checkpoint]


def weights_init(init_type='gaussian'):
    '''
    An implementation of model weights initialization based off of
    https://github.com/NVlabs/MUNIT
    '''
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):            
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun
