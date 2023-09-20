import os
import logging
import yaml
import matplotlib.pyplot as plt
import shutil

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def load_config_file(params, type_exp):

    if params.stage == 'train':
        if not os.path.exists('./results/config_logs'):
            os.makedirs('./results/config_logs')
        if not os.path.exists('./results/config_logs/%s/' %type_exp):
            os.makedirs('./results/config_logs/%s/' %type_exp)

        shutil.copy(params.config_file, \
            os.path.join('./results/config_logs', type_exp, '%s_%s.yaml' %(params.dataset,params.exp_name)))
        shutil.copy('./models/%s_model.py' %type_exp, \
                os.path.join('./results/config_logs', type_exp, '%s_%s_model.py' %(params.dataset,params.exp_name)))
        shutil.copy('./datasets/%s_sem.py' %type_exp, \
                os.path.join('./results/config_logs', type_exp, '%s_%s_dataset.py' %(params.dataset,params.exp_name)))

    config_params = get_config(params.config_file)
    return config_params


def create_logger(logger_name):
    '''
    Create logger object
    '''

    if not os.path.exists('./results/logs'):
        os.makedirs('./results/logs')
    if not os.path.exists('./results/logs/%s' %logger_name.split('/')[-2]):
        os.makedirs('./results/logs/%s' %logger_name.split('/')[-2])

    if os.path.exists(logger_name):
        os.remove(logger_name)

    logger = logging.getLogger(logger_name.split('/')[-1][:-4])
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(logger_name); fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(console)

    return logger


def log_params(logger, params):
    '''
    Log user params
    '''

    logger.info('##### Params #####')
    logger.info('Dataset: {:}'.format(params.dataset))
    logger.info('Exp name: {:}'.format(params.exp_name))
    logger.info('Config file: {:}'.format(params.config_file))
    logger.info('Train file: {:}'.format(params.train_dir))
    logger.info('Val file: {:}'.format(params.val_dir))


def get_color_mean():
    return [0.485, 0.456, 0.406]


def get_color_std():
    return [0.229, 0.224, 0.225]


def convert_vendors_list(vendors):
    return [x for x in vendors.split(',')]


def set_grads(model, value, name=None):
    for pname, param in model.named_parameters():
        if name == None or name in pname:
            param.requires_grad = value


def convert_grayscale(inp):
    inp_new = torch.mean(inp, 1, keepdims=True)
    if inp.size(1) == 3:
        return inp_new.repeat(1,3,1,1)
    return inp_new


def unnormalize(inp):
    mu = get_color_mean(); std = get_color_std()
    for i in range(inp.size(1)):
        inp[:,i,...] = inp[:,i,...]*std[i] + mu[i]
    return inp
