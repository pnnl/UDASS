import argparse

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from datasets.clf_sem import *
from models.clf_model import *

from utils.utils import *
from utils.clf_utils import *

TASK_TYPE = {'sem':'clf'}

def main():
    ## Parse input from command prompt
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default = "0", help = 'GPU#')
    parser.add_argument('--exp_name', type=str, default = 'temp', help = 'Name of current experiment')
    parser.add_argument('--dataset', type=str, default = '', help = 'Name of dataset [sem]')
    parser.add_argument('--train_dir', type=str, default = '', help = 'Root directory of training source dataset')
    parser.add_argument('--val_dir', type=str, default = '', help = 'Root directory of val source dataset')
    parser.add_argument('--target_train_dir', type=str, default = '', help = 'Root directory of training target dataset')
    parser.add_argument('--test_dir', type=str, default = '', help = 'Root directory of test dataset')
    parser.add_argument('--config_file', type=str, default = 'sem_config.yaml', help = 'Params file')
    parser.add_argument('--seed', type=int, default = 0, help = 'Seed for random split of train/val files (0-9)')

    parser.add_argument('--stage', type=str, default = 'train', help='Specify the process [train/test]')
    parser.add_argument('--adapt_type', type=str, default = 'na', help='Adaptation method during inference [na|tent|hm|wct|random_app|app|app_tent]')
    parser.add_argument('--ref_stats', type=str, default = '', help='Reference CDF for HM')
    parser.add_argument('--saved_model_dir', type=str, help='Directory of task model\'s checkpoint')
    parser.add_argument('--style_config_file', type=str, default = '', help='Configs file for appearance transformation model')
    parser.add_argument('--style_checkpoint_file', type=str, default = '', help='Checkpoint of appearance transformation to be restored')
    parser.add_argument('--vqvae_config_file', type=str, default = '', help='Configs file for vqvae model')
    parser.add_argument('--vqvae_checkpoint_dir', type=str, default = '', help='Checkpoint of vqvae model to be restored')
    parser.add_argument('--result_file', type=str, default = '', help='Store results of multiple experiments in a spreadsheet')

    params, unparsed = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= params.gpu_id
    ## Add seed info
    params.exp_name = add_seed_name(params.exp_name, params.seed)
    params.train_dir = add_seed_name(params.train_dir, params.seed)
    params.val_dir = add_seed_name(params.val_dir, params.seed)
    params.test_dir = add_seed_name(params.test_dir, params.seed)
    params.target_train_dir = add_seed_name(params.target_train_dir, params.seed)
    params.ref_stats = add_seed_name(params.ref_stats, params.seed)
    if params.vqvae_config_file != '':
        params.vqvae_config_file = add_seed_name(params.vqvae_config_file, params.seed)
        params.vqvae_checkpoint_dir = add_seed_name(params.vqvae_checkpoint_dir, params.seed)
    if params.stage == 'test':
        params.config_file = add_seed_name(params.config_file, params.seed)

    curr_task = TASK_TYPE[params.dataset]
    ## Load config file
    config_params = load_config_file(params, curr_task)

    ## Load config file for style model if needed
    style_config_params = None
    if params.style_config_file != '':
        style_config_params = get_config(params.style_config_file)

    ## Create logger
    saved_exp_name = '%s_%s' %(params.dataset,params.exp_name)
    if params.stage == 'train':
        logger_name = './results/logs/%s/%s.log' %(curr_task, saved_exp_name)
        logger = create_logger(logger_name)
        log_params(logger, params)
    else:
        logger = None
        print('\nSet-up inference for: %s' %params.exp_name)
        print('Restore config file: %s' %params.config_file)
        print('Restore style config file: %s' %params.style_config_file)
        print('Restore style checkpoint dir: %s' %params.style_checkpoint_file)
        print('Restore vqvae config file: %s' %params.vqvae_config_file)
        print('Restore vqvae checkpoint dir: %s' %params.vqvae_checkpoint_dir)
        print('Restore ref statistics file: %s\n' %params.ref_stats)

    ## Load data
    if params.stage == 'train' and params.target_train_dir == '':
        datatype_list = ['train','val']
    elif params.stage == 'test' and ('tent' in params.adapt_type or 'app_tent' == params.adapt_type):
        datatype_list = ['target_train','test']
    elif params.stage == 'test':
        datatype_list = ['test']

    dataloader_dict = {}
    for x in datatype_list:
        if x == 'train':
            curr_rdir = params.train_dir
        elif 'val' in x:
            curr_rdir = params.val_dir
        elif 'test' in x:
            curr_rdir = params.test_dir
        elif 'target_train' in x:
            curr_rdir = params.target_train_dir

        curr_dataset = ClfSEM_Dataset(curr_rdir, x, params.adapt_type, \
                                    params.ref_stats, config_params['type_network'])
        classes = curr_dataset.classes

        if params.stage == 'train':
            logger.info('Dir of {:} dataset: {:}'.format(x, curr_rdir))
            logger.info('# of images in {:} dataset: {:}'.format(x,len(curr_dataset)))
        else:
            print('Dir of {:} dataset: {:}'.format(x, curr_rdir))
            print('# of images in {:} dataset: {:}'.format(x,len(curr_dataset)))
            if 'test' in x:
                print('Adapt Type: {:}'.format(params.adapt_type))

        if x == 'test' and params.adapt_type == 'random_app':
            config_params['bz'] = 1
        else:
            ## Find a batch size that doesn't contain a single sample in a batch
            while len(curr_dataset) - (len(curr_dataset) // config_params['bz'])*config_params['bz'] == 1:
                config_params['bz'] -= 1

        curr_dataloader = DataLoader(curr_dataset, batch_size=config_params['bz'], \
                                            shuffle=True, num_workers=4, pin_memory=True)
        dataloader_dict[x] = curr_dataloader

    ## Initiate model
    model = Clf_Model(params, config_params, saved_exp_name, style_config_params, \
                        classes, curr_dataset.num_mags, logger=logger)


    ## Execute the task model
    if params.stage == 'train':
        model.train(dataloader_dict, logger)
    elif params.stage == 'test':
        model.inference(dataloader_dict, classes, params.result_file)


if __name__ == '__main__':
    main()
