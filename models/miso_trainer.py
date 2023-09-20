import time
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np

import torchvision
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import torchvision.models as models

from networks.miso import *
from utils.utils import *
from utils.clf_utils import *


class Miso_Trainer(object):
    '''
    The MISO model class declaration

    Args:
        params - user's input parameters defined in terminal
        config_params - parameters defined in yaml config file
        device - type of device used for running the model
        style_model - apperance transformation model
        num_classes - number of classes
        logger - logging object
        num_mags(optional-only applicable to the experimental dataset)
    '''
    def __init__(self, params, config_params, device,\
                num_classes=None, num_mags=1, logger=None, style_model=None):

        self.config_params = config_params
        self.device = device
        self.isNorm = self.config_params['isNorm']
        self.norm_mean = get_color_mean()
        self.norm_std = get_color_std()

        self.criterion = nn.CrossEntropyLoss()

        if self.config_params['type_network'] == 'unshared':
            self.model_ftrs = Ftrs_UnSharedNet(num_mags, num_classes, self.config_params['resnet']).to(self.device)
        elif self.config_params['type_network'] == 'shared':
            self.model_ftrs = Ftrs_SharedNet(num_mags, num_classes, self.config_params['resnet']).to(self.device)
        self.model_clf = MLP_Clf(num_mags, num_classes, self.config_params['resnet']).to(self.device)

        ## Load pretrained MISO
        if params.stage == 'train' and self.config_params['miso_pretrained']:            
            saved_model_name = self.config_params['pretrained_file']
            logger.info('Restore pretrained MISO file: {}'.format(saved_model_name))

            self.model_ftrs.load_state_dict(torch.load(saved_model_name))

            pretrained_dict = torch.load(saved_model_name.replace("ftrs","clf"))
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc_out' not in k}
            self.model_clf.load_state_dict(pretrained_dict, strict=False)

        params_ftrs, params_fcbn_ftrs = self._get_network_params(self.model_ftrs)
        params_clf, params_fcbn_clf = self._get_network_params(self.model_clf)

        params_net = params_ftrs + params_clf
        params_fcbn = params_fcbn_ftrs + params_fcbn_clf

        self.optimizer1 = optim.SGD([{'params':params_fcbn, 'lr':self.config_params['lr']*10}], momentum=0.9)
        self.optimizer2 = optim.SGD([{'params': params_net, 'lr':self.config_params['lr']},
                                 {'params': params_fcbn, 'lr':self.config_params['lr']}], momentum=0.9)

        self.style_model = style_model


    def _get_network_params(self, net):
        '''
        Split up network params
        '''
        params = []; params_fcbn = []
        for name, param in net.named_parameters():
            if 'fc' in name or 'BN' in name:
                params_fcbn.append(param)
            else:
                params.append(param)
        return params, params_fcbn


    def train(self, dataloader_dict, logger, writer, saved_exp_name):
        '''
        Training stage

        Args:
            dataloader_dict - dataloader
            logger - logging object
            writer - tensorboard object
            saved_exp_name - current experiment name
        '''
        since = time.time()

        best_acc = 0.0; best_ep = -1
        num_epochs = self.config_params['num_epochs']
        val_acc_vec = []

        trainingPhase = ['train','val']
        for epoch in range(num_epochs):
            logger.info('-' * 10)
            logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

            for phase in trainingPhase:
                running_loss = 0.0; running_acc = 0.0
                if phase == 'train':
                    self.model_ftrs.train()
                    self.model_clf.train()
                else:
                    self.model_ftrs.eval()
                    self.model_clf.eval()

                for bi, batch in enumerate(dataloader_dict[phase]):
                    label_ten = batch['label'].to(self.device)

                    inp_ten = [batch['inp'][0].to(self.device)]
                    aug_inp_ten = []
                    if self.isNorm:
                        aug_inp_ten.append(TF.normalize(inp_ten[-1], mean=self.norm_mean, std=self.norm_std))
                    else:
                        aug_inp_ten.append(inp_ten[-1])
                    ## Get input from multiple magnifications
                    ## mag x image
                    for ii in range(1, len(batch['inp'])):
                        inp_ten.append(batch['inp'][ii].to(self.device))
                        if self.isNorm:
                            aug_inp_ten.append(TF.normalize(inp_ten[-1], mean=self.norm_mean, std=self.norm_std))
                        else:
                            aug_inp_ten.append(inp_ten[-1])

                    if epoch < self.config_params['ep_init']:
                        self.optimizer1.zero_grad()
                    else:
                        self.optimizer2.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        ftrs = self.model_ftrs(aug_inp_ten)
                        outputs = self.model_clf(ftrs)
                        loss = self.criterion(outputs, label_ten)

                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            if epoch < self.config_params['ep_init']:
                                self.optimizer1.step()
                            else:
                                self.optimizer2.step()

                    running_loss += loss * aug_inp_ten[-1].size(0)
                    running_acc += torch.sum(preds == label_ten)

                num_samples = len(dataloader_dict[phase].dataset)                
                epoch_loss = running_loss.item() / num_samples
                epoch_acc = running_acc.item() / num_samples

                if epoch < self.config_params['ep_init']:
                    curr_lr = self.optimizer1.param_groups[0]['lr']
                else:
                    curr_lr = self.optimizer2.param_groups[0]['lr']

                writer.add_scalar('%s/Loss' %phase, epoch_loss, epoch)
                writer.add_scalar('%s/Acc' %phase, epoch_acc, epoch)

                if epoch % 5 == 0:
                    org_img_grid = torch.cat(inp_ten, 0)
                    aug_img_grid = torch.cat(aug_inp_ten, 0)
                    aug_img_grid = unnormalize(aug_img_grid)
                    viz_grid = vutils.make_grid(torch.cat((org_img_grid,aug_img_grid),0), nrow=org_img_grid.size(0))
                    writer.add_image('Inp', viz_grid, epoch)

                curr_info = '{:} Phase -> Loss - {:.4f}; Acc - {:.4f}'.format(phase,epoch_loss,epoch_acc)
                if phase == 'val':
                    curr_info += '\n'
                    val_acc_vec.append(epoch_acc)
                    if epoch_acc > best_acc or epoch == 0:
                        best_acc = epoch_acc
                        best_ep = epoch

                        best_model_ftrs = self._saved_models(self.model_ftrs, './results/checkpoints/clf/%s/%s_ftrs.pt' \
                                                            %(saved_exp_name,saved_exp_name))
                        best_model_clf = self._saved_models(self.model_clf, './results/checkpoints/clf/%s/%s_clf.pt' \
                                                            %(saved_exp_name,saved_exp_name))

                logger.info(curr_info)

        time_elapsed = time.time() - since
        logger.info('\nTraining completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logger.info('Best epoch saved: {:}'.format(best_ep))
        logger.info('Best epoch acc: {:.5f}'.format(val_acc_vec[best_ep]))

        writer.close()


    def predict(self, inp_ten, isTrainMode=False):
        '''
        Perform prediction
        '''
        if isTrainMode:
            self.model_ftrs.train()
            self.model_clf.train()
        else:
            self.model_ftrs.eval()
            self.model_clf.eval()

        ftrs = self.model_ftrs(inp_ten)
        outputs = self.model_clf(ftrs)

        return outputs


    def restore_model(self, model_dir, exp_name):
        '''
        Load pretrained weights
        '''
        self.model_ftrs.load_state_dict(torch.load('%s/%s/%s_ftrs.pt' %(model_dir,\
                                                    exp_name,exp_name)))
        self.model_clf.load_state_dict(torch.load('%s/%s/%s_clf.pt' %(model_dir,\
                                                    exp_name,exp_name)))


    def _saved_models(self, model, model_name):
        '''
        Save model weights
        '''

        best_model = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), model_name)

        return best_model


    def extract_bn(self):
        '''
        Extract model params
        '''
        params = []; names = []

        for net in [self.model_ftrs, self.model_clf]:
            for nm, m in net.named_modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    for nmp, p in m.named_parameters():
                        if nmp in ['weight', 'bias']:  # weight is scale, bias is shift
                            params.append(p)
                            names.append(f"{nm}.{nmp}")

        return params
