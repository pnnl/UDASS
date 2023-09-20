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

from networks.vqvae_clf import *
from models.stylize_model import *
from networks.vqvae import *
from utils.utils import *
from utils.clf_utils import *
from utils.style_utils import *

class Vqvae_Trainer(object):
    '''
    The classfication model for VQVAE class declaration

    Args:
        params - user's input parameters defined in terminal
        config_params - parameters defined in yaml config file
        device - type of device used for running the model
        style_model - apperance transformation model
        num_classes - number of classes        
    '''
    def __init__(self, params, config_params, device, num_classes=None, logger=None, style_model=None):

        self.config_params = config_params
        self.device = device
        self.isNorm = self.config_params['isNorm']
        self.norm_mean = get_color_mean()
        self.norm_std = get_color_std()

        ## Load VQVAE model
        print('---Initiate VQVAE---')
        vqvae_config_params = get_config(params.vqvae_config_file)
        encoders = [QuarterEncoder(vqvae_config_params['input_dim'], vqvae_config_params['latent_size'], vqvae_config_params['latent_count']), \
                    HalfEncoder(vqvae_config_params['latent_size'], vqvae_config_params['latent_size'], vqvae_config_params['latent_count'])]
        decoders = [HalfDecoder(vqvae_config_params['latent_size'], vqvae_config_params['latent_size']), \
                    HalfQuarterDecoder(vqvae_config_params['latent_size'], vqvae_config_params['input_dim'])]
        self.vqvae_model = VQVAE(encoders, decoders).to(self.device)
        ## Load pretrained weights
        last_model_name = get_model_list(params.vqvae_checkpoint_dir,"vqvae", -1)
        print('Load VQVAE model with: %s' %last_model_name)

        state_dict = torch.load(last_model_name)
        self.vqvae_model.load_state_dict(state_dict)
        set_grads(self.vqvae_model, False, name='dictionary')
        set_grads(self.vqvae_model, False, name='decoder')
        self.vqvae_model.eval()
        self.latent_count = vqvae_config_params['latent_count']

        self.style_model = style_model

        self.criterion = nn.CrossEntropyLoss()

        in_dim = vqvae_config_params['latent_size']
        ## Specify the type of input used for the classifier (vectors from codebook or indicies of codebook)
        self.isEmbedded = False
        if config_params['type_network'] != 'vqvae_clf1':
            in_dim *= 2
            self.isEmbedded = True

        self.model_clf = VQVAE_Clf(num_classes, in_dim, \
                                    mid_dim=vqvae_config_params['latent_size']*4, \
                                    model_type=config_params['type_network']).to(self.device)

        if params.stage == 'train':
            logger.info("Model: {:}".format(self.model_clf))

        self.optimizer = optim.Adam(self.model_clf.parameters(), lr=self.config_params['lr'])


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
                    self.model_clf.train()
                else:
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

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        indices = self.vqvae_model.get_indices(aug_inp_ten[0][:,0:1,...],self.latent_count,self.isEmbedded)

                        outputs = self.model_clf(indices)
                        loss = self.criterion(outputs, label_ten)

                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss * aug_inp_ten[-1].size(0)
                    running_acc += torch.sum(preds == label_ten)

                num_samples = len(dataloader_dict[phase].dataset)
                epoch_loss = running_loss.item() / num_samples
                epoch_acc = running_acc.item() / num_samples

                curr_lr = self.optimizer.param_groups[0]['lr']

                writer.add_scalar('%s/Loss' %phase, epoch_loss, epoch)
                writer.add_scalar('%s/Acc' %phase, epoch_acc, epoch)

                if epoch % 5 == 0:
                    org_img_grid = torch.cat(inp_ten, 0)
                    aug_img_grid = torch.cat(aug_inp_ten, 0)
                    if self.isNorm:
                        aug_img_grid = unnormalize(aug_img_grid)
                    _, real_recons, _, _ = [x for x in self.vqvae_model.full_reconstructions(aug_inp_ten[0][:,0:1,...])]
                    real_recons = torch.clamp(real_recons, 0, 1)

                    viz_grid = vutils.make_grid(torch.cat((org_img_grid,aug_img_grid,real_recons.repeat(1,3,1,1)),0), nrow=org_img_grid.size(0))
                    writer.add_image('Inp', viz_grid, epoch)

                curr_info = '{:} Phase -> Loss - {:.4f}; Acc - {:.4f}'.format(phase,epoch_loss,epoch_acc)
                if phase == 'val':
                    curr_info += '\n'
                    val_acc_vec.append(epoch_acc)
                    if epoch_acc > best_acc or epoch == 0:
                        best_acc = epoch_acc
                        best_ep = epoch

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
            self.model_clf.train()
        else:
            self.model_clf.eval()

        indices = self.vqvae_model.get_indices(inp_ten[0][:,0:1,...],self.latent_count,self.isEmbedded)
        outputs = self.model_clf(indices)

        return outputs


    def restore_model(self, model_dir, exp_name):
        self.model_clf.load_state_dict(torch.load('%s/%s/%s_clf.pt' %(model_dir,\
                                                    exp_name,exp_name)))


    def _saved_models(self, model, model_name):
        best_model = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), model_name)

        return best_model


    def extract_bn(self):
        params = []; names = []

        for net in [self.model_clf]:
            for nm, m in net.named_modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):                    
                    for nmp, p in m.named_parameters():
                        if nmp in ['weight', 'bias']:  # weight is scale, bias is shift
                            params.append(p)
                            names.append(f"{nm}.{nmp}")

        return params
