import time
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cm
import os
import numpy as np
from PIL import Image
from scipy.stats import entropy

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
from models.miso_trainer import *
from models.vqvaeclf_trainer import *
from models.stylize_model import *
from utils.utils import *
from utils.clf_utils import *


class Clf_Model(object):
    '''
    The classification task class declaration

    Args:
        params - user's input parameters defined in terminal
        config_params - parameters defined in yaml config file
        saved_exp_name - name of current experiment
        style_config_params - parameters for the appearance transformation model defined in yaml config file
        classes - number of classes
        logger - logging object
        mags(optional-only applicable to the experimental dataset)
    '''
    def __init__(self, params, config_params, saved_exp_name, \
                style_config_params=None, classes=None, num_mags=1, logger=None):

        self.params = params
        self.saved_exp_name = saved_exp_name
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.num_mags = num_mags
        self.classes = classes
        num_classes = len(classes)
        self.isViz = False # Save org & transformed test images

        self.style_model = None
        if style_config_params is not None and ('app' in params.adapt_type):
            print('--- Initiate Gen Model ---')
            self.style_model = Stylize(style_config_params, params.style_checkpoint_file, device)
            self.style_config_params = style_config_params

        if 'vqvae_clf' in config_params['type_network']:
            self.trainer = Vqvae_Trainer(self.params, config_params, device, num_classes, \
                                        logger=logger, style_model=self.style_model)
        else:
            self.trainer = Miso_Trainer(self.params, config_params, device,\
                                        num_classes=num_classes, num_mags=num_mags, \
                                        logger=logger, style_model=self.style_model)


        if params.stage == 'train':
            if not os.path.exists('./results/checkpoints/clf'):
                os.makedirs('./results/checkpoints/clf')
            if not os.path.exists('./results/checkpoints/clf/%s' %self.saved_exp_name):
                os.makedirs('./results/checkpoints/clf/%s' %self.saved_exp_name)

            if not os.path.exists('./results/tensorboards/clf'):
                os.makedirs('./results/tensorboards/clf')
            tensorboard_dir = './results/tensorboards/clf/%s' %self.saved_exp_name
            self.writer = SummaryWriter(tensorboard_dir)

    def train(self, dataloader_dict, logger):
        self.trainer.train(dataloader_dict, logger, self.writer, self.saved_exp_name)


    def inference(self, dataloader_dict, classes, result_file):
        '''
        Inference stage

        Args:
            dataloader_dict - dataloader 
            classes - number of classes
            result_file - name of csv file for storing results when validating multiple models
        '''
        if not os.path.exists('./results/viz/clf'):
            os.makedirs('./results/viz/clf')
        viz_dir = '%s_%s_%s_%s' %(self.saved_exp_name,self.params.adapt_type,\
                self.params.test_dir.split('/')[-1].split('_')[1],self.params.test_dir.split('/')[-1].split('_')[2])
        if not os.path.exists('./results/viz/clf/%s' %viz_dir):
            os.makedirs('./results/viz/clf/%s' %viz_dir )

        ## Store results in a spreadsheet when running multiple exps
        saved_result_file = None
        if result_file != '' and not os.path.exists('./results/%s.csv' %result_file):
            saved_result_file = open('./results/%s.csv' %result_file,'w')
            saved_result_file.write('Exp Name, Acc.\n')
        elif result_file != '' and os.path.exists('./results/%s.csv' %result_file):
            saved_result_file = open('./results/%s.csv' %result_file,'a')

        self.trainer.restore_model(self.params.saved_model_dir, self.saved_exp_name)

        ## Type of adaptation 
        ## na - no adaptation; hm - histogram matching; wct - whitening & coloring transformation
        ## tent - model adaptation with entropy minimization (https://openreview.net/pdf?id=uXl3bZLkr3c)
        ## random_app - appearance transformation with random style vector
        ## app - appearance transformation by iteratively updating style vector
        ## app_tent - combining app & tent
        if self.params.adapt_type == 'na' or self.params.adapt_type == 'hm' or self.params.adapt_type == 'wct':
            total_preds, total_labels = self._no_adapt(dataloader_dict, viz_dir)

        elif 'tent' in self.params.adapt_type and 'app' not in self.params.adapt_type:
            total_preds, total_labels = self._tent(dataloader_dict, viz_dir)

        elif self.params.adapt_type == 'app_tent':
            total_preds, total_labels = self._app_tent(dataloader_dict, viz_dir)

        elif self.params.adapt_type == 'random_app' or self.params.adapt_type == 'app':
            total_preds, total_labels = self._app(dataloader_dict['test'], self.params.adapt_type, viz_dir)

        acc = np.sum(total_preds == total_labels) / len(total_preds)

        print('Validation Accuracy = {:.4f}'.format(acc))
        conf_matrix = cm(total_labels, total_preds, labels=np.arange(len(classes)))
        plot_confusion_matrix(conf_matrix, classes, \
            './results/viz/clf/%s/%s' %(viz_dir,self.saved_exp_name), normalize=True)

        if saved_result_file is not None:
            saved_result_file.write('%s,%.4f\n' %(self.saved_exp_name,acc))
            saved_result_file.close()


    def _no_adapt(self, dataloader_dict, viz_dir):
        '''
        Inference stage with No Adaptation
        '''
        total_labels = -1*torch.ones((len(dataloader_dict['test'].dataset),), dtype = torch.int64).to(self.trainer.device)
        total_preds = -1*torch.ones((len(dataloader_dict['test'].dataset),), dtype = torch.int64).to(self.trainer.device)

        for bi, batch in enumerate(dataloader_dict['test']):
            label_ten = batch['label'].to(self.trainer.device)
            inp_ten, aug_inp_ten = self._rearrange_inp(batch)

            ## Visualize histogram matching or wct
            if False:
                for iii in range(inp_ten[0].size(0)):
                    viz = torch.cat( (inp_ten[0][iii:iii+1,...],inp_ten[1][iii:iii+1,...]), 0)
                    for im in inp_ten[2:]:
                        viz = torch.cat( (viz,im[iii:iii+1,...]), 0 )

                    viz = vutils.make_grid(viz, nrow=4)
                    viz = viz.cpu().numpy().transpose(1,2,0)
                    im = Image.fromarray( (viz*255).astype('uint8'), mode='RGB')
                    mag = '_%s' %batch['name'][0][iii].split('/')[-1].split('_')[1]
                    curr_name = batch['name'][0][iii].split('/')[-1].replace(mag,'')[:-4]
                    im.save('./results/viz/clf/%s/%s_viz.png' %(viz_dir,curr_name))

            with torch.set_grad_enabled(False):
                outputs = self.trainer.predict(aug_inp_ten)
                _, preds = torch.max(outputs, 1)

                curr_ent = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)

                start_ind = bi*self.trainer.config_params['bz']
                end_ind = bi*self.trainer.config_params['bz']+inp_ten[0].size(0)
                total_labels[start_ind:end_ind] = label_ten
                total_preds[start_ind:end_ind] = preds

                if self.isViz:
                    self._viz_pred(aug_inp_ten, viz_dir, batch['name'], aug_inp_ten)

        return total_preds.cpu().numpy(), total_labels.cpu().numpy()


    def _tent(self, dataloader_dict, viz_dir):
        '''
        Inference stage with model adaptation using entropy minimization
        '''
        bn_params = self.trainer.extract_bn()
        opt = torch.optim.Adam(bn_params, lr=1e-4)

        total_labels = -1*torch.ones((len(dataloader_dict['test'].dataset),), dtype = torch.int64).to(self.trainer.device)
        total_preds = -1*torch.ones((len(dataloader_dict['test'].dataset),), dtype = torch.int64).to(self.trainer.device)

        ## Load training target data for model adaptation
        num_ep = 1
        dl = 'target_train'
        for ep in range(num_ep):
            epoch_loss = 0.0
            for bi, batch in enumerate(dataloader_dict[dl]):
                label_ten = batch['label'].to(self.trainer.device)

                inp_ten, aug_inp_ten = self._rearrange_inp(batch)

                with torch.set_grad_enabled(True):
                    opt.zero_grad()
                    outputs = self.trainer.predict(aug_inp_ten, isTrainMode=True)

                    _, preds = torch.max(outputs.detach(), 1)

                    loss = self._compute_ent(outputs)
                    loss.backward()
                    opt.step()

                    epoch_loss += loss.item()

                    ## Restore original weights before the adaptation
                    if False:
                        self.trainer.model_ftrs.load_state_dict(torch.load('%s/%s/%s_ftrs.pt' %(self.params.saved_model_dir,\
                                                            self.saved_exp_name,self.saved_exp_name)))
                        self.trainer.model_clf.load_state_dict(torch.load('%s/%s/%s_clf.pt' %(self.params.saved_model_dir,\
                                                        self.saved_exp_name,self.saved_exp_name)))
            epoch_loss /= len(dataloader_dict[dl])

        ## Use the adapted model for inference
        total_preds, total_labels = self._no_adapt(dataloader_dict, viz_dir)
        return total_preds, total_labels


    def _app_tent(self, dataloader_dict, viz_dir):
        '''
        Inference stage with apperance transformation + entropy minimization
        '''
        bn_params = self.trainer.extract_bn()
        opt = torch.optim.Adam(bn_params, lr=1e-4)

        num_ep = 1
        dl = 'target_train'

        for ep in range(num_ep):
            epoch_loss = 0.0
            for bi, batch in enumerate(dataloader_dict[dl]):
                with torch.set_grad_enabled(True):
                    opt.zero_grad()

                    inp_ten = batch['inp']
                    style_vec = []; content_vec = []
                    for it in range(len(inp_ten)):
                        curr_ref_content, curr_ref_style = self.style_model.get_embeddings(inp_ten[it].to(self.trainer.device),\
                                                                                        True)
                        content_vec.append(curr_ref_content)

                        s = Variable(curr_ref_style, requires_grad=True)
                        style_vec.append(s)

                    s_opt = torch.optim.Adam(style_vec, lr=1e-1)

                    ## Transform input images
                    for iii in range(1):
                        s_opt.zero_grad()
                        new_aug_img_ten = []
                        for c, s in zip(content_vec, style_vec):
                            new_aug_img_ten.append(self.style_model.stylize(c, s,\
                                            isGrayscale=True,isNorm=self.trainer.isNorm))

                        outputs = self.trainer.predict(new_aug_img_ten, isTrainMode=False)
                        s_loss = self._compute_ent(outputs)
                        s_loss.backward()
                        s_opt.step()

                    new_aug_img_ten = []
                    for c, s in zip(content_vec, style_vec):
                        new_aug_img_ten.append(self.style_model.stylize(c, s,\
                                    isGrayscale=True,isNorm=self.trainer.isNorm))

                    ## Use transformed input to update BN layers
                    outputs = self.trainer.predict(new_aug_img_ten, isTrainMode=True)
                    loss = self._compute_ent(outputs)
                    loss.backward()
                    opt.step()

                    epoch_loss += loss.item()
            epoch_loss /= len(dataloader_dict[dl])

        ## Use the adapted model along with transformed target data for inference
        total_preds, total_labels = self._app(dataloader_dict['test'], 'app', viz_dir)

        return total_preds, total_labels


    def _app(self, dataloader, adapt_type, viz_dir):
        '''
        Inference stage with appearance transformation
        '''
        num_styles = 1
        total_labels = -1*torch.ones((len(dataloader.dataset),), dtype = torch.int64).to(self.trainer.device)
        total_preds = -1*torch.ones((len(dataloader.dataset),), dtype = torch.int64).to(self.trainer.device)

        for bi, batch in enumerate(dataloader):
            label_ten = batch['label'].to(self.trainer.device)

            with torch.set_grad_enabled(adapt_type != 'random_app'):

                inp_ten = batch['inp']; org_inp = []
                style_vec = []; content_vec = []
                for it in range(len(inp_ten)):
                    org_inp.append(inp_ten[it].to(self.trainer.device))
                    curr_ref_content, curr_ref_style = self.style_model.get_embeddings(org_inp[it], True)
                    content_vec.append(curr_ref_content)

                    if adapt_type == 'random_app':
                        s = None
                    else:
                        s = Variable(curr_ref_style, requires_grad=True)
                    style_vec.append(s)

                ## Optimize the appearance representation
                if adapt_type != 'random_app':
                    opt = torch.optim.Adam(style_vec, lr=1e-1)

                    for iii in range(1):
                        opt.zero_grad()
                        new_aug_img_ten = []
                        for c, s in zip(content_vec, style_vec):
                            new_aug_img_ten.append(self.style_model.stylize(c, s,\
                                            isGrayscale=True,isNorm=self.trainer.isNorm))

                        outputs = self.trainer.predict(new_aug_img_ten, isTrainMode=False)
                        loss = self._compute_ent(outputs)
                        loss.backward()
                        opt.step()

                new_aug_img_ten = []
                for c, s in zip(content_vec, style_vec):
                    new_aug_img_ten.append(self.style_model.stylize(c, s,\
                                isGrayscale=True,isNorm=self.trainer.isNorm).detach())

                outputs = self.trainer.predict(new_aug_img_ten, isTrainMode=False)
                _, preds = torch.max(outputs, 1)

                if self.isViz:
                    self._viz_pred(org_inp, viz_dir, batch['name'], new_aug_img_ten)

                start_ind = bi*self.trainer.config_params['bz']
                end_ind = bi*self.trainer.config_params['bz']+inp_ten[0].size(0)
                total_labels[start_ind:end_ind] = label_ten
                total_preds[start_ind:end_ind] = preds

        return total_preds.cpu().numpy(), total_labels.cpu().numpy()


    def _rearrange_inp(self, batch):
        ''' 
        Rearrange batch input into a list in which
        each index corresponds to a tensor of input at a certain magnification
        '''
        inp_ten = [batch['inp'][0].to(self.trainer.device)]
        aug_inp_ten = []
        if self.trainer.isNorm:
            aug_inp_ten.append(TF.normalize(inp_ten[-1], mean=self.trainer.norm_mean, std=self.trainer.norm_std))
        else:
            aug_inp_ten.append(inp_ten[-1])

        ## Get input from the other magnifications
        for ii in range(1, len(batch['inp'])):
            inp_ten.append(batch['inp'][ii].to(self.trainer.device))
            if self.trainer.isNorm:
                aug_inp_ten.append(TF.normalize(inp_ten[-1], mean=self.trainer.norm_mean, std=self.trainer.norm_std))
            else:
                aug_inp_ten.append(inp_ten[-1])

        return inp_ten, aug_inp_ten


    def _compute_ent(self, outputs):
        '''
        Compute entropy
        '''
        if outputs.size(1) > 1:
            ent = (-(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)).mean()

            div1 = outputs.softmax(1).mean(0,keepdim=True)
            div = (div1 * div1.log()).sum(1)

            return ent# + div
        else:
            m = outputs.squeeze(1)
            return (-m.sigmoid()*m.sigmoid().log()).mean()


    def _viz_pred(self, img, viz_dir, name, transf_img):
        '''
        Visualize transformed input
        '''
        for i in range(len(img[0])):
            curr_img = []; curr_transf = []
            if len(img) > 1:
                mag = '_%s' %name[0][i].split('/')[-1].split('_')[1]
                curr_name = name[0][i].split('/')[-1].replace(mag,'')[:-4]
            else:
                curr_name = name[0][i].split('/')[-1][:-4]

            for j in range(len(img)):

                img_j = img[j][i:i+1,...]
                timg_j = transf_img[j][i:i+1,...]

                x = img_j.size(2) if j == 0 else curr_img[0].size(2)
                y = img_j.size(3) if j == 0 else curr_img[0].size(3)

                img_j = F.interpolate(img_j, size=(x,y), \
                            mode='bilinear', align_corners=False)
                timg_j = F.interpolate(timg_j, size=(x,y), \
                            mode='bilinear', align_corners=False)

                if img_j.size(1) == 1:
                    img_j = img_j.repeat(1,3,1,1)
                if timg_j.size(1) == 1:
                    timg_j = timg_j.repeat(1,3,1,1)

                if self.trainer.isNorm:
                    timg_j = unnormalize(timg_j)

                curr_img.append(img_j)
                curr_transf.append(timg_j)

            viz = torch.cat(curr_img+curr_transf, 0)

            viz = vutils.make_grid(viz, nrow=viz.size(0)//2)*255.0
            viz = viz.squeeze().detach().cpu().numpy().transpose(1,2,0)

            im = Image.fromarray(viz.astype('uint8'), mode='RGB')
            im.save('./results/viz/clf/%s/%s_viz.png' %(viz_dir,curr_name))
