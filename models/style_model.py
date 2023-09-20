import os
import time
from skimage.exposure import histogram
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import copy

from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import torchvision.transforms.functional as TF

from networks.style_networks import *
from utils.style_utils import *
from utils.utils import *

class Style_Model(object):
    '''
    The appearance transformation model class declaration

    Args:
        params - user's input parameters defined in terminal
        config_params - parameters defined in yaml config file        
    '''
    def __init__(self, params, config_params):

        self.params = params
        self.config_params = config_params

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.isGrayscale = True # Change this when running a different dataset than SEM images

        ## Initiate the networks
        self.gen = AdaINGen(config_params['input_dim'], config_params['gen']).to(self.device)
        self.dis = MsImageDis(config_params['input_dim'], config_params['input_res'], \
                            config_params['dis'], self.device).to(self.device)

        self.vgg = Vgg16(extracted_layers=['c21r']).to(self.device)
        set_grads(self.vgg, False)

        self.style_dim = config_params['gen']['style_dim']

        ## Fix style noise
        self.s = torch.randn( (1, self.style_dim, 1, 1), device=self.device )

        ## Setup the optimizers
        beta1 = config_params['beta1']; beta2 = config_params['beta2']
        dis_params = list(self.dis.parameters()); gen_params = list(self.gen.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=config_params['lr'], betas=(beta1, beta2), weight_decay=config_params['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=config_params['lr'], betas=(beta1, beta2), weight_decay=config_params['weight_decay'])
        self.dis_scheduler = lr_scheduler.StepLR(self.dis_opt, step_size=config_params['step_size'],
                                                gamma=config_params['gamma'], last_epoch=-1)
        self.gen_scheduler = lr_scheduler.StepLR(self.gen_opt, step_size=config_params['step_size'],
                                                gamma=config_params['gamma'], last_epoch=-1)

        ## Network weight initialization
        self.gen.apply(weights_init(config_params['init']))
        self.dis.apply(weights_init('gaussian'))

        self.l1_criterion = nn.L1Loss()
        self.mse_criterion = nn.MSELoss()

        if not os.path.exists('./results/tensorboards/style'):
            os.makedirs('./results/tensorboards/style')
        if not os.path.exists('./results/checkpoints/style'):
            os.makedirs('./results/checkpoints/style')

        if params.stage != 'test':
            self.save_checkpoint_dir = './results/checkpoints/style/%s_%s' %(params.dataset,params.exp_name)
            if not os.path.exists(self.save_checkpoint_dir):
                os.makedirs(self.save_checkpoint_dir)

            if params.stage == 'train':
                self.curr_writer = SummaryWriter('./results/tensorboards/style/%s_%s' \
                                            %(params.dataset,params.exp_name))


    def train(self, dataloader_dict, logger):
        '''
        Training stage
        '''
        start_time = time.time()
        logger.info("Gen Architecture: {:}".format(self.gen))
        logger.info("Dis Architecture: {:}".format(self.dis))
        self.gen.train(); self.dis.train()

        metrics = {'Dis': 0.0, 'Gen/Overall': 0.0, \
                    'Gen/Adv': 0.0, 'Gen/X_recon': 0.0, 'Gen/c_recon': 0.0, \
                    'Gen/s_recon': 0.0, 'Gen/percep_recon': 0.0, \
                    'Gen/mi_loss': 0.0}

        itr = 0
        while True:
            for batch in dataloader_dict['train']:
                if itr >= self.config_params['max_iter']+1:
                    self.curr_writer.close()
                    logger.info("Finished training in %.4f min" %( (time.time()-start_time)/60 ))
                    return

                inp_tensor = batch[0].to(self.device)
                self.dis_update(inp_tensor, metrics)
                self.gen_update(inp_tensor, metrics, itr)

                if self.dis_scheduler is not None:
                    self.dis_scheduler.step()
                if self.gen_scheduler is not None:
                    self.gen_scheduler.step()

                if itr % self.config_params['tensorboard_itr'] == 0:
                    for item in metrics:
                        self.curr_writer.add_scalar(item, metrics[item], itr)

                    logger.info("Itr# {:} -> Loss - D = {:.4f} - G = {:.4f}"\
                            .format(itr, metrics['Dis'], metrics['Gen/Overall']))

                if itr % self.config_params['save_itr'] == 0 and itr != 0:
                    num_styles = 2
                    train_recon_error = self.visualize_img(inp_tensor, batch[1], num_styles, 'train', itr)

                    ridx = np.random.randint(0,len(dataloader_dict['test']))
                    test_tensor = dataloader_dict['test'][ridx][0].to(self.device).unsqueeze(0)
                    test_img_name = dataloader_dict['test'][ridx][1]
                    test_recon_error = self.visualize_img(test_tensor, test_img_name, num_styles, 'test', itr)

                    logger.info("Train Recon Error = {:.5f} - Test Recon Error = {:.5f}".format(train_recon_error,test_recon_error))

                    self.save_checkpoint(self.save_checkpoint_dir, itr)


                itr += 1        


    def inference(self, dataloader, params):
        '''
        Inference stage
        '''
        self.load_checkpoint(params)

        if not os.path.exists(params.gen_dir):
            os.makedirs(params.gen_dir)
        if not os.path.exists('%s/%s/' %(params.gen_dir,params.exp_name)):
            os.makedirs('%s/%s/' %(params.gen_dir,params.exp_name))
        if not os.path.exists('%s/recon_%s/' %(params.gen_dir,params.exp_name)):
            os.makedirs('%s/recon_%s/' %(params.gen_dir,params.exp_name))
        if not os.path.exists('%s/viz_%s/' %(params.gen_dir,params.exp_name)):
            os.makedirs('%s/viz_%s/' %(params.gen_dir,params.exp_name))

        with torch.set_grad_enabled(False):
            for bi, batch in enumerate(dataloader):
                if bi >= 20:
                    return
                inp_tensor = batch[0].to(self.device)

                curr_name = batch[1][0].split('/')[-1][:-4] if isinstance(batch[1][0],str) \
                                            else str(batch[1][0].item())

                curr_sample_viz = self.sample(inp_tensor, batch[1], num_styles=params.num_styles, \
                                                isTesting=True, save_gen_img=True, params=params)
                viz = vutils.make_grid(curr_sample_viz, nrow=curr_sample_viz.size(0)).unsqueeze(0)

                save_gen_imgs(viz, '%s/viz_%s/gen%d_%s_viz.tif' \
                            %(params.gen_dir,params.exp_name,params.num_styles,curr_name), self.isGrayscale)
    

    def dis_update(self, x, metrics):
        '''
        An implementation of updating the discriminator based off of
        https://github.com/NVlabs/MUNIT
        '''
        self.dis_opt.zero_grad()
        s = torch.randn( size=(x.size(0), self.style_dim, 1, 1), device=self.device, requires_grad=True)

        ## Encode
        content, _ = self.gen.encode(x)

        ## Decode to new style
        x_new = self.gen.decode(content, s)

        ## D loss
        loss_dis = self.dis.calc_dis_loss(x_new.detach(), x)
        loss_dis_total = self.config_params['gan_w'] * loss_dis
        loss_dis_total.backward()
        metrics['Dis'] = loss_dis_total.item()

        self.dis_opt.step()

    
    def gen_update(self, x, metrics, itr):
        '''
        An implementation of updating the generator based off of
        https://github.com/NVlabs/MUNIT
        '''
        self.gen_opt.zero_grad()
        s = torch.randn( size=(x.size(0), self.style_dim, 1, 1), device=self.device, requires_grad=True)

        ## Encode
        content, curr_style = self.gen.encode(x)

        ## Decode (within domain)
        x_recon = self.gen.decode(content, curr_style)
        ## Decode to new style
        x_new = self.gen.decode(content, s)

        ## Encode new style
        c_new_recon, s_new_recon = self.gen.encode(x_new)

        ## Decode again (if needed)
        x_aba = self.gen.decode(c_new_recon, curr_style) if self.config_params['recon_x_cyc_w'] > 0 else None

        ## Reconstruction losses
        loss_gen_recon_x = self.l1_criterion(x_recon, x)
        loss_gen_recon_s = self.l1_criterion(s_new_recon, s)
        loss_gen_recon_c = self.l1_criterion(c_new_recon, content)

        loss_gen_cycrecon_x = self.l1_criterion(x_aba, x) if self.config_params['recon_x_cyc_w'] > 0 else 0

        ## GAN loss
        loss_gen_adv = self.dis.calc_gen_loss(x_new)

        ## Preprocess for perceptual loss
        x_pre = x*0.5+0.5; x_new_pre = x_new*0.5+0.5;
        if x.size(1) == 1:
            x_pre = x_pre.repeat(1,3,1,1); x_new_pre = x_new_pre.repeat(1,3,1,1)
        x_pre = TF.normalize(x_pre, mean=get_color_mean(), std=get_color_std())
        x_new_pre = TF.normalize(x_new_pre, mean=get_color_mean(), std=get_color_std())

        ## Compute perceptual loss
        ftrs_org = self.vgg(x_pre)
        ftrs_recon = self.vgg(x_new_pre)
        loss_gen_recon_perp = self.mse_criterion(ftrs_recon, ftrs_org)

        metrics['Gen/Adv'] = loss_gen_adv.item()
        metrics['Gen/X_recon'] = loss_gen_recon_x.item()
        metrics['Gen/c_recon'] = loss_gen_recon_c.item()
        metrics['Gen/s_recon'] = loss_gen_recon_s.item()
        metrics['Gen/percep_recon'] = loss_gen_recon_perp.item()

        ## Total loss
        loss_gen_total = self.config_params['gan_w'] * loss_gen_adv + \
                              self.config_params['recon_x_w'] * loss_gen_recon_x + \
                              self.config_params['recon_s_w'] * loss_gen_recon_s + \
                              self.config_params['recon_c_w'] * loss_gen_recon_c + \
                              self.config_params['recon_x_cyc_w'] * loss_gen_cycrecon_x + \
                              self.config_params['vgg_w'] * loss_gen_recon_perp \

        loss_gen_total.backward()
        metrics['Gen/Overall'] = loss_gen_total.item()

        self.gen_opt.step()


    def sample(self, inp_tensor, inp_name, num_styles=1, isTesting=False, save_gen_img=False, params=None):
        '''
        Generate new images via sampling style vector

        Args:
            inp_tensor - input
            inp_name - input's file name
            num_styles - number of style vector
            isTesting - in inference stage [True|False]
            save_gen_img - saving generated images [True|False] 
            params - user's input parameters defined in terminal
        '''
        self.gen.eval(); self.dis.eval()

        s = torch.randn( size=(num_styles, self.style_dim, 1, 1), device=self.device)

        curr_viz = []
        for i in range(inp_tensor.size(0)):
            curr_viz.append(inp_tensor[i].unsqueeze(0))

            content, curr_style = self.gen.encode(inp_tensor[i].unsqueeze(0))
            curr_x_recon = self.gen.decode(content, curr_style)

            if save_gen_img:
                curr_file_name = inp_name[0].split('/')[-1] if isinstance(inp_name[0],str) \
                                            else str(inp_name[0].item())+'.tif'

                save_gen_imgs(curr_x_recon, '%s/recon_%s/recon_%s' \
                                %(params.gen_dir,params.exp_name,curr_file_name), self.isGrayscale)

            curr_viz.append(curr_x_recon)

            if not isTesting:
                curr_x_new1 = self.gen.decode(content, self.s)
                curr_viz.append(curr_x_new1)

            for j in range(num_styles):
                curr_x_new2 = self.gen.decode(content, s[j].unsqueeze(0))
                curr_viz.append(curr_x_new2)

                if save_gen_img:
                    save_gen_imgs(curr_x_new2, '%s/%s/gen%d_%s' \
                                    %(params.gen_dir,params.exp_name,j,curr_file_name), self.isGrayscale)

        sample_viz = torch.cat(curr_viz,0)

        if not isTesting:
            self.gen.train(); self.dis.train()

        return sample_viz


    def load_checkpoint(self, params):        
        ## Load generators
        last_model_name = get_model_list(params.checkpoint_dir,'gen', params.checkpoint)
        print('Load Gen model with %s' %last_model_name)
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict)        


    def save_checkpoint(self, snapshot_dir, iterations):
        ## Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations))
        torch.save(self.gen.state_dict(), gen_name)     


    def visualize_img(self, inp_tensor, img_name, num_styles, datatype, itr):
        '''
        Create visualization for tensorboard

        Args:
            inp_tensor - input
            inp_name - input's file name
            num_styles - number of style vector
            datatype - current stage [train|val]
            itr - current iteration
        '''
        curr_sample_viz = self.sample(inp_tensor, img_name, num_styles=num_styles)*0.5 + 0.5
        recon_error =  torch.sum(torch.sqrt(torch.mean((curr_sample_viz[1:2,...] \
                                    - inp_tensor*0.5+0.5)**2, dim=(1,2,3)))).item()
        viz = vutils.make_grid(curr_sample_viz, \
                            nrow=inp_tensor.size(0)*(num_styles+3) if inp_tensor.size(0)<4 else 12 )
        self.curr_writer.add_image('Viz/%s' %datatype, viz, itr)

        return recon_error


    def viz_dataset(self, dataloader, dataset):
        '''
        Visualize a specific dataset
        '''
        for train_batch, test_batch in zip(dataloader['train'],dataloader['test']):
            train_grid = vutils.make_grid(train_batch[0], nrow=3)
            save_gen_imgs(train_grid, './train_%s_viz.png' %dataset)
            
            test_grid = vutils.make_grid(test_batch[0], nrow=3)
            save_gen_imgs(test_grid, './test_%s_viz.png' %dataset)
            return
