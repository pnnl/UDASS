import os
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import copy

from torch.autograd import Variable
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import torchvision.transforms.functional as TF

from models.stylize_model import *
from networks.vqvae import *
from utils.style_utils import *
from utils.utils import *

class VQVAE_Model(object):
    '''
    VQVAE class declaration

    Args:
        params - user's input parameters defined in terminal
        config_params - parameters defined in yaml config file        
        style_config_params - parameters for the appearance transformation model defined in yaml config file        
    '''
    def __init__(self, params, config_params, style_config_params=None):

        self.params = params
        self.config_params = config_params

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ## Initiate the networks
        encoders = [QuarterEncoder(self.config_params['input_dim'], self.config_params['latent_size'], self.config_params['latent_count']), \
                    HalfEncoder(self.config_params['latent_size'], self.config_params['latent_size'], self.config_params['latent_count'])]
        decoders = [HalfDecoder(self.config_params['latent_size'], self.config_params['latent_size']), \
                    HalfQuarterDecoder(self.config_params['latent_size'], self.config_params['input_dim'])]
        self.vqvae_model = VQVAE(encoders, decoders).to(self.device)

        self.optimizer = torch.optim.Adam(self.vqvae_model.parameters(), lr=self.config_params['lr'],\
                                            betas=(self.config_params['beta1'],self.config_params['beta2']))

        if not os.path.exists('./results/tensorboards/vqvae'):
            os.makedirs('./results/tensorboards/vqvae')
        if not os.path.exists('./results/checkpoints/vqvae'):
            os.makedirs('./results/checkpoints/vqvae')

        if params.stage != 'test':
            self.save_checkpoint_dir = './results/checkpoints/vqvae/%s_%s' %(params.dataset,params.exp_name)
            if not os.path.exists(self.save_checkpoint_dir):
                os.makedirs(self.save_checkpoint_dir)

            if params.stage == 'train':
                self.curr_writer = SummaryWriter('./results/tensorboards/vqvae/%s_%s' \
                                            %(params.dataset,params.exp_name))

        self.style_model = None
        if style_config_params is not None:
            print('--- Initiate Gen Model ---')
            self.style_model = Stylize(style_config_params, params.style_checkpoint_file, self.device)
            self.style_config_params = style_config_params


    def train(self, dataloader_dict, logger):
        '''
        Training stage
        '''
        start_time = time.time()
        self.vqvae_model.train()

        metrics = {'Overall_Loss': 0.0, 'VQ_Loss': 0.0, 'Recon_Loss': 0.0}

        itr = 0
        for epoch in range(self.config_params['num_ep']):
            for batch in dataloader_dict['train']:

                self.optimizer.zero_grad()

                inp_ten = batch[0].to(self.device)
                curr_terms = self.vqvae_model(inp_ten, commitment=self.config_params['commitment'])

                curr_terms['Overall_Loss'].backward()
                self.optimizer.step()
                self.vqvae_model.revive_dead_entries()

                if itr % self.config_params['tensorboard_itr'] == 0:
                    line = "Itr# %d ->" %itr
                    for item in metrics:
                        self.curr_writer.add_scalar(item, curr_terms[item].item(), itr)
                        line += ' %s = %.4f -' %(item, curr_terms[item].item())
                    logger.info(line[:-1])

                if itr % self.config_params['save_itr'] == 0 and itr != 0:
                    self.vqvae_model.eval()
                    with torch.no_grad():
                        _, _ = self._viz_img(inp_ten, '', itr)

                        ## Load test data
                        ridx = np.random.randint(0,len(dataloader_dict['test']))
                        test_tensor = dataloader_dict['test'][ridx][0].to(self.device).unsqueeze(0)
                        test_recon_error, test_occup = self._viz_img(test_tensor, 'test', itr)

                        logger.info("Test Recon Error = {:.5f}; Occupied Cookbood = {:.5f}".format(test_recon_error, test_occup))

                    self.vqvae_model.train()
                    torch.save(self.vqvae_model.state_dict(), os.path.join(self.save_checkpoint_dir, 'vqvae_%08d.pt' % (itr)))

                itr += 1

        torch.save(self.vqvae_model.state_dict(), os.path.join(self.save_checkpoint_dir, 'vqvae_%08d.pt' % (itr)))
        self.curr_writer.close()
        logger.info("Finished training in %.4f min" %( (time.time()-start_time)/60 ))


    def inference(self, dataloader):
        '''
        Inference stage
        '''
        self._load_checkpoint(self.params)
        self.vqvae_model.eval()

        if not os.path.exists('./results/viz/vqvae'):
            os.makedirs('./results/viz/vqvae')
        viz_dir = '%s_%s' %(self.params.dataset,self.params.exp_name)
        if not os.path.exists('./results/viz/vqvae/%s' %viz_dir):
            os.makedirs('./results/viz/vqvae/%s' %viz_dir )

        occup = 0.0; recon_error = 0.0

        with torch.set_grad_enabled(False):
            for bi, batch in enumerate(dataloader):

                inp_tensor = batch[0].to(self.device)
                curr_name = batch[1][0].split('/')[-1][:-4]

                top_recons, real_recons, curr_occup, curr_recon_error = self._recon_img(inp_tensor)
                occup += curr_occup; recon_error += curr_recon_error
                curr_sample_viz = torch.cat((inp_tensor,real_recons), 0)
                viz = vutils.make_grid(curr_sample_viz, nrow=curr_sample_viz.size(0))
                viz = viz.repeat([1,3,1,1])

                save_gen_imgs(viz, './results/viz/vqvae/%s/%s_viz.tif' \
                            %(viz_dir,curr_name), isNorm=False, isGrayscale=True)

        occup /= len(dataloader.dataset); recon_error /= len(dataloader.dataset)
        print('Average Occupied Indices: %.4f' %occup)
        print('Average RMSE: %.6f' %recon_error)


    def _viz_img(self, inp, datatype, itr):
        '''
        Create visualization for tensorboard

        Args:
            inp - input            
            datatype - current stage [train|val]
            itr - current iteration
        '''
        top_recons, real_recons, occup, _ = self._recon_img(inp)
        curr_sample_viz = torch.cat((inp,real_recons,top_recons), 0)

        viz = vutils.make_grid(curr_sample_viz, nrow=curr_sample_viz.size(0) if inp.size(0) == 1 else self.config_params['batch_size'])
        self.curr_writer.add_image('Viz/%s' %datatype, viz, itr)

        if datatype == 'test':
            recon_error =  torch.mean((real_recons- inp)**2)
            return recon_error.item(), occup / inp.size(0)

        return 0.0, 0.0


    def _recon_img(self, inp):
        '''
        Reconstruct input
        '''
        recons = [x for x in self.vqvae_model.full_reconstructions(inp)]

        top_recons, real_recons, real_idx, top_idx = recons
        top_recons = torch.clamp(top_recons, 0, 1)
        real_recons = torch.clamp(real_recons, 0, 1)
        
        recon_error =  torch.sum(torch.mean((real_recons- inp)**2, dim=(1,2,3))).item()
        occup = 0.0
        for i in range(real_idx.size(0)):
            occup += (torch.unique(real_idx[i,...]).size(0)/self.config_params['latent_count'] \
                        + torch.unique(top_idx[i,...]).size(0)/self.config_params['latent_count'])/2

        return top_recons, real_recons, occup, recon_error


    def _load_checkpoint(self, params):
        last_model_name = get_model_list(params.checkpoint_dir,"vqvae", params.checkpoint)
        print('Load Gen model with %s' %last_model_name)
        state_dict = torch.load(last_model_name)
        self.vqvae_model.load_state_dict(state_dict)
