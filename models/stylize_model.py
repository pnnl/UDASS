import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from utils.utils import *
from networks.style_networks import *

class Stylize(object):
    '''
    The appearance transformation model buffer class for transforming input for downstream task

    Args:        
        config_params - parameters defined in yaml config file        
        checkpoint_file - name of checkpoint file to be restored
        device - type of device used for running the model
    '''
    def __init__(self, config_params, checkpoint_file, device):

        self.style_gen = AdaINGen(config_params['input_dim'], config_params['gen']).to(device)         
        self.style_gen.load_state_dict( torch.load(checkpoint_file))
        
        set_grads(self.style_gen.enc_style, False)
        set_grads(self.style_gen.enc_content, False)
        self.style_gen.eval()    
    
        self.style_dim = config_params['gen']['style_dim']
    
    def get_embeddings(self, inp, isNorm=True):
        '''
        Extract content and style vectors of a given input

        Args:        
            inp - input tensor
            isNorm - normalized input [True|False]
        '''
        if isNorm:            
            norm_vec = [0.5,0.5,0.5] if inp.size(1) == 3 else [0.5]    
            style_aug_inp = TF.normalize(inp, mean=norm_vec, std=norm_vec)                
        else:
            style_aug_inp = inp                    
        content_code, style_code = self.style_gen.encode(style_aug_inp)
        
        return content_code, style_code

    def stylize(self, content, s=None, isGrayscale=False, isNorm=False):
        '''
        Generate an image from given content and style vectors

        Args:        
            content - contect vector
            s - style vector (randomly sample if None is given)
            isGrayscale - set generated image to be grayscle
            isNorm - normalize generated image
        '''
        if s is None:
            s = torch.randn( size=(content.size(0), self.style_dim, 1, 1) ).cuda()
        inp_new = self.style_gen.decode(content, s)*0.5 + 0.5            
        
        if isGrayscale:
            inp_new = convert_grayscale(inp_new)            
    
        if isNorm:
            inp_new = TF.normalize(inp_new, mean=get_color_mean(), std=get_color_std())

        return inp_new
    
