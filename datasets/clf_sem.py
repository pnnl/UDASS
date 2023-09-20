from PIL import Image as PILimage
import glob
import numpy as np
import os
import sys
import tifffile as tiff

from torch.utils.data import Dataset
from torchvision import transforms


class ClfSEM_Dataset(Dataset):
    '''
    The dataset class for classification task

    Args:
        txt_file - text file stores a list of samples in which each line contains file name and its associated class
        datatype - type of dataset [train|val|test]
        adapt_type - type of adaptation
        ref - statistics of training samples used in histogram matching/whitening and coloring transformation
        type_network - type of classification model [miso|vqvae_clf]
    '''
    def __init__(self, txt_file, datatype, adapt_type, ref=None, type_network=None):

        self.img_files = []
        self.labels = []
        file = open(txt_file)
        for l in file:
            val = l.strip('\n').split(' ')

            if 'vqvae_clf' in type_network:
                for v in val[:-1]:
                    if [v] not in self.img_files:
                        self.img_files.append([v])
                        self.labels.append(val[-1])
            else:
                self.labels.append(val[-1])
                self.img_files.append(val[:-1])

        self.num_mags = len(self.img_files[0])
        self.classes = self._get_classes()

        self.adapt_type = adapt_type
        ## Load reference statistics from specified binary file for WCT & HM
        if ref != '':
            self.ref = np.load(ref)

        self.data_transforms = self._get_transforms(datatype)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):

        img_name = self.img_files[idx]
        inp_ten = []
        for mi, m in enumerate(img_name):
            pil_img = tiff.imread(m)
            if pil_img.dtype == np.uint16:
                pil_img = pil_img.astype(np.float32)
                pil_img = (pil_img / 65535) * 255
                pil_img = pil_img.astype('uint8')

            if len(pil_img.shape) < 3:
                pil_img = np.tile(np.expand_dims(pil_img,2), [1,1,3])

            ## Perform histogram matching
            if 'hm' in self.adapt_type:
                pil_img = pil_img.astype(np.float32) / 255.0
                pil_img = self._match_hist(pil_img[...,0])
                pil_img = (pil_img*255.0).astype('uint8')
                pil_img = np.tile(np.expand_dims(pil_img,2), [1,1,3])

            ## Perform WCT transformation
            elif 'wct' in self.adapt_type:
                pil_img = pil_img.astype(np.float32) / 255.0
                pil_img = self._wct(pil_img[...,0])
                pil_img = (pil_img*255.0).astype('uint8')
                pil_img = np.tile(np.expand_dims(pil_img,2), [1,1,3])

            pil_img = PILimage.fromarray(pil_img, mode='RGB')

            curr_ten = self.data_transforms(pil_img)
            inp_ten.append(curr_ten)

        label = self.classes.index( self.labels[idx] )

        return {'inp':inp_ten, 'label':label, 'name':img_name}

    def _get_classes(self):        
        ## TO-DO: Replace this with the corresponding classes in a desired dataset
        return ['A','B','C','D','E','F','G','H']

    def _get_transforms(self, datatype):
        '''
        Get data augmentation processes
        '''

        sz = 280
        if 'train' == datatype:
            return transforms.Compose([\
                    transforms.RandomHorizontalFlip(),\
                    transforms.RandomVerticalFlip(),\
                    transforms.RandomCrop(sz),\
                    transforms.ColorJitter(contrast=0.25),\
                    transforms.ToTensor()])
        elif 'target_train' == datatype:
            return transforms.Compose([\
                    transforms.RandomHorizontalFlip(),\
                    transforms.RandomVerticalFlip(),\
                    transforms.RandomCrop(sz),\
                    transforms.ToTensor()])
        else:
            return transforms.Compose([\
                    transforms.ToTensor()])
    

    def _match_hist(self, img):
        '''
        An implementation of histogram matching based off of
        https://gist.github.com/jcjohnson/e01e4fcf7b7dfa9e0dbee6c53d3120b6
        '''
        img_shape = img.shape
        _, s_indices, s_counts = np.unique(img,return_counts=True,return_inverse=True)

        ## Compute the cumulative sum of the counts
        s_quantiles = np.cumsum(s_counts).astype(float) / (img.size + sys.float_info.epsilon)

        ## Interpolate linearly to find the pixel values in the reference
        ## that correspond most closely to the quantiles in the source image
        interp_values = np.interp(s_quantiles, self.ref, np.arange(256)/255.0)

        ## Pick the interpolated pixel values using the inverted source indices
        transf_img = interp_values[s_indices]
        transf_img = transf_img.reshape(img_shape)

        return transf_img

    def _wct(self, img):
        '''
        Whitening and coloring transformation
        '''

        rz_img = img.reshape(-1,1)
        tbar_img = rz_img - np.mean(rz_img)

        var_tbar = np.var(tbar_img)
        img_white = tbar_img / np.sqrt(var_tbar)

        new_img = img_white * np.sqrt(self.ref[1]) + self.ref[0]
        new_img = np.clip(new_img, 0, 1)
        new_img = np.reshape(new_img, img.shape)

        return new_img
