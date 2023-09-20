from PIL import Image as PILImage
import numpy as np
import glob
import copy
import os
import tifffile as tiff

from torch.utils.data import Dataset
from torchvision import transforms
import torch
SEED = [10,25,3,42,8]

def get_transforms(datatype, n_channel, new_size=None):
    '''
    Define transformations for dataset class
    '''
    transform_list = [transforms.ToTensor()]

    transform_list += [transforms.RandomHorizontalFlip()] if datatype=='train' else []
    transform_list += [transforms.RandomVerticalFlip()] if datatype=='train' else []
    transform_list += [transforms.Resize(new_size)] if new_size is not None else []

    norm = (0.5)
    if n_channel == 3:
        norm = (0.5,0.5,0.5)

    transform = transforms.Compose(transform_list)
    return transform

class VQ_Dataset(Dataset):
    '''
    The dataset class of the experimental dataset shown in the manuscript for VQVAE model

    Args:
        txt_file - text file stores a list of samples
        datatype - type of dataset [train|val|test]
        n_channel - number of input's channel
        new_size(optional) - new spatial dimension for resize transformation
        mags(optional-only applicable to the experimental dataset)
    '''
    def __init__(self, txt_file, mags, datatype, n_channel, new_size=None):

        files = []
        f = open(txt_file)
        for l in f:
            val = l.strip('\n').split(' ')
            files += val[:-1]

        ## Select only images from the specified magnifications
        files = self._filter_mags(files, mags)
        files.sort()

        self.im_list = files
        self.num_file = len(self.im_list)
        self.n_channel = n_channel

        self.data_transforms = get_transforms(datatype, n_channel)

    def _filter_mags(self, files, mags):
        return list(set([x for x in copy.copy(files) if x.split('/')[-2] in mags]))


    def __len__(self):
        return self.num_file

    def __getitem__(self, idx):

        img_name = self.im_list[idx]

        img = tiff.imread(img_name)
        if self.n_channel > 1 and len(img.shape) < 3:
            img = np.expand_dims(img, 2)
            img = np.tile(img,[1,1,3])
        elif self.n_channel == 1 and len(img.shape) == 3:
            img = img[...,:1]
        elif self.n_channel == 1 and len(img.shape) < 3:
            img = np.expand_dims(img, 2)

        if img.dtype == np.uint16:
            img = img.astype(np.float32)
            img = (img / 65535) * 255
            img = img.astype('uint8')

        img_ten = self.data_transforms(img)

        return img_ten, img_name
