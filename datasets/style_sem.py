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


def get_transforms(logger, datatype, n_channel, new_size=None):
    '''
    Define transformations for dataset class
    '''
    transform_list = [transforms.ToTensor()]

    transform_list += [transforms.RandomHorizontalFlip()] if datatype=='train' else []
    transform_list += [transforms.Resize(new_size)] if new_size is not None else []

    norm = (0.5)
    if n_channel == 3:
        norm = (0.5,0.5,0.5)

    transform_list += [transforms.Normalize(norm,norm)]

    transform = transforms.Compose(transform_list)
    logger.info("Transform funcs: {:}\n".format(transform))

    return transform

class Basic_Dataset(Dataset):
    '''
    An abstract class for dataset class used in the appearance transformation    
    '''
    def __init__(self, logger, rdir, datatype, n_channel, new_size):

        self.rdir = rdir
        self.data_transforms = get_transforms(logger, datatype, n_channel, new_size)


class SEM_Dataset(Basic_Dataset):
    '''
    The dataset class of the experimental dataset shown in the manuscript

    Args:
        logger - logging object
        rdir - root directory of the dataset
        datatype - type of dataset [train|val|test]
        n_channel - number of input's channel
        new_size(optional) - new spatial dimension for resize transformation
        subdir(optional-only applicable to the experimental dataset)
    '''
    def __init__(self, logger, rdir, datatype, n_channel, new_size=None, subdir=[]):
        Basic_Dataset.__init__(self, logger, rdir, datatype, n_channel, new_size)

        files = []
        subdir = subdir[1:-1].split(',')

        ## Load image from multiple magnifications
        for s in subdir:
            files += glob.glob(rdir + '/%s/*.tif' %(s))

        self.im_list = files
        self.num_file = len(self.im_list)
        self.n_channel = n_channel

    def __len__(self):
        return self.num_file

    def __getitem__(self, idx):

        img_name = self.im_list[idx]

        img = tiff.imread(img_name)
        if len(img.shape) < 3:
            img = np.expand_dims(img, 2)
            img = np.tile(img,[1,1,3])

        if img.dtype == np.uint16:
            img = img.astype(np.float32)
            img = (img / 65535) * 255
            img = img.astype('uint8')


        img_ten = self.data_transforms(img)

        return img_ten, img_name
