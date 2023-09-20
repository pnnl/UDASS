'''
    An implementation of image-to-image translation model based off of
    https://github.com/NVlabs/MUNIT
'''

from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torchvision import models

from packaging import version
import numpy as np

from networks.layers import *

LAYERS_IND = {'c11':0, 'c11r':1, 'c12':2, 'c12r':3, 'pool1':4,
                'c21':5, 'c21r':6, 'c22':7, 'c22r':8, 'pool2':9,
                'c31':10, 'c31r':11, 'c32':12, 'c32r':13, 'c33':14, 'c33r':15, 'pool3':16,
                'c41':17, 'c41r':18, 'c42':19, 'c42r':20, 'c43':21, 'c43r':22, 'pool4':23,
                'c51':24, 'c51r':25, 'c52':26, 'c52r':27, 'c53':28, 'c53r':29, 'pool5':30}


class MsImageDis(nn.Module):
    ## Multi-scale discriminator architecture
    def __init__(self, input_dim, input_res, params, device):
        super(MsImageDis, self).__init__()

        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.input_dim = input_dim
        self.input_res = input_res

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for s in range(self.num_scales):
            self.cnns.append(self._make_net(s))

        if self.gan_type == 'org':
            self.criterion = nn.BCEWithLogitsLoss()

        self.device = device


    def _make_net(self, scale):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2

        if self.gan_type == 'org':
            cnn_x += [nn.Flatten()]
            num_ftrs = dim * (( self.input_res // (2**scale) // 16 )**2)
            cnn_x += [nn.Linear(num_ftrs,1)]
        else:
            cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]

        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x


    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs


    def calc_dis_loss(self, input_fake, input_real):
        ## Calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'org':
                b_size = out0.size(0)

                label_real = torch.full((b_size,), 1.0, dtype=torch.float, device=self.device)
                label_fake = torch.full((b_size,), 0.0, dtype=torch.float, device=self.device)

                errD_real = self.criterion(out1.view(-1,), label_real)
                errD_fake = self.criterion(out0.view(-1,), label_fake)

                loss += errD_real + errD_fake

            elif self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)

            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


    def calc_gen_loss(self, input_fake):
        ## Calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type  == 'org':
                b_size = out0.size(0)

                label_real = torch.full((b_size,), 1.0, dtype=torch.float, device=self.device)
                loss += self.criterion(out0.view(-1,), label_real)

            elif self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2)

            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))

            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


##################################################################################
# Generator
##################################################################################
class AdaINGen(nn.Module):
    ## Auto-encoder architecture
    def __init__(self, input_dim, params):
        super(AdaINGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']

        ## Style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        ## Content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images):
        ## Reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        ## Encode an image into its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):
        ## Decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        ## Assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        ## Return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params


##################################################################################
# Encoder and Decoders
##################################################################################
class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)]
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        ## Downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2

        ## Residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        ## AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]

        ## Upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2,mode='nearest'),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2

        ## Use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# VGG network definition
##################################################################################
class Vgg16(nn.Module):
    def __init__(self, extracted_layers=['c11r','c21r','c31r','c41r','c51r']):

        super(Vgg16, self).__init__()

        model_class = models.vgg16(pretrained=True)
        model_class = model_class.features

        ## Put layers in between extracted outputs into a sequential block
        self.network = nn.ModuleList( [nn.Sequential(*model_class[:LAYERS_IND[extracted_layers[0]]+1])] )
        for li,layer in enumerate(extracted_layers[1:], start=1):
            self.network.append( nn.Sequential(*model_class[LAYERS_IND[extracted_layers[li-1]]+1:LAYERS_IND[layer]+1]) )

        print('Sequential Blocks for layer extraction:')
        for l in self.network:
            print(l)

        self.extracted_layers = extracted_layers

    def forward(self, x):
        ftrs_dict = {}
        ftrs_dict[self.extracted_layers[0]] = self.network[0](x)
        for li,layer in enumerate(self.extracted_layers[1:], start=1):
            ftrs_dict[layer] = self.network[li](ftrs_dict[self.extracted_layers[li-1]])

        return ftrs_dict[self.extracted_layers[0]]
