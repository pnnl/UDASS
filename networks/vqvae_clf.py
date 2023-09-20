import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VQVAE_Clf(nn.Module):
    '''
    Classification network for using VQVAE embeddings as input

    Args:
        resnet - type of resnet
        num_classes - number of classes     
        input_dim - number of input's channel   
        mid_dim - number of hidden layer's channel
        model_type - type of classification layer
    '''
    def __init__(self, num_classes, input_dim, mid_dim=512, model_type='vqvae_clf1'):
        super(VQVAE_Clf, self).__init__()

        self.model_type = model_type

        if model_type == 'vqvae_clf1':

            self.classifier = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=mid_dim),
                nn.SELU(),
                nn.Linear(in_features=mid_dim, out_features=mid_dim),
                nn.SELU(),
                nn.Linear(in_features=mid_dim, out_features=num_classes),
            )

        elif model_type == 'vqvae_clf2':

            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(1024,128), nn.ReLU(),
                nn.Linear(128,num_classes)
                )

        elif model_type == 'vqvae_clf3':

            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1),nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(mid_dim,128), nn.ReLU(),
                nn.Linear(128,num_classes)
            )

        elif model_type == 'vqvae_clf4':

            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(mid_dim), nn.ReLU(),
                nn.Conv2d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(mid_dim), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(mid_dim), nn.ReLU(),
                nn.Conv2d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(mid_dim), nn.ReLU(),
                nn.Conv2d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(mid_dim), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(mid_dim), nn.ReLU(),
                nn.Conv2d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(mid_dim), nn.ReLU(),
                nn.Conv2d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(mid_dim), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(mid_dim,128), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Linear(128,num_classes)
            )

        elif model_type == 'vqvae_clf5':

            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(mid_dim), nn.ReLU(),                                            
                nn.AdaptiveMaxPool2d(1), nn.Flatten(),
                nn.Linear(in_features=mid_dim, out_features=mid_dim//4),
                nn.BatchNorm1d(mid_dim//4), nn.ReLU(),
                nn.Linear(in_features=mid_dim//4, out_features=mid_dim//4),
                nn.BatchNorm1d(mid_dim//4), nn.ReLU(),
                nn.Linear(in_features=mid_dim//4, out_features=num_classes),
            )


    def forward(self, x):
        if self.model_type == 'vqvae_clf1':
            x = x.type(torch.cuda.FloatTensor)

        return self.classifier(x)
