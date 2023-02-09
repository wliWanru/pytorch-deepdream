import os
from collections import namedtuple


import torch
from torchvision import models
from torch.hub import download_url_to_file


from utils.constants import *


class AlexNet(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""
    def __init__(self, pretrained_weights,requires_grad=False, show_progress=False):
        super().__init__()
        if pretrained_weights == SupportedPretrainedWeights.IMAGENET.name:
            alexnet = models.alexnet(weights=None, progress=show_progress).eval()
        else:
            alexnet = models.alexnet(weights=None, progress=show_progress).eval()
            state_dict = torch.load('weights/' + pretrained_weights)

            new_state_dict = {}  # modify key names and make it compatible with current PyTorch model naming scheme
            for old_key in state_dict.keys():
                new_key = old_key.replace('.module', '')
                new_state_dict[new_key] = state_dict[old_key]
            alexnet=torch.nn.DataParallel(alexnet)
            alexnet.module.classifier[-1] = torch.nn.Linear(alexnet.module.classifier[-1].in_features, new_state_dict['module.classifier.6.bias'].shape[0])
            alexnet.load_state_dict(new_state_dict, strict=True)

        alexnet_pretrained_features = alexnet.module.features
        alexnet_avgp = alexnet.module.avgpool
        alexnet_fc = alexnet.module.classifier
        self.layer_names = ['relu1', 'relu2', 'relu3', 'relu4', 'relu5','avg','fc7']

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])

        self.slice6.add_module('avg',alexnet_avgp)
        for x in range(5):
            self.slice7.add_module(str(x), alexnet_fc[x])

        # Set these to False so that PyTorch won't be including them in it's autograd engine - eating up precious memory
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, interested_channel):
        x = self.slice1(x)
        relu1 = x
        x = self.slice2(x)
        relu2 = x
        x = self.slice3(x)
        relu3 = x
        x = self.slice4(x)
        relu4 = x
        x = self.slice5(x)
        relu5 = x
        x = self.slice6(x)
        avg5 = x
        x = torch.flatten(x, 1)
        x = self.slice7(x)
        fc7 = x[0,interested_channel]

        alexnet_outputs = namedtuple("AlexNetOutputs", self.layer_names)
        out = alexnet_outputs(relu1, relu2, relu3, relu4, relu5,avg5,fc7)
        return out