
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.sampler as sampler

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np

from layers import DomainAdaptiveBatchNorm1d, L2NormScaled
import utils


class FeatureExtractor(nn.Module):

    supported_base_models = set([
        'resnet34',
        'resnet50'
    ])

    def __init__(self, num_domains=2, base_model='resnet34', pretrained=True):
        super().__init__()

        if base_model not in FeatureExtractor.supported_base_models:
            raise NotImplementedError("Base model {} not supported or registered.".format(base_model))

        if base_model == 'resnet34':
            base = models.resnet34(pretrained=pretrained)
            base = utils.convert_to_adaptive_batchnorm(base, num_domains)
            # We remove all but the last linear layer.
            # The last layer we are left with is an AdaptiveAvgPool2d, which gives us
            # a single column of 512 features.
            self.base = nn.Sequential()
            for name, child in base.named_children():
                if name == 'fc':
                    break
                self.base.add_module(name, child)
            self.out_channels = 512
        elif base_model == 'resnet50':
            base = models.resnet50(pretrained=pretrained)
            base = utils.convert_to_adaptive_batchnorm(base, num_domains)
            self.base = nn.Sequential()
            for name, child in base.named_children():
                if name == 'fc':
                    break
                self.base.add_module(name, child)
            self.out_channels = 512
        else:
            pass
            # Add other model definitions here.
    
        self.flatten = nn.modules.Flatten(start_dim=1)

    def forward(self, x):
        # Flatten spatial dimensions. What we should have is a minibatch
        # of feature vectors.
        x = self.base.forward(x)
        x = self.flatten.forward(x)
        return x


class Model(nn.Module):

    def __init__(self, num_domains=2, base_model='resnet34', feature_depth=512):
        super().__init__()

        # Feature depth of 1024 according to 
        #  Snell et al 2017. Prototypical Networks for Few-shot Learning. Section 3.3 on Caltech-UCSD birds
        feature_depth = feature_depth // 2

        self.extractor = FeatureExtractor(num_domains, base_model)
        self.l2norm = L2NormScaled(c=100, p=0.9)
        self.feat_visual = nn.Linear(self.extractor.out_channels, feature_depth, bias=None)
        self.feat_semantic = nn.Linear(self.extractor.out_channels, feature_depth, bias=None)

        for param in self.extractor.parameters(): # Freeze extractor
            param.requires_grad = False

        self.da_layers = utils.list_domain_adaptive_layers(self)


    def set_domain(self, idx_domain):
        for layer in self.da_layers:
            layer.idx_domain.fill_(idx_domain)


    def forward(self, x) -> torch.Tensor:
        x = self.extractor.forward(x)
        x = self.l2norm.forward(x)
        phi_vis = self.feat_visual.forward(x)
        phi_sem = self.feat_semantic.forward(x)
        x = torch.cat((phi_vis, phi_sem), dim=0)
        return x, phi_vis, phi_sem

