
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # Flatten spatial dimensions. What we should have is a minibatch of feature vectors.
        x = self.base.forward(x)
        x = self.flatten.forward(x)
        return x



class Model(nn.Module):

    # num_classes  : How many centroids we want to track.
    # num_domains  : How many domains is the model aware of. Should be 2 unless
    #                we are doing multi-DA
    # base_model   : Model used for the feature extractor.
    # label_groups : Dict<int,set<int>>. Keys are label groups. Values are its class members.
    # label_names, group_names, domain_names : Dict<int,string> (Optional)

    # Model expects everything to be contiguously 0-indexed: classes, domains, label groups.

    def __init__(self, num_classes, num_domains=2, label_groups:dict=None,
                label_names:dict=None, group_names:dict=None, domain_names:dict=None,
                base_model:str='resnet34', feature_depth=512,
                    proto_momentum=0.1):
        super().__init__()

        # Total no. of features is 1024 according to 
        #  Snell et al 2017. Prototypical Networks for Few-shot Learning. Section 3.3 on Caltech-UCSD birds
        # Whatever it's set to, we split 1/2 between semantic and visual features.
        feature_depth = feature_depth // 2

        self.num_classes = num_classes
        self.num_domains = num_domains
        self.num_groups = 0 if label_groups is None else len(label_groups.keys())
        self.proto_momentum = proto_momentum

        # Membership matrix mapping labels to groups and vice versa
        self.register_buffer('idx_g2l', torch.zeros((self.num_groups, self.num_classes), dtype=torch.bool))
        self.register_buffer('idx_l2g', self.idx_g2l.T) # This is just a transposed view of the same matrix
        for group, label_set in label_groups.items():
            self.idx_g2l[group][list(label_set)] = 1

        # Names
        self.domain_names = domain_names if domain_names is not None else dict(((domain, str(domain)) for domain in range(self.num_domains)))
        if label_groups is not None:
            self.label_names = label_names if label_names is not None else dict(((label, str(label)) for label in range(self.num_classes)))
            self.group_names = group_names if group_names is not None else dict(((group, str(group)) for group in range(self.num_groups)))
        else:
            self.label_names = None
            self.group_names = None
        
        # Component layers
        self.extractor = FeatureExtractor(num_domains, base_model)
        self.l2norm = L2NormScaled(c=100, p=0.9)
        self.feat_visual = nn.Linear(self.extractor.out_channels, feature_depth, bias=None)
        self.feat_semantic = nn.Linear(self.extractor.out_channels, feature_depth, bias=None)
        for param in self.extractor.parameters(): # Freeze extractor
            param.requires_grad = False
        self.da_layers = utils.list_domain_adaptive_layers(self)

        # Prototypes
        self.proto_class_visual = nn.Parameter(torch.Tensor(num_classes, feature_depth))
        self.proto_class_semantic = nn.Parameter(torch.Tensor(num_classes, feature_depth))
        if self.num_groups > 0:
            self.proto_group_visual = nn.Parameter(torch.Tensor(self.num_groups, feature_depth))
            self.proto_group_semantic = nn.Parameter(torch.Tensor(self.num_groups, feature_depth))
        else:
            self.register_parameter('proto_group_visual', None)
            self.register_parameter('proto_group_semantic', None)


    def set_domain(self, idx_domain):
        if len(self.da_layers) != 0:
            for layer in self.da_layers:
                layer.idx_domain.fill_(idx_domain)


    def forward(self, x) -> torch.Tensor:
        x = self.extractor.forward(x)
        x = self.l2norm.forward(x)
        phi_vis = self.feat_visual.forward(x)
        phi_sem = self.feat_semantic.forward(x)
        x = torch.cat((phi_vis, phi_sem), dim=0)
        return x, phi_vis, phi_sem

