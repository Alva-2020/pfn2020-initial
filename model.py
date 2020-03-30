
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np

from layers import DomainAdaptiveBatchNorm1d, L2NormScaled, PrototypeSquaredEuclidean, PrototypeCosine
import utils


class FeatureExtractor(nn.Module):

    # Remember that pre-trained ResNet expects whitened inputs based on ImageNet statistics.
    supported_base_models = set([
        'resnet18',
        'resnet34',
        'resnet50'
    ])

    def __init__(self, num_domains=2, base_model='resnet34', pretrained=True):
        super().__init__()

        if base_model not in FeatureExtractor.supported_base_models:
            raise NotImplementedError("Base model {} not supported or registered.".format(base_model))

        if base_model == 'resnet18':
            base = models.resnet18(pretrained=pretrained)
            base = utils.convert_to_adaptive_batchnorm(base, num_domains)
            self.base = nn.Sequential()
            for name, child in base.named_children():  # We remove all but the last linear layer.
                if name == 'fc':                       # The last layer we are left with is an AdaptiveAvgPool2d, which gives us
                    break                              # a single column of 512 features.
                self.base.add_module(name, child)
            self.out_channels = 512

        if base_model == 'resnet34':
            base = models.resnet34(pretrained=pretrained)
            base = utils.convert_to_adaptive_batchnorm(base, num_domains)
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
    # label_groups : Dict<int,set<int>>. Keys are label groups. Values are its class members. (Optional)
    # label_names, group_names, domain_names : Dict<int,string> (Optional)

    # Model expects everything to be contiguously 0-indexed: classes, domains, label groups.

    def __init__(self, num_classes, num_domains=2, label_groups:dict=None,
                label_names:dict=None, group_names:dict=None, domain_names:dict=None,
                base_model:str='resnet34', feature_depth=512,
                lamb = 0.5, proto_momentum=0.1):
        super().__init__()

        # Total no. of features is 1024 according to 
        #  Snell et al 2017. Prototypical Networks for Few-shot Learning. Section 3.3 on Caltech-UCSD birds
        # Whatever it's set to, we split 1/2 between semantic and visual features.
        feature_depth = feature_depth // 2

        self.num_classes = num_classes
        self.num_domains = num_domains
        self.num_groups = 0 if label_groups is None else len(label_groups.keys())
        self.lamb = lamb # lambda hyperparameter
        self.proto_momentum = proto_momentum

        # Names
        self.domain_names = domain_names if domain_names is not None else dict(((domain, str(domain)) for domain in range(self.num_domains)))
        self.label_names = label_names if label_names is not None else dict(((label, str(label)) for label in range(self.num_classes)))
        if label_groups is not None:
            self.group_names = group_names if group_names is not None else dict(((group, str(group)) for group in range(self.num_groups)))
        else:
            self.group_names = None
        self._verify_names(silent=False) # If name dict was passed in, make sure it's completely defined.

        # Membership matrix mapping labels to groups and vice versa.
        if label_groups is not None:
            self.register_buffer('idx_g2l', torch.zeros((self.num_groups, self.num_classes), dtype=torch.bool))
            for group, label_set in label_groups.items():
                self.idx_g2l[group][list(label_set)] = 1
            self.register_buffer('idx_l2g', self.idx_g2l.T.contiguous()) # Contiguous call makes a copy
        else:
            self.register_parameter('idx_g2l', None)
            self.register_parameter('idx_l2g', None)
        
        # Component layers
        self.extractor = FeatureExtractor(num_domains, base_model)
        self.l2norm = L2NormScaled(c=100, p=0.9)
        self.feat_visual = nn.Linear(self.extractor.out_channels, feature_depth, bias=None)
        self.feat_semantic = nn.Linear(self.extractor.out_channels, feature_depth, bias=None)

        self.da_layers = utils.list_domain_adaptive_layers(self)

        # Centroids or prototypes. Ck = class centroids, Cg = group centroids.

        self.Ck_vis = PrototypeSquaredEuclidean(self.num_classes, feature_depth)
        self.Ck_sem = PrototypeSquaredEuclidean(self.num_classes, feature_depth)

        # self.register_buffer('Ck_vis', torch.Tensor(num_classes, feature_depth))
        # self.register_buffer('Ck_vis', torch.Tensor(num_classes, feature_depth))
        # self.Ck_vis = nn.Parameter(torch.Tensor(feature_depth, num_classes).normal_(mean=0, std=1))
        # self.Ck_sem = nn.Parameter(torch.Tensor(num_classes, feature_depth).normal_(mean=0, std=1))
        if self.num_groups > 0:
            self.Cg_sem = PrototypeSquaredEuclidean(self.num_groups, feature_depth)
            # self.register_buffer('Cg_vis', torch.Tensor(self.num_groups, feature_depth))
            # self.register_buffer('Cg_sem', torch.Tensor(self.num_groups, feature_depth))
            # self.Cg_vis = nn.Parameter(torch.Tensor(self.num_groups, feature_depth).normal_(mean=0, std=1))
            # self.Cg_sem = nn.Parameter(torch.Tensor(self.num_groups, feature_depth).normal_(mean=0, std=1))
        else:
            # self.register_parameter('Cg_vis', None)
            self.register_parameter('Cg_sem', None)


    def _verify_names(self, silent=False) -> None:
        for idx in range(self.num_domains):
            if idx not in self.domain_names:
                if silent:
                    self.domain_names[idx] = str(idx)
                else:
                    raise KeyError('Missing name for domain {}'.format(idx))
        for idx in range(self.num_classes):
            if idx not in self.label_names:
                if silent:
                    self.label_names[idx] = str(idx)
                else:
                    raise KeyError('Missing name for label {}'.format(idx))
        if self.group_names is not None:
            for idx in range(self.num_groups):
                if idx not in self.group_names:
                    if silent:
                        self.group_names[idx] = str(idx)
                    else:
                        raise KeyError('Missing name for group {}'.format(idx))


    def reset_running_mean(self) -> None:
        pass #TODO


    def set_domain(self, idx_domain:int) -> None:
        if len(self.da_layers) != 0:
            for layer in self.da_layers:
                layer.idx_domain.fill_(idx_domain)


    def forward(self, x) -> torch.Tensor:
        x = self.extractor.forward(x)
        x = self.l2norm.forward(x)
        phix_vis = self.feat_visual.forward(x)
        phix_sem = self.feat_semantic.forward(x)
        # phi_x = torch.cat((phi_vis, phi_sem), dim=1)

        # These are L2 distances to centroids of the respective layers.
        l2_vis = self.Ck_vis.forward(phix_vis)
        l2_sem = self.Ck_sem.forward(phix_sem)

        # Negative argument because we want a bigger probability when distance is smaller.
        # print(l2_vis)
        tmp_vis = F.softmax(-l2_vis, dim=1)
        tmp_sem = F.softmax(-l2_sem, dim=1)

        soft_y_pred = self.lamb * tmp_vis + (1-self.lamb) * tmp_sem
        y_pred = torch.argmax(soft_y_pred, dim=1)

        # TODO label group penalty
        if self.num_groups > 0:
            pass #TODO
        

        return y_pred, soft_y_pred, l2_vis, l2_sem

