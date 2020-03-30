
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.sampler as sampler

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np

class DomainAdaptiveModule(nn.Module):
    r"""Base module for domain-aware models.
    Has a dict that serves as the counterpart to module_name._modules
    to track all domain-aware modules, so before we pass in a minibatch
    we can set the domain flag for the whole model accordingly.

    All modules that subclass this need to have the attribute .idx_domain
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('idx_domain', torch.zeros(1, dtype=torch.long))



class _DomainAdaptiveBatchNorm(DomainAdaptiveModule):

    # 'Li et al 2016. Revisiting Batch Normalization for Practical Domain Adaptation.'
    # It's just batchnorm with different mean and variance for different domains so
    # layers learn a whitened representation despite domain shift.

    # In 'Saito et al 2020, Universal Domain Adaptation through Self Supervision',
    # authors just use keep source and target samples in different minibatches.
    # However, the model they use (pre-trained ResNet50) has momentum set to 0.1,
    # which means a 0.9 weight to the running average. I'm not sure if this really
    # achieves domain-specific normalization.

    # Setting num_domains = 1 makes this into vanilla batchnorm.
    # Expects domains indexed as integers starting from 0.
    # We rely on the buffer .idx_domain to set and retrieve the domain index, so
    # we don't have to add another parameter to .forward() for better compatibility
    # with other modules.

    # Tried just using index slices of tensors to store values of different domains,
    # but this makes the node not a leaf. Had to bind individual parameters to different
    # attributes.

    def __init__(self, num_features, num_domains=2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        # Just follow _Normbase setup. https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        # We just want one extra beta and gamma for each target domain we have.
        self.num_features = num_features
        self.num_domains = num_domains
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            for idx in (str(domain) for domain in range(self.num_domains)):
                self.register_parameter('weight_{}'.format(idx), nn.Parameter(torch.Tensor(num_features)))
                self.register_parameter('bias_{}'.format(idx), nn.Parameter(torch.Tensor(num_features)))
        else:
            for idx in (str(domain) for domain in range(self.num_domains)):
                self.register_parameter('weight_{}'.format(idx), None)
                self.register_parameter('weight_{}'.format(idx), None)
        if self.track_running_stats:
            for idx in (str(domain) for domain in range(self.num_domains)):
                self.register_buffer('running_mean_{}'.format(idx), torch.zeros(num_features))
                self.register_buffer('running_var_{}'.format(idx), torch.ones(num_features))
                self.register_buffer('num_batches_tracked_{}'.format(idx), torch.zeros(1, dtype=torch.long))
        else:
            for idx in (str(domain) for domain in range(self.num_domains)):
                self.register_parameter('running_mean_{}'.format(idx), None)
                self.register_parameter('running_var_{}'.format(idx), None)
                self.register_parameter('num_batches_tracked_{}'.format(idx), None)
        self.reset_parameters()

        # print(getattr(self, 'num_batches_tracked_{}'.format(str(0))))
        
    def reset_running_stats(self):
        if self.track_running_stats:
            # Remember torch.tensor always copies data. Resetting this way avoids the copy.
            for idx in (str(domain) for domain in range(self.num_domains)):
                getattr(self, 'running_mean_{}'.format(idx)).zero_()
                getattr(self, 'running_var_{}'.format(idx)).fill_(1)
                getattr(self, 'num_batches_tracked_{}'.format(idx)).zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            for idx in (str(domain) for domain in range(self.num_domains)):
                nn.init.ones_(getattr(self, 'weight_{}'.format(idx)))
                nn.init.zeros_(getattr(self, 'bias_{}'.format(idx)))

    def _check_input_dim(self, input): # Unmodified
        raise NotImplementedError

    def extra_repr(self): # Unmodified
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def forward(self, input):
        # Just follow _BatchNorm.forward(), but pass the right beta and gamma for the domain label
        self._check_input_dim(input)
        if self.idx_domain >= self.num_domains:
            raise IndexError('Number of domains set to {} but domain index {} requested.'
                .format(self.num_domains, self.idx_domain))

        idx = str(self.idx_domain.item()) # idx_domain is a tensor, so unpack with .item() to get the int
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if getattr(self, 'num_batches_tracked_{}'.format(idx)) is not None:
                setattr(self, 'num_batches_tracked_{}'.format(idx),
                        getattr(self, 'num_batches_tracked_{}'.format(idx)) + 1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(getattr(self, 'num_batches_tracked_{}'.format(idx)))
                else:
                    exponential_average_factor = self.momentum

        return F.batch_norm(
            input,
            getattr(self, 'running_mean_{}'.format(idx)),
            getattr(self, 'running_var_{}'.format(idx)),
            getattr(self, 'weight_{}'.format(idx)),
            getattr(self, 'bias_{}'.format(idx)),
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps
        )


class DomainAdaptiveBatchNorm1d(_DomainAdaptiveBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'.format(input.dim()))

class DomainAdaptiveBatchNorm2d(_DomainAdaptiveBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

class DomainAdaptiveBatchNorm3d(_DomainAdaptiveBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))



class L2NormScaled(nn.Module):

    # 'Ranjan et al 2017. L2-constrained Softmax Loss for Discriminative Face Verification.'
    # Used in Saito 2019 before last feature layer. This just embeds feature vectors
    # in a hypersphere of radius alpha in Euclidean space. Alpha is a learnable hyperparameter.

    def __init__(self, c=1000, p=0.9):
        super().__init__()
        # C = approximate number of class labels. If not given, default to ImageNet 1k
        # p = classification probability.
        # Both used to set *initial* alpha according to lower bound in paper.
        self.initial_alpha = np.log(p*(c-2)/(1-p))
        self.alpha = nn.Parameter(torch.Tensor(1).fill_(self.initial_alpha))

    def forward(self, input):
        self._check_input_dim(input)
        x = input
        l2 = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
        x = self.alpha * x / l2
        return x

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input). Should be a minbatch of feature vectors.'
                .format(input.dim()))


class PrototypeSquaredEuclidean(nn.Module):

    # Returns Euclidean distance matrix to own prototypes

    def __init__(self, num_prototypes, feature_depth):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.feature_depth = feature_depth
        self.prototypes = nn.Parameter(torch.Tensor(feature_depth, num_prototypes).normal_(0,1))

    def forward(self, input):
        # Expects input shape to be (minibatch, feature_depth)
        # Output shape should be (minibatch, num_prototypes)
        m1 = input[:,:,None]
        m2 = self.prototypes[None,:,:]

        x = (m1 - m2) * (m1 - m2)
        x = torch.sum(x, dim=1)
        # x = torch.sqrt(x) # Squared Euclidean distance; no square root. Snell 2017 section 2.7.
        return x


class PrototypeCosine(nn.Module):
    
    # Returns Cosine distance matrix to own prototypes

    def __init__(self, num_prototypes, feature_depth):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.feature_depth = feature_depth
        self.prototypes = nn.Parameter(torch.Tensor(feature_depth, num_prototypes).normal_(0,1))

    def forward(self, input):
        # Expects input shape to be (minibatch, feature_depth)
        # Output shape should be (minibatch, num_prototypes)
        return input.matmul(self.prototypes)
