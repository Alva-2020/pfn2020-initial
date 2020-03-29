import torch
import torch.nn as nn
from layers import (DomainAdaptiveModule, DomainAdaptiveBatchNorm1d,
        DomainAdaptiveBatchNorm2d, DomainAdaptiveBatchNorm3d)

def list_domain_adaptive_layers(module:nn.Module, ls=None) -> list:
    r"""Given an nn.Module instance, returns a list of modules which all
    subclass DomainAdaptiveModule.
    """
    if ls is None:
        ls = []
    for sub_module in module._modules.values():
        if isinstance(sub_module, DomainAdaptiveModule):
            ls.append(sub_module)
        list_domain_adaptive_layers(sub_module, ls) # Depth-first recursion
    yield ls


def convert_to_adaptive_batchnorm(module:nn.Module, num_domains) -> nn.Module:
    r"""Recursively steps through a module, converting in-place BatchNorm1d,
    2d, 3d layers into the adaptive equivalent. If the model was trained, then
    this initializes all domains' beta and gamma in the DABatchNorm layer to the
    trained beta and gamma of the original batchnorm layer.
    """
    # Stores list of adaptive batchnorm layers in module.
    # Use it to set idx_domain on all such layers when changing domains.
    for name, sub_module in module._modules.items():
        is_bn1d, is_bn2d, is_bn3d = False, False, False
        if isinstance(sub_module, nn.BatchNorm1d):
            is_bn1d = True
        elif isinstance(sub_module, nn.BatchNorm2d):
            is_bn2d = True
        elif isinstance(sub_module, nn.BatchNorm3d):
            is_bn3d = True

        if is_bn1d or is_bn2d or is_bn3d:
            num_features = sub_module.num_features
            num_domains = num_domains
            eps = sub_module.eps
            momentum = sub_module.momentum
            affine = sub_module.affine
            track_running_stats = sub_module.track_running_stats

            if is_bn1d:
                replacement_module = DomainAdaptiveBatchNorm1d(
                    num_features, num_domains, eps, momentum, affine,
                    track_running_stats)
            elif is_bn2d:
                replacement_module = DomainAdaptiveBatchNorm2d(
                    num_features, num_domains, eps, momentum, affine,
                    track_running_stats)
            else:
                replacement_module = DomainAdaptiveBatchNorm3d(
                    num_features, num_domains, eps, momentum, affine,
                    track_running_stats)

            if affine:
                for idx in range(num_domains):
                    setattr(replacement_module, 'weight_{}'.format(str(idx)), sub_module.weight)
                    getattr(replacement_module, 'weight_{}'.format(str(idx))).requires_grad = sub_module.weight.requires_grad
                    setattr(replacement_module, 'bias_{}'.format(str(idx)), sub_module.bias)
                    getattr(replacement_module, 'bias_{}'.format(str(idx))).requires_grad = sub_module.bias.requires_grad

            module._modules[name] = replacement_module

        convert_to_adaptive_batchnorm(sub_module, num_domains) # Depth-first recursion

    return module

