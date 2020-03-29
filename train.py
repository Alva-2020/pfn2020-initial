import torch
import torch.optim as optim
import torch.utils.data.sampler as sampler

import numpy as np

from model import Model

def train_model(model, train_loader, val_loader, n_epochs=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.set_domain(0) # Train on source

    # TODO


def adapt_model(model, train_loader, val_loader, n_epochs=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.set_domain(1)

    # TODO


def test_model(model, train_loader, val_loader, n_epochs=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # TODO


if __name__ == '__main__':
    import utils

    # Setup arbitrary groups to test initialization
    group2label = {
        0: set([0,1,2,3,4]),
        1: set([5,6,7,8,9])
    }
    group_names = {
        0: 'lt5',
        1: 'gte5'
    }

    # Try to setup for MNIST - SVN, so 10 classes to track
    model = Model(num_classes=10, num_domains=2, label_groups=group2label, group_names=group_names)
    model.set_domain(1)
    x = torch.from_numpy(np.zeros((16,3,255,255))).float()
    x = model.forward(x)
    print(x)
    # print(model.idx_l2g)
    # print(model.idx_l2g.T)
    print(model.idx_g2l[0] == 0)
