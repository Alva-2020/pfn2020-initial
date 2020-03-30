import torch
import torch.optim as optim
import torch.utils.data.sampler as sampler

import numpy as np

from model import Model

def train_model(model:Model, train_loader, val_loader, n_epochs=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.set_domain(0) # Train on source
    for param in model.extractor.parameters(): # Freeze extractor
        param.requires_grad = False

    # TODO lr schedule
    optimizer = optim.SGD(model.parameters(), momentum=0.9, nesterov=True)

    # Original algorithm in Snell 2017.
    # However, let's try to implement a training algorithm that's a bit more friendly to minibatches.
    # We randomly initialize all centroids in separate memory in the same space i.e. same feature depth.
    # For each minibatch we get the label prediction. Label prediction is by distance to centroid.
    # The loss penalizes wrong labels. What would this do?
    # This would cause the centroid to move away from wrong \phi(x) and towards the right ones.
    # Likewise this would cause \phi(x) to move towards the right centroid and away from the wrong ones.
    # What we want to see is, at convergence, despite being initialized separately, the centroids
    # should align with the feature space \phi.



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

    # print(model.idx_l2g)
    # print(model.idx_l2g.T)
