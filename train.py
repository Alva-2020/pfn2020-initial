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
    model = Model(num_domains=2)
    x = torch.from_numpy(np.zeros((16,3,255,255))).float()
    x = model.forward(x)
    print(x)    
