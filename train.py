import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import torch.utils.data.sampler as sampler

import torchvision.datasets
import torchvision.transforms as transforms

import numpy as np

from model import Model


def train_mnist():

    # Remark: 
    # This seems to be unstable. Train and val accuracy seems to plateau at around 80-85, dropping to
    # 65-75, even with hyperparameter tuning. Reducing learning rate for the centroid doesn't seem
    # to help very much. Let's see if the original method works better.


    # Original algorithm in Snell 2017.
    # However, let's try to implement a training algorithm that's a bit more friendly to minibatches.
    # We randomly initialize all centroids in separate memory in the same space i.e. same feature depth.
    # For each minibatch we get the label prediction. Label prediction is by distance to centroid.
    # The loss penalizes wrong labels. What would this do?
    # This would cause the centroid to move away from wrong \phi(x) and towards the right ones.
    # Likewise this would cause \phi(x) to move towards the right centroid and away from the wrong ones.
    # What we want to see is, at convergence, despite being initialized separately, the centroids
    # should align with the feature space \phi. The alignment target for the 'centroid' vector is
    # the actual centroid of \phi(x) because that is the position that minimizes the loss.



    # Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)), # Don't normalize since MNIST is 1-channel
        # transforms.Resize((224,224)) # Resize not necessary because of AdaptiveAvg2dPool
        # Don't need to do random crops etc for mnist.
    ])

    # Data
    feature_depth = 64
    batch_size = 256
    n_epochs   = 20
    train_set = torchvision.datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
    val_set   = torchvision.datasets.MNIST('data/mnist', train=False, download=True, transform=transform) # Just use test split for val
    # No need for samplers.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # Model, optim, loss. Define optim before moving model with .to()
    model = Model(num_classes=10, num_domains=2, base_model='resnet18', feature_depth=feature_depth)
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-3)
    optimizer = optim.Adam([
        {'params': model.feat_visual.parameters()},
        {'params': model.feat_semantic.parameters()},
        {'params': model.Ck_vis.parameters(), 'lr': 1e-3},
        {'params': model.Ck_sem.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-3, lr=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5, nesterov=True)
    # lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (1+(epoch/n_epochs))**0.75) 
    loss_fn = nn.CrossEntropyLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Model config
    model.set_domain(0) # Train on source
    for param in model.extractor.parameters(): # Freeze extractor
        param.requires_grad = False


    for epoch in range(n_epochs):

        # Any variables for tracking state etc need to be tensors on device,
        # so everything else doesn't wait for the transfer to happen.
        minibatch = torch.zeros((1), dtype=torch.float32, device=device)
        mean_loss = torch.zeros((1), dtype=torch.float32, device=device)
        mean_acc = torch.zeros((1), dtype=torch.float32, device=device)

        for _, data in enumerate(train_loader):
            mean_loss, mean_acc, minibatch = 0, 0, 0

            x = data[0] # Mnist is 1-channel, so duplicate it along channel axis to get RGB for ResNet
            x = torch.cat((x,x,x), dim=1).to(device, non_blocking=True)
            y = data[1].to(device, non_blocking=True)

            y_pred, soft_y_pred, l2_vis, l2_sem = model.forward(x)
            loss = loss_fn(soft_y_pred, y)

            # y = y.reshape(-1,1)
            # y_pred = y_pred.reshape(-1,1)

            # y_onehot = torch.zeros((256, 10), dtype=torch.long).to(device, non_blocking=True)
            # y_onehot.scatter_(1,y.view(-1,1),1)

            # y_pred_onehot = torch.zeros((256, 10), dtype=torch.float).to(device, non_blocking=True)
            # y_pred_onehot.scatter_(1,y_pred.view(-1,1),1)

            # print(y_onehot)
            # print(y_pred_onehot)

            # loss = loss_fn(y_pred_onehot, y_onehot)

            optimizer.zero_grad() # Reset gradient
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()

            minibatch += 1
            mean_loss = mean_loss + (loss - mean_loss) / minibatch if minibatch > 0 else loss
            acc = torch.sum(y == y_pred).float() / len(y_pred)
            mean_acc = mean_acc + (acc - mean_acc) / minibatch if minibatch > 0 else acc
        print('Epoch: {}, train_loss: {}, train_acc: {}'.format(epoch, mean_loss.item(), mean_acc.item()))
    

        for _, data in enumerate(val_loader):
            mean_loss, mean_acc, minibatch = 0, 0, 0

            x = data[0] # Mnist is 1-channel, so duplicate it along channel axis to get RGB for ResNet
            x = torch.cat((x,x,x), dim=1).to(device, non_blocking=True)
            y = data[1].to(device, non_blocking=True)

            y_pred, soft_y_pred, l2_vis, l2_sem = model.forward(x)
            loss = loss_fn(soft_y_pred, y)

            minibatch += 1
            mean_loss = mean_loss + (loss - mean_loss) / minibatch if minibatch > 0 else loss
            acc = torch.sum(y == y_pred).float() / len(y_pred)
            mean_acc = mean_acc + (acc - mean_acc) / minibatch if minibatch > 0 else acc
        print('Epoch: {}, val_loss: {}, val_acc: {}'.format(epoch, mean_loss.item(), mean_acc.item()))



if __name__ == '__main__':

    train_mnist()
