import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import torch.utils.data.sampler as sampler

import torchvision.datasets
import torchvision.transforms as transforms

import importlib
import numpy as np

from model import Model
from dataset import OfficeHomeDataset


def train_mnist():



    # Remark: 
    # Problem with instability was with input sizing. Moving on.
    """
    Epoch: 0, train_loss: 1.4926507472991943, train_acc: 1.0
    Epoch: 0, val_loss: 1.529036521911621, val_acc: 0.9375
    Epoch: 1, train_loss: 1.554311990737915, train_acc: 0.9375
    Epoch: 1, val_loss: 1.5333893299102783, val_acc: 0.9375
    Epoch: 2, train_loss: 1.5894888639450073, train_acc: 0.84375
    Epoch: 2, val_loss: 1.517560601234436, val_acc: 0.9375
    Epoch: 3, train_loss: 1.5763475894927979, train_acc: 0.875
    Epoch: 3, val_loss: 1.5333492755889893, val_acc: 1.0
    Epoch: 4, train_loss: 1.5285528898239136, train_acc: 0.96875
    Epoch: 4, val_loss: 1.531996488571167, val_acc: 0.9375
    """

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
        # transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)), # Don't normalize since MNIST is 1-channel
        transforms.Resize((224,224)),
        transforms.ToTensor()
        # Don't need to do random crops etc for mnist.
    ])

    # Parameters
    feature_depth = 64
    batch_size = 64
    n_epochs   = 20

    num_workers = 0 if os.name == 'nt' else 8

    train_set = torchvision.datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
    val_set   = torchvision.datasets.MNIST('data/mnist', train=False, download=True, transform=transform) # Just use test split for val
    # No need for samplers.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # Model, optim, loss. Define optim before moving model with .to()
    model = Model(num_classes=10, num_domains=2, base_model='resnet18', feature_depth=feature_depth)
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    """
    optimizer = optim.Adam([
        {'params': model.feat_visual.parameters()},
        {'params': model.feat_semantic.parameters()}
    ], weight_decay=1e-3, lr=1e-3)
    """
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5, nesterov=True)
    # lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (1+(epoch/n_epochs))**0.75) 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Model config
    model.set_domain(0) # Train on source
    for param in model.extractor.parameters(): # Freeze extractor
        param.requires_grad = False

    print('training on device: {}'.format(device))
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

            y_pred, soft_y_pred = model.forward(x)
            cross_entropy_loss = nn.CrossEntropyLoss()
            loss = cross_entropy_loss(soft_y_pred, y)

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

            y_pred, soft_y_pred = model.forward(x)
            loss = cross_entropy_loss(soft_y_pred, y)

            minibatch += 1
            mean_loss = mean_loss + (loss - mean_loss) / minibatch if minibatch > 0 else loss
            acc = torch.sum(y == y_pred).float() / len(y_pred)
            mean_acc = mean_acc + (acc - mean_acc) / minibatch if minibatch > 0 else acc
        print('Epoch: {}, val_loss: {}, val_acc: {}'.format(epoch, mean_loss.item(), mean_acc.item()))




def train_officehome():

    """
    Validation error very noisy as we don't have that many examples per class
    in the first place, and the test set is very small.

    Epoch: 0, train_loss: 2.1045565605163574, train_acc: 0.46153849363327026
    Epoch: 0, val_loss: 2.170750856399536, val_acc: 0.23529411852359772
    Epoch: 1, train_loss: 1.604364275932312, train_acc: 0.5576923489570618
    Epoch: 1, val_loss: 2.182983875274658, val_acc: 0.4117647111415863
    Epoch: 2, train_loss: 1.7322044372558594, train_acc: 0.5192307829856873
    Epoch: 2, val_loss: 2.130432605743408, val_acc: 0.29411765933036804
    Epoch: 3, train_loss: 0.9551757574081421, train_acc: 0.75
    Epoch: 3, val_loss: 2.3338825702667236, val_acc: 0.47058823704719543
    Epoch: 4, train_loss: 1.285475254058838, train_acc: 0.634615421295166
    Epoch: 4, val_loss: 2.2644689083099365, val_acc: 0.4117647111415863
    Epoch: 5, train_loss: 1.495586633682251, train_acc: 0.5769230723381042
    Epoch: 5, val_loss: 2.1218161582946777, val_acc: 0.47058823704719543
    Epoch: 6, train_loss: 1.0011026859283447, train_acc: 0.75
    Epoch: 6, val_loss: 2.344646692276001, val_acc: 0.3529411852359772
    Epoch: 7, train_loss: 0.8445066809654236, train_acc: 0.75
    Epoch: 7, val_loss: 1.7455739974975586, val_acc: 0.529411792755127
    Epoch: 8, train_loss: 1.4034640789031982, train_acc: 0.6538462042808533
    Epoch: 8, val_loss: 2.337390899658203, val_acc: 0.529411792755127
    Epoch: 9, train_loss: 0.9853366017341614, train_acc: 0.7115384936332703
    Epoch: 9, val_loss: 1.5394541025161743, val_acc: 0.6470588445663452
    Epoch: 10, train_loss: 1.3720628023147583, train_acc: 0.634615421295166
    Epoch: 10, val_loss: 2.39518404006958, val_acc: 0.47058823704719543
    Epoch: 11, train_loss: 1.0457422733306885, train_acc: 0.75
    Epoch: 11, val_loss: 0.9711345434188843, val_acc: 0.7058823704719543
    Epoch: 12, train_loss: 0.7154629230499268, train_acc: 0.75
    Epoch: 12, val_loss: 1.1216915845870972, val_acc: 0.7647058963775635
    Epoch: 13, train_loss: 0.6426477432250977, train_acc: 0.7692307829856873
    Epoch: 13, val_loss: 1.4780492782592773, val_acc: 0.7647058963775635
    Epoch: 14, train_loss: 0.5531157851219177, train_acc: 0.826923131942749
    Epoch: 14, val_loss: 1.5254217386245728, val_acc: 0.6470588445663452
    Epoch: 15, train_loss: 1.0691739320755005, train_acc: 0.7307692766189575
    Epoch: 15, val_loss: 2.264802932739258, val_acc: 0.47058823704719543
    Epoch: 16, train_loss: 0.7412697076797485, train_acc: 0.7307692766189575
    Epoch: 16, val_loss: 1.8852485418319702, val_acc: 0.529411792755127
    Epoch: 17, train_loss: 0.9330698847770691, train_acc: 0.692307710647583
    Epoch: 17, val_loss: 1.906640887260437, val_acc: 0.5882353186607361
    Epoch: 18, train_loss: 0.6543956995010376, train_acc: 0.8076923489570618
    Epoch: 18, val_loss: 1.7633739709854126, val_acc: 0.5882353186607361
    Epoch: 19, train_loss: 0.7804287672042847, train_acc: 0.75
    Epoch: 19, val_loss: 3.011734962463379, val_acc: 0.4117647111415863
    Epoch: 20, train_loss: 0.5154317021369934, train_acc: 0.8076923489570618
    Epoch: 20, val_loss: 1.020255446434021, val_acc: 0.7647058963775635
    Epoch: 21, train_loss: 0.41287606954574585, train_acc: 0.8653846383094788
    Epoch: 21, val_loss: 1.08881413936615, val_acc: 0.5882353186607361
    Epoch: 22, train_loss: 0.3803752064704895, train_acc: 0.884615421295166
    Epoch: 22, val_loss: 1.8894821405410767, val_acc: 0.5882353186607361
    Epoch: 23, train_loss: 0.4629223942756653, train_acc: 0.826923131942749
    Epoch: 23, val_loss: 2.1499507427215576, val_acc: 0.5882353186607361
    Epoch: 24, train_loss: 0.4160309135913849, train_acc: 0.8461538553237915
    Epoch: 24, val_loss: 2.180814743041992, val_acc: 0.529411792755127
    Epoch: 25, train_loss: 0.6183852553367615, train_acc: 0.7692307829856873
    Epoch: 25, val_loss: 2.0254199504852295, val_acc: 0.529411792755127
    Epoch: 26, train_loss: 0.5028316974639893, train_acc: 0.9230769872665405
    Epoch: 26, val_loss: 2.122297763824463, val_acc: 0.6470588445663452
    Epoch: 27, train_loss: 0.41770315170288086, train_acc: 0.9230769872665405
    Epoch: 27, val_loss: 1.4673168659210205, val_acc: 0.6470588445663452
    Epoch: 28, train_loss: 0.5833584666252136, train_acc: 0.7884615659713745
    Epoch: 28, val_loss: 1.819322943687439, val_acc: 0.529411792755127
    Epoch: 29, train_loss: 0.5923662781715393, train_acc: 0.826923131942749
    Epoch: 29, val_loss: 1.5179270505905151, val_acc: 0.7058823704719543
    Epoch: 30, train_loss: 0.30948689579963684, train_acc: 0.9230769872665405
    Epoch: 30, val_loss: 2.770216464996338, val_acc: 0.47058823704719543
    Epoch: 31, train_loss: 0.7050920724868774, train_acc: 0.7884615659713745
    Epoch: 31, val_loss: 0.5349208116531372, val_acc: 0.8823529481887817
    Epoch: 32, train_loss: 0.2787531614303589, train_acc: 0.9230769872665405
    Epoch: 32, val_loss: 1.2750859260559082, val_acc: 0.7058823704719543
    Epoch: 33, train_loss: 0.43062761425971985, train_acc: 0.8461538553237915
    Epoch: 33, val_loss: 3.218092918395996, val_acc: 0.3529411852359772
    Epoch: 34, train_loss: 0.5603920817375183, train_acc: 0.8076923489570618
    Epoch: 34, val_loss: 2.07070255279541, val_acc: 0.47058823704719543
    Epoch: 35, train_loss: 0.8086485266685486, train_acc: 0.7692307829856873
    Epoch: 35, val_loss: 1.338321328163147, val_acc: 0.47058823704719543
    Epoch: 36, train_loss: 0.40635982155799866, train_acc: 0.884615421295166
    Epoch: 36, val_loss: 2.198305368423462, val_acc: 0.5882353186607361
    Epoch: 37, train_loss: 0.6418908834457397, train_acc: 0.8076923489570618
    Epoch: 37, val_loss: 3.1480724811553955, val_acc: 0.4117647111415863
    Epoch: 38, train_loss: 0.42835667729377747, train_acc: 0.9038462042808533
    Epoch: 38, val_loss: 1.4505081176757812, val_acc: 0.47058823704719543
    Epoch: 39, train_loss: 0.5038093328475952, train_acc: 0.8653846383094788
    Epoch: 39, val_loss: 1.9349818229675293, val_acc: 0.6470588445663452
    """

    train_transform = transforms.Compose([
        transforms.Resize((255,255)),
        transforms.RandomCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((255,255)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])

    # Parameters

    batch_size = 64
    n_epochs = 40

    num_workers = 0 if os.name == 'nt' else 12

    dataset = OfficeHomeDataset(transform=train_transform, splits=[('train',9), ('test',1)]).set_active_domain(1)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    model = Model(num_classes=dataset.num_classes, num_domains=2, label_names=dataset.idx2class,
                domain_names=dataset.idx2domain, base_model='resnet18', feature_depth=dataset.num_classes)
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print('training on device: {}'.format(device))
    for epoch in range(n_epochs):
        # Any variables for tracking state etc need to be tensors on device,
        # so everything else doesn't wait for the transfer to happen.
        minibatch = torch.zeros((1), dtype=torch.float32, device=device)
        mean_loss = torch.zeros((1), dtype=torch.float32, device=device)
        mean_acc = torch.zeros((1), dtype=torch.float32, device=device)

        dataset.set_active_split('train')
        for _, data in enumerate(loader):
            mean_loss, mean_acc, minibatch = 0, 0, 0

            x = data[0].to(device, non_blocking=True)
            y = data[1].to(device, non_blocking=True)
            d = data[2]

            y_pred, soft_y_pred = model.forward(x)
            cross_entropy_loss = nn.CrossEntropyLoss()
            loss = cross_entropy_loss(soft_y_pred, y)

            optimizer.zero_grad() # Reset gradient
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()

            minibatch += 1
            mean_loss = mean_loss + (loss - mean_loss) / minibatch if minibatch > 0 else loss
            acc = torch.sum(y == y_pred).float() / len(y_pred)
            mean_acc = mean_acc + (acc - mean_acc) / minibatch if minibatch > 0 else acc
        print('Epoch: {}, train_loss: {}, train_acc: {}'.format(epoch, mean_loss.item(), mean_acc.item()))
    
        dataset.set_active_split('test')
        for _, data in enumerate(loader):
            mean_loss, mean_acc, minibatch = 0, 0, 0

            x = data[0].to(device, non_blocking=True)
            y = data[1].to(device, non_blocking=True)
            d = data[2]

            y_pred, soft_y_pred = model.forward(x)
            loss = cross_entropy_loss(soft_y_pred, y)

            minibatch += 1
            mean_loss = mean_loss + (loss - mean_loss) / minibatch if minibatch > 0 else loss
            acc = torch.sum(y == y_pred).float() / len(y_pred)
            mean_acc = mean_acc + (acc - mean_acc) / minibatch if minibatch > 0 else acc
        print('Epoch: {}, val_loss: {}, val_acc: {}'.format(epoch, mean_loss.item(), mean_acc.item()))
    


if __name__ == '__main__':

    # train_mnist()

    train_officehome()
