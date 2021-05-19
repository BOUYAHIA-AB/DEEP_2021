import os
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import poutyne as pt

plt.rcParams['figure.dpi'] = 150

from deeplib.datasets import load_mnist,  train_valid_loaders
from sklearn.metrics import accuracy_score
from deeplib.history import History
from deeplib.visualization import plot_images

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset

from question5a import porcentage_dead, SaveOutput


class MnistNet(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x) :
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def weights_init_uniform(net):
    for module in net.modules():
        if isinstance(module, nn.Linear):
            torch.nn.init.uniform_(module.weight, -1.0,1.0)
            torch.nn.init.zeros_(module.bias)

def weights_init_normal(net):
    for module in net.modules():
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, 0.0,1.0)
            torch.nn.init.zeros_(module.bias)

def weights_init_constant(net):
    for module in net.modules():
        if isinstance(module, nn.Linear):
            torch.nn.init.constant_(module.weight, 0.1)
            torch.nn.init.zeros_(module.bias)

def weights_init_xavier_normal(net):
    for module in net.modules():
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, 1.0)
            torch.nn.init.zeros_(module.bias)

def weights_init_kaiming_uniform(net):
    for module in net.modules():
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, 1.0)
            torch.nn.init.zeros_(module.bias)
            

def get_256_sample(dataset) :
    indices = torch.randperm(len(dataset))[:256]

    data = Subset(dataset, indices)

    return data


if __name__ == "__main__":
    mnist, mnist_test = load_mnist()
    
    mnist.transform = ToTensor()
    mnist_test.transform = ToTensor()

    dataset = get_256_sample(mnist)
    dataset_loader = DataLoader(dataset, batch_size=256)

    dataset = next(iter(dataset_loader))

    epoch = 20
    batch_size = 64
    learning_rate = 0.001

    
    train_loader, valid_loader = train_valid_loaders(mnist, batch_size=batch_size)
    test_loader = DataLoader(mnist_test, batch_size=batch_size)

    net = MnistNet()
        
    #weights_init_constant(net)
    #weights_init_uniform(net)
    #weights_init_normal(net)
    #weights_init_xavier_normal(net)
    weights_init_kaiming_uniform(net)


    porcentage_dead_neuron = porcentage_dead(net, dataset)
    print(porcentage_dead_neuron)

    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    os.makedirs('logs', exist_ok=True)

    checkpoint = pt.ModelCheckpoint('logs/best_epoch_{epoch}.ckpt', monitor='val_acc', mode='max', save_best_only=True, restore_best=True, verbose=True, temporary_filename='best_epoch.ckpt.tmp')
    scheduler = pt.ReduceLROnPlateau(monitor='val_acc', mode='max', patience=3, factor=0.5, verbose=True)
    earlystopping = pt.EarlyStopping(monitor='val_acc', mode='max', patience=5, verbose=True)
    callbacks = [checkpoint, scheduler, earlystopping]

    model = pt.Model(net, optimizer, 'cross_entropy', batch_metrics=['accuracy'])
    history = model.fit_generator(train_loader, valid_loader, epochs=epoch, callbacks=callbacks)

    porcentage_dead_neuron = porcentage_dead(net, dataset)
    print(porcentage_dead_neuron)


    test_loss, test_acc = model.evaluate_generator(test_loader)
    print('test_loss: {:.4f} test_acc: {:.2f}'.format(test_loss, test_acc))



    
    
    
    
