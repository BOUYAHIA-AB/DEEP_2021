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
from torch.utils.data import DataLoader


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


if __name__ == "__main__":
    mnist, mnist_test = load_mnist()
    
    mnist.transform = ToTensor()
    mnist_test.transform = ToTensor()

    epoch = 50
    batch_size = 64
    learning_rate = 0.001


    train_loader = DataLoader(mnist, batch_size=batch_size)
    test_loader = DataLoader(mnist_test, batch_size=batch_size)

    net = MnistNet()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = pt.ReduceLROnPlateau(monitor='acc', mode='max', patience=1, factor=0.5, verbose=True)

    model = pt.Model(net, optimizer, 'cross_entropy', batch_metrics=['accuracy'])
    history = model.fit_generator(train_loader, test_loader, epochs=epoch, callbacks=[scheduler])

    History(history).display()

    test_loss, test_acc = model.evaluate_generator(test_loader)
    print('test_loss: {:.4f} test_acc: {:.2f}'.format(test_loss, test_acc))



    
    
    
    
