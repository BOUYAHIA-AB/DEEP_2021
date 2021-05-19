import os
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import poutyne as pt

from deeplib.datasets import load_mnist,  train_valid_loaders

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

"""
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
        x = F.softmax(x)

        return x
"""

class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []

def porcentage_dead(model : nn.Module, dataset) :

    save_output = SaveOutput()

    hook_handles = []

    for layer in model.modules():
        if isinstance(layer, torch.nn.modules.Linear):
            handle = layer.register_forward_hook(save_output)
            hook_handles.append(handle)

    # Si le datset est de type dataset pytorch : à utiliser si on veut tester cette méthode à partir de ce fichier

    #model_output = model(dataset.data.float())


    model_output = model(dataset[0])
    
    relu_out = [torch.sum(F.relu(tensor).detach(), 0) for tensor in save_output.outputs]
    porcentage_dead_neuron = [(len(tensor.tolist()) - len(torch.nonzero(tensor).tolist()))/len(tensor.tolist()) for tensor in relu_out]

    del porcentage_dead_neuron[-1]

    return porcentage_dead_neuron


"""
if __name__ == "__main__":
    mnist_train, mnist_test = load_mnist()
    
    mnist_test.transform = ToTensor()

    net = MnistNet()

    porcentage_dead_neuron = porcentage_dead(net, mnist_test)

    print(porcentage_dead_neuron)

"""












    
    
    
    
