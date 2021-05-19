import os
from shutil import copyfile
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, random_split, DataLoader



def make_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

"""
Cette fonction sépare les images de CUB200 en un jeu d'entraînement et de test.

dataset_path: Path où se trouve les images de CUB200
train_path: path où sauvegarder le jeu d'entraînement
test_path: path où sauvegarder le jeu de test
"""


def separate_train_test(dataset_path, train_path, test_path):

    class_index = 1
    for classname in sorted(os.listdir(dataset_path)):
        if classname.startswith('.'):
            continue
        make_dir(os.path.join(train_path, classname))
        make_dir(os.path.join(test_path, classname))
        i = 0
        for file in sorted(os.listdir(os.path.join(dataset_path, classname))):
            if file.startswith('.'):
                continue
            file_path = os.path.join(dataset_path, classname, file)
            if i < 15:
                copyfile(file_path, os.path.join(test_path, classname, file))
            else:
                copyfile(file_path, os.path.join(train_path, classname, file))
            i += 1

        class_index += 1





    



    

    






