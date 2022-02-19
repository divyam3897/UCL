# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
import torch.nn.functional as F
from datasets.seq_tinyimagenet import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
import torch
from augmentations import get_aug
from PIL import Image


class SequentialFMNIST(ContinualDataset):

    NAME = 'seq-fmnist'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 1
   
    def get_data_loaders(self, args):
        mean=(0.2190,) # Mean and std including the padding
        std=(0.3318,)
        train_dataset = FashionMNIST(base_path() + 'FMNIST', train=True,
                                  download=True, transform=transforms.Compose([
                    transforms.Pad(padding=2,fill=0),transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3,1,1)),
                    transforms.Normalize(mean,std)]))
        
        memory_dataset = FashionMNIST(base_path() + 'FMNIST', train=True,
                                  download=True, transform=transforms.Compose([
                    transforms.Pad(padding=2,fill=0),transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3,1,1)),
                    transforms.Normalize(mean,std)]))

        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset, test_transform, self.NAME)
            memory_dataset, _ = get_train_val(memory_dataset, test_transform, self.NAME)
        else:
            test_dataset = FashionMNIST(base_path() + 'FMNIST',train=False,
                                   download=True, transform=transforms.Compose([
                    transforms.Pad(padding=2,fill=0),transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3,1,1)),
                    transforms.Normalize(mean,std)]))

        train, memory, test = store_masked_loaders(train_dataset, test_dataset, memory_dataset, self)
        return train, memory, test
    
