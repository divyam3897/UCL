# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision.datasets import SVHN
import torchvision.transforms as transforms
import torch.nn.functional as F
from datasets.seq_tinyimagenet import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_label_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
import torch
from augmentations import get_aug
from PIL import Image


class SequentialSVHN(ContinualDataset):

    NAME = 'seq-svhn'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 1
   
    def get_data_loaders(self, args):
        transform = get_aug(train=True, **args.aug_kwargs)
        test_transform = get_aug(train=False, train_classifier=False, **args.aug_kwargs)

        train_dataset = SVHN(base_path() + 'SVHN', split='train',
                                  download=True, transform=transform)
        
        memory_dataset = SVHN(base_path() + 'SVHN', split='train',
                                  download=True, transform=test_transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset, test_transform, self.NAME)
            memory_dataset, _ = get_train_val(memory_dataset, test_transform, self.NAME)
        else:
            test_dataset = SVHN(base_path() + 'SVHN', split='test',
                                   download=True, transform=test_transform)

        train, memory, test = store_masked_label_loaders(train_dataset, test_dataset, memory_dataset, self)
        return train, memory, test
    
    def get_transform(self, args):
        svhn_norm = [[0.4377,0.4438,0.4728], [0.198,0.201,0.197]]
        transform = transforms.Compose(
                [transforms.ToPILImage(),
                transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*svhn_norm)
                ])
        return transform
