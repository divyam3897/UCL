# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.optim import SGD
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
import numpy as np
from ..optimizers import get_optimizer, LR_Scheduler


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
            args: Namespace, len_train_lodaer, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.net = nn.DataParallel(self.net)
        self.loss = loss
        self.args = args
        self.transform = transform
        
        self.opt = get_optimizer(
            args.train.optimizer.name, self.net, 
            lr=args.train.base_lr*args.train.batch_size/256, 
            momentum=args.train.optimizer.momentum,
            weight_decay=args.train.optimizer.weight_decay)
        
        self.lr_scheduler = LR_Scheduler(
            self.opt,
            args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256, 
            args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256, 
            len_train_lodaer,
            constant_predictor_lr=True # see the end of section 4.2 predictor
        )
        self.device = get_device()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net.module.backbone.forward(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass
