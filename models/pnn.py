# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.optim as optim
from torch.optim import SGD
import torch
import torch.nn as nn
from utils.conf import get_device
from utils.args import *
from datasets import get_dataset
from .utils.continual_model import ContinualModel
from .optimizers import get_optimizer, LR_Scheduler


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


def get_backbone(bone, old_cols=None, x_shape=None):
    from .backbones import resnet18
    from .backbones import resnet18_pnn
    print(bone)

    return resnet18_pnn(bone.n_classes, bone.inplanes, old_cols, x_shape)


class Pnn(ContinualModel):
    NAME = 'pnn'
    COMPATIBILITY = ['task-il']

    def __init__(self, backbone, loss, args, len_train_lodaer, transform):
        super(Pnn, self).__init__(backbone, loss, args, len_train_lodaer, transform)
        self.loss = loss
        self.args = args
        self.transform = transform
        self.device = get_device()
        self.x_shape = None
        self.nets = [self.net]
        self.net = self.nets[-1]

        self.soft = torch.nn.Softmax(dim=0)
        self.logsoft = torch.nn.LogSoftmax(dim=0)
        self.dataset = get_dataset(args)
        self.task_idx = 0

    def forward(self, x, task_label):
        if self.x_shape is None:
            self.x_shape = x.shape

        if self.task_idx == 0:
            out = self.net(x)
        else:
            self.nets[task_label].to(self.device)
            out = self.nets[task_label](x)
            if self.task_idx != task_label:
                self.nets[task_label].cpu()
        return out

    def end_task(self, dataset):
        # instantiate new column
        if self.task_idx == 4:
            return
        self.task_idx += 1
        self.nets[-1].cpu()
        self.nets.append(self.net)
        self.net = self.nets[-1]
        self.opt = get_optimizer(
            self.args.train.optimizer.name, self.net, 
            lr=self.args.train.base_lr*self.args.train.batch_size/256, 
            momentum=self.args.train.optimizer.momentum,
            weight_decay=self.args.train.optimizer.weight_decay)
        
    def observe(self, inputs1, labels, inputs2, notaug_inputs):
        if self.x_shape is None:
            self.x_shape = inputs1.shape

        self.opt.zero_grad()
        if self.args.cl_default:
            outputs = self.net.module.backbone(inputs1)
            loss = self.loss(outputs, labels)
            data_dict = {'loss': loss, 'penalty': 0.0}
        else:
            data_dict = self.net.forward(inputs1.to(self.device, non_blocking=True), inputs2.to(self.device, non_blocking=True))
            data_dict['loss'] = data_dict['loss'][0].mean()
            data_dict['penalty'] = 0.0
            loss = data_dict['loss']
        loss.backward()
        self.opt.step()
        data_dict.update({'lr': self.args.train.base_lr})

        return data_dict
