import torch
import torch.nn as nn
from utils.args import *
from .utils.continual_model import ContinualModel


class SI(ContinualModel):
    NAME = 'si'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, len_train_loader, transform):
        super(SI, self).__init__(backbone, loss, args, len_train_loader, transform)
        
        self.checkpoint = self.net.module.backbone.get_params().data.clone().to(self.device)
        self.big_omega = None
        self.small_omega = 0
        self.c = args.train.alpha
        self.xi = 1.0

    def penalty(self):
        if self.big_omega is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.big_omega * ((self.net.module.backbone.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def end_task(self, dataset):
        # big omega calculation step
        if self.big_omega is None:
            self.big_omega = torch.zeros_like(self.net.module.backbone.get_params()).to(self.device)

        self.big_omega += self.small_omega / ((self.net.module.backbone.get_params().data - self.checkpoint) ** 2 + self.xi)

        self.checkpoint = self.net.module.backbone.get_params().data.clone().to(self.device)
        self.small_omega = 0

    def observe(self, inputs1, labels, inputs2, notaug_inputs):
        self.opt.zero_grad()
        if self.args.cl_default:
            labels = labels.to(self.device)
            outputs = self.net.module.backbone(inputs1.to(self.device))
            penalty = self.c * self.penalty()
            loss = self.loss(outputs, labels).mean() + penalty
            data_dict = {'loss': loss, 'penalty': penalty}
        else:
            data_dict = self.net.forward(inputs1.to(self.device, non_blocking=True), inputs2.to(self.device, non_blocking=True))
            data_dict['penalty'] = self.c * self.penalty() 
            data_dict['loss'] = data_dict['loss'].mean()
            loss = data_dict['loss'] + data_dict['penalty']
            
        loss.backward()
        nn.utils.clip_grad.clip_grad_value_(self.net.parameters(), 1)
        self.opt.step()
        data_dict.update({'lr': self.args.train.base_lr})

        self.small_omega += self.args.train.base_lr * self.net.module.backbone.get_grads().data ** 2

        return data_dict
