import torch
import numpy as np
from utils.buffer import Buffer
from models.gem import overwrite_grad
from models.gem import store_grad
from models.utils.continual_model import ContinualModel


def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


class AGem(ContinualModel):
    NAME = 'agem'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, len_train_loader, transform):
        super(AGem, self).__init__(backbone, loss, args, len_train_loader, transform)

        self.buffer = Buffer(self.args.model.buffer_size, self.device)
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)

    def end_task(self, dataset):
        samples_per_task = self.args.model.buffer_size // dataset.N_TASKS
        loader = dataset.not_aug_dataloader(samples_per_task)
        cur_x, cur_y = next(iter(loader))
        self.buffer.add_data(
            examples=cur_x.to(self.device),
            labels=cur_y.to(self.device)
        )

    def observe(self, inputs1, labels, inputs2, notaug_inputs):

        self.zero_grad()
        labels = labels.to(self.device)
        p = self.net.module.backbone(inputs1.to(self.device))
        loss = self.loss(p, labels)
        loss.backward()
        data_dict = {'loss': loss, 'penalty': 0}

        if not self.buffer.is_empty():
            store_grad(self.parameters, self.grad_xy, self.grad_dims)

            buf_inputs, buf_labels = self.buffer.get_data(self.args.train.batch_size, transform=self.transform)
            self.net.zero_grad()
            buf_outputs = self.net.module.backbone(buf_inputs)
            penalty = self.loss(buf_outputs, buf_labels)
            penalty.backward()
            data_dict['penalty'] = penalty
            store_grad(self.parameters, self.grad_er, self.grad_dims)

            dot_prod = torch.dot(self.grad_xy, self.grad_er)
            if dot_prod.item() < 0:
                g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                overwrite_grad(self.parameters, g_tilde, self.grad_dims)
            else:
                overwrite_grad(self.parameters, self.grad_xy, self.grad_dims)

        self.opt.step()
        data_dict.update({'lr': self.args.train.base_lr})

        return data_dict

