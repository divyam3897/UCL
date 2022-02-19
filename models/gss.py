import torch
from utils.gss_buffer import Buffer as Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Gradient based sample selection'
                                        'for online continual learning')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--batch_num', type=int, required=True,
                        help='Number of batches extracted from the buffer.')
    parser.add_argument('--gss_minibatch_size', type=int, default=None,
                        help='The batch size of the gradient comparison.')
    return parser


class Gss(ContinualModel):
    NAME = 'gss'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, len_train_loader, transform):
        super(Gss, self).__init__(backbone, loss, args, len_train_loader, transform)
        self.buffer = Buffer(self.args.model.buffer_size, self.device,
                            self.args.train.batch_size, self)
        self.alj_nepochs = 1  # batch_num parameter

    def get_grads(self, inputs, labels):
        self.net.eval()
        self.opt.zero_grad()
        labels = labels.to(self.device)
        outputs = self.net.module.backbone(inputs.to(self.device))
        loss = self.loss(outputs, labels)
        loss.backward()
        grads = self.net.module.backbone.get_grads().clone().detach()
        self.opt.zero_grad()
        self.net.train()
        if len(grads.shape) == 1:
            grads = grads.unsqueeze(0)
        return grads

    def observe(self, inputs1, labels, inputs2, notaug_inputs):

        real_batch_size = inputs1.shape[0]
        self.buffer.drop_cache()
        self.buffer.reset_fathom()
        labels = labels.to(self.device)

        for _ in range(self.alj_nepochs):
            self.opt.zero_grad()
            if not self.buffer.is_empty():
                buf_inputs, buf_labels = self.buffer.get_data(
                    self.args.train.batch_size, transform=self.transform)
                tinputs = torch.cat((inputs1.to(self.device), buf_inputs))
                tlabels = torch.cat((labels, buf_labels))
            else:
                tinputs = inputs1.to(self.device)
                tlabels = labels

            outputs = self.net.module.backbone(tinputs)
            loss = self.loss(outputs, tlabels)
            loss.backward()
            self.opt.step()

        self.buffer.add_data(examples=notaug_inputs,
                             labels=labels[:real_batch_size])
        data_dict = {'loss': loss, 'penalty': 0}
        data_dict.update({'lr': self.args.train.base_lr})

        return data_dict

