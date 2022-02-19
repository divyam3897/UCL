from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from augmentations import get_aug

class Finetune(ContinualModel):
    NAME = 'finetune'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, len_train_loader, transform):
        super(Finetune, self).__init__(backbone, loss, args, len_train_loader, transform)

    def observe(self, inputs1, labels, inputs2, notaug_inputs):

        self.opt.zero_grad()
        if self.args.cl_default:
            labels = labels.to(self.device)
            outputs = self.net.module.backbone(inputs1.to(self.device))
            loss = self.loss(outputs, labels).mean()
            data_dict = {'loss': loss}
            data_dict['penalty'] = 0.0
        else:
            data_dict = self.net.forward(inputs1.to(self.device, non_blocking=True), inputs2.to(self.device, non_blocking=True))
            data_dict['loss'] = data_dict['loss'].mean()
            loss = data_dict['loss']
            data_dict['penalty'] = 0.0

        loss.backward()
        self.opt.step()
        data_dict.update({'lr': self.args.train.base_lr})

        return data_dict
