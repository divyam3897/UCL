import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter, knn_monitor
from datasets import get_dataset
from models.optimizers import get_optimizer, LR_Scheduler
from utils.loggers import *


def evaluate_single(model, dataset, test_loader, memory_loader, device, k, last=False) -> Tuple[list, list, list, list]:
    accs, accs_mask_classes = [], []
    knn_accs, knn_accs_mask_classes = [], []
    correct = correct_mask_classes = total = 0
    knn_acc, knn_acc_mask = knn_monitor(model.net.module.backbone, dataset, memory_loader, test_loader, device, args.cl_default, task_id=k, k=min(args.train.knn_k, len(dataset.memory_loaders[k].dataset))) 

    return knn_acc


def main(device, args):

    dataset_copy = get_dataset(args)
    train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(args)
    model = get_model(args, device, len(train_loader), get_aug(train=False, train_classifier=False, **args.aug_kwargs))

    for t in range(dataset_copy.N_TASKS - 1):
        _, _, _ = dataset_copy.get_data_loaders(args)

    knn_acc = []
    for t in tqdm(range(0, dataset_copy.N_TASKS), desc='Evaluatinng'):
      dataset = get_dataset(args)
      model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}_{args.name}_{t}.pth")
      save_dict = torch.load(model_path, map_location='cpu')

      msg = model.net.module.backbone.load_state_dict({k[16:]:v for k, v in save_dict['state_dict'].items() if 'backbone.' in k}, strict=True)
      model = model.to(args.device)
    
      task_knn_acc = []
      for t1 in tqdm(range(0, dataset_copy.N_TASKS), desc='Inner tasks'):
        train_loader, memory_loader, test_loader = dataset.get_data_loaders(args)
        t1_knn_acc = evaluate_single(model, dataset, test_loader, memory_loader, device, t1)
        task_knn_acc.append(t1_knn_acc)
      knn_acc.append(task_knn_acc)
      print(f'Task {t}: {task_knn_acc}')
    
    mean_knn_acc = sum(knn_acc[-1][:len(knn_acc[-1])]) / len(knn_acc[-1])
    print(f'KNN accuracy on Task {t1}: {mean_knn_acc}')

    max_knn_acc = [max(idx) for idx in zip(*knn_acc)]
    mean_knn_fgt = sum([x1 - x2 for (x1, x2) in zip(max_knn_acc, knn_acc[-1])]) / len(knn_acc[-1])
    print(f'KNN Forgetting: {mean_knn_fgt}')


if __name__ == "__main__":
    args = get_args()
    main(device=args.device, args=args)
