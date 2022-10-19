import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from datasets import get_dataset
from datetime import datetime
from utils.loggers import *
from utils.metrics import mask_classes
from utils.loggers import CsvLogger
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from typing import Tuple


def evaluate(model: ContinualModel, dataset: ContinualDataset, device, classifier=None) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.training
    model.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if classifier is not None:
                outputs = classifier(outputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()
        
        accs.append(correct / total * 100)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.train(status)
    return accs, accs_mask_classes


def main(device, args):
    dataset = get_dataset(args)
    dataset_copy = get_dataset(args)
    train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(args)
    results = {'knn-cls-acc':[],
                'knn-cls-each-acc':[],
                'knn-cls-max-acc':[],
                'knn-cls-fgt':[],}

    # define model
    model = get_model(args, device, len(train_loader), dataset.get_transform(args))

    logger = Logger(matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    accuracy = 0 
    
    train_loaders, memory_loaders, test_loaders = [], [], []
    for t in range(dataset.N_TASKS):
      tr, me, te = dataset.get_data_loaders(args)
      train_loaders.append(tr)
      memory_loaders.append(me)
      test_loaders.append(te)

    for t in range(dataset.N_TASKS):
      # train_loader, memory_loader, test_loader = dataset.get_data_loaders(args)
      if args.eval.type == 'all':
          eval_tids = [j for j in range(dataset.N_TASKS)]
      elif args.eval.type == 'curr':
          eval_tids = [t]
      elif args.eval.type == 'accum':
          eval_tids = [j for j in range(t + 1)]
      else:
          sys.exit('Stopped!! Wrong eval-type.')

      global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
      for epoch in global_progress:
        model.train()
        
        local_progress=tqdm(train_loaders[t], desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        for idx, ((images1, images2, notaug_images), labels) in enumerate(local_progress):
            data_dict = model.observe(images1, labels, images2, notaug_images)
            logger.update_scalers(data_dict)

        global_progress.set_postfix(data_dict)

        # if args.train.knn_monitor and epoch % args.train.knn_interval == 0: 
        if (epoch + 1) == args.train.stop_at_epoch:
            # depend on args.eval.type
            if args.train.knn_monitor:
                knn_acc_list = []
                for i in eval_tids:
                    acc, acc_mask = knn_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i],
                                                device, args.cl_default, task_id=i, k=min(args.train.knn_k, len(eval_tids)))
                    knn_acc_list.append(acc)

                kfgt = []
                # memorize current task acc
                results['knn-cls-each-acc'].append(knn_acc_list[-1])
                results['knn-cls-max-acc'].append(knn_acc_list[-1])
                # memorize max accuracy
                for j in range(t):
                    if knn_acc_list[j] > results['knn-cls-max-acc'][j]:
                        results['knn-cls-max-acc'][j] = knn_acc_list[j]
                    kfgt.append(results['knn-cls-each-acc'][j] - knn_acc_list[j])
                results['knn-cls-acc'].append(np.mean(knn_acc_list))
                results['knn-cls-fgt'].append(np.mean(kfgt))

      model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}_{args.name}_{t}.pth")
      torch.save({
        'epoch': epoch+1,
        'state_dict':model.net.state_dict()
      }, model_path)
      print(f"Task Model saved to {model_path}")
      with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
        f.write(f'{model_path}')
      with open(os.path.join(f'{args.log_dir}', f"%s_accuracy_logs.txt"%args.name), 'w+') as f:
        f.write(str(results))
      if hasattr(model, 'end_task'):
        model.end_task(dataset)

    if args.eval is not False and args.cl_default is False:
        args.eval_from = model_path

if __name__ == "__main__":
    args = get_args()
    main(device=args.device, args=args)
    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')


