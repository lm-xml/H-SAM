import math
from datasets.autoaug import CIFAR10Policy, Cutout
import random
import time
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from datasets.cifar10 import CIFAR10_LT
from datasets.cifar100 import CIFAR100_LT
import models
from datasets.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from models.losses import LDAMLoss, FocalLoss, VSLoss, LALoss
from scipy.interpolate import interp1d
from H_SAM import H_SAM, FocalSAM, SAM
from H_SAM_train_step import H_SAM_step, focal_sam_step, SAM_step
from sklearn.metrics import confusion_matrix
import shutil

import argparse
import os
from utils import config, update_config, create_logger
from utils import AverageMeter, ProgressMeter
from utils import accuracy, calibration
from utils.utils import *
import pprint
import torch.nn.functional as F
from collections import defaultdict
import numpy as np

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='path to config yaml',
                        default='/configs/', type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    file_name = os.path.abspath(__file__)
    parser.add_argument('--file_name', default=file_name, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


best_acc1 = 0


def main():
    args = parse_args()
    logger, model_dir = create_logger(config, args.cfg, args.file_name)
    logger.info('\n' + pprint.pformat(args))
    logger.info('\n' + str(config))

    if config.seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    if config.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)

    main_worker(config.gpu, config, logger, model_dir)


def main_worker(gpu, config, logger, model_dir):
    global best_acc1
    config.gpu = gpu
    if config.gpu is not None:
        logger.info("Use GPU: {} for training".format(config.gpu))

    # create model
    print("=> creating model '{}'".format(config.arch))
    num_classes = 100 if config.dataset == 'cifar100' else 10
    use_norm = True if config.loss_type == 'LDAM' else False
    model = models.__dict__[config.arch](num_classes=num_classes, use_norm=use_norm)

    model = model.cuda()
    base_optimizer = torch.optim.SGD

    if config.dataset == 'cifar10':
        data_loader = CIFAR10_LT(root=config.data_path, imb_factor=config.imb_factor,
                                 batch_size=config.batch_size, num_works=config.workers)
        train_loader = data_loader.train_instance
        val_loader = data_loader.eval
        config.cls_num_list = data_loader.cls_num_list
    elif config.dataset == 'cifar100':
        data_loader = CIFAR100_LT(root=config.data_path, imb_factor=config.imb_factor,
                                  batch_size=config.batch_size, num_works=config.workers)
        train_loader = data_loader.train_instance
        val_loader = data_loader.eval
        config.cls_num_list = data_loader.cls_num_list
    else:
        warnings.warn('Dataset is not listed')
        return
    print('cls num list:')
    print(config.cls_num_list)


    if config.SAM_type == 'H_SAM':
        optimizer = H_SAM(params=model.parameters(), base_optimizer=base_optimizer, rho=config.rho, adaptive=False,
                    lmbda=config.lmbda, lr=config.lr, momentum=config.momentum,
                    weight_decay=config.weight_decay)
    elif config.SAM_type == 'Focal-SAM':
        optimizer = FocalSAM(params=model.parameters(), base_optimizer=base_optimizer, rho=config.rho, adaptive=False,
                             lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.SAM_type == 'SAM':
        optimizer = SAM(base_optimizer=base_optimizer, rho=config.rho, params=model.parameters(), lr=config.lr,
                        momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum,
                                    weight_decay=config.weight_decay)

    # optionally resume from a checkpoint
    if config.resume:
        if os.path.isfile(config.resume):
            logger.info("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume, map_location='cuda:0')
            config.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if config.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.cuda()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(config.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.resume))

    few_shot_class = []
    medium_shot_class = []
    many_shot_class = []

    if config.dataset == 'cifar10':
        if config.imb_factor == 0.01:
            few_judge = 200
            medium_judge = 1000
        elif config.imb_factor == 0.1:
            medium_judge = 2319
            few_judge = 835
        else:
            few_judge = 200
            medium_judge = 1000
    elif config.dataset == 'cifar100':
        if config.imb_factor == 0.01:
            few_judge = 20
            medium_judge = 100
        elif config.imb_factor == 0.1:
            few_judge = 99
            medium_judge = 265
        else:
            few_judge = 20
            medium_judge = 100
    else:
        raise ValueError('Dataset is not listed')

    for idx, cls_num in enumerate(config.cls_num_list):
        if cls_num < few_judge:
            few_shot_class.append(idx)
        elif cls_num <= medium_judge:
            medium_shot_class.append(idx)
        else:
            many_shot_class.append(idx)
    config.head_class_idx = many_shot_class
    config.med_class_idx = medium_shot_class
    config.tail_class_idx = few_shot_class

    for epoch in range(config.start_epoch, config.epochs):


        adjust_learning_rate(optimizer, epoch, config)

        adjust_rho(optimizer, epoch, config)


        if config.train_rule == None:
            train_sampler = None
            per_cls_weights = None
        elif config.train_rule == 'Reweight':
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, config.cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(config.cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        elif config.train_rule == 'DRW':
            train_sampler = None
            idx = epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], config.cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(config.cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        else:
            warnings.warn('Sample rule is not listed')

        if config.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights, reduction='none').cuda()
        elif config.loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=config.cls_num_list, max_m=0.5, s=30, weight=per_cls_weights,
                                 reduction='none').cuda()
        elif config.loss_type == 'VS':
            criterion = VSLoss(cls_num_list=config.config.cls_num_list, tau=config.tau, gamma=config.gamma,
                               weight=per_cls_weights, reduction='none').cuda()
        elif config.loss_type == 'LA':
            criterion = LALoss(cls_num_list=config.cls_num_list, reduction='none').cuda()
        else:
            warnings.warn('Loss type is not listed')
            return

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, config, few_shot_class, medium_shot_class,
              many_shot_class, config.cls_num_list, logger, optimizer.param_groups[0]['lr'])

        # evaluate on validation set
        is_best = validate(val_loader, model, config, logger, optimizer.param_groups[0]['lr'])
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': config.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, model_dir)

def train(train_loader, model, criterion, optimizer, epoch, config, few_shot_class, medium_shot_class,
          many_shot_class, cls_num_list, logger, lr):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader), [lr, config.rho],
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    all_preds = []
    all_targets = []
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if config.gpu is not None:
            input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if config.SAM_type == "H_SAM":
            output, loss = H_SAM_step(config, epoch, model, criterion, input, target, optimizer, cls_num_list)
        elif config.SAM_type == "Focal-SAM":
            output, loss = focal_sam_step(config, model, criterion, input, target, optimizer, cls_num_list)
        elif config.SAM_type == "SAM":
            output, loss = SAM_step(model, criterion, input, target, optimizer)
        else:
            output = model(input)
            loss = criterion(output, target)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        _, pred = torch.max(output, 1)
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.display(i, logger)

    if epoch % config.split == 0:
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        sorted_indices = np.argsort(-cls_acc)
        top_indices = sorted_indices[:len(cls_acc)]
        config.class_idx = top_indices.tolist()
        print(config.class_idx)


best_acc1 = defaultdict(float)

def validate(val_loader, model, config, logger, lr):
    batch_time = AverageMeter('Time', ':6.3f')
    acc_meter = {
        'classifier': AccMeter(),}
    progress = ProgressMeter(
        len(val_loader), [lr, config.rho],
        [batch_time, acc_meter['classifier'].top1, acc_meter['classifier'].top5],
        prefix='Eval: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if config.gpu is not None:
                input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            # measure accuracy and record loss
            acc_meter['classifier'].update(output, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                progress.display(i, logger)

        global best_acc1
        is_classifier_best = False

        for name in acc_meter.keys():
            entry = acc_meter[name]

            acc1, acc5 = entry.top1.avg, entry.top5.avg
            head_acc, med_acc, tail_acc = entry.get_shot_acc()

            # remember best acc@1
            is_best = acc1 > best_acc1[name]
            if is_best:
                best_acc1[name] = acc1
                if name == 'classifier':
                    is_classifier_best = True

            logger.info(('* ({name})  Acc@1 {acc1:.3f}  HAcc {head_acc:.3f}  MAcc {med_acc:.3f}  TAcc {tail_acc:.3f}  '
                         '(Best Acc@1 {best_acc1:.3f}).').format(
                name=name, acc1=acc1, acc5=acc5, head_acc=head_acc, med_acc=med_acc, tail_acc=tail_acc,
                best_acc1=best_acc1[name]))

    return is_classifier_best

def save_checkpoint(state, is_best, model_dir):
    filename = model_dir + '/current.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_dir + '/model_best.pth.tar')

class AccMeter:
    def __init__(self):
        self.top1 = AverageMeter('Acc@1', ':6.3f')
        self.top5 = AverageMeter('Acc@5', ':6.3f')

        self.class_num = torch.zeros(config.num_classes).cuda()
        self.correct = torch.zeros(config.num_classes).cuda()

        self.confidence = np.array([])
        self.pred_class = np.array([])
        self.true_class = np.array([])

    def update(self, output, target, is_prob=False):
        if not is_prob:
            output = torch.softmax(output, dim=1)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        self.top1.update(acc1[0], target.size(0))
        self.top5.update(acc5[0], target.size(0))

        _, predicted = output.max(1)
        target_one_hot = F.one_hot(target, config.num_classes)
        predict_one_hot = F.one_hot(predicted, config.num_classes)
        self.class_num = self.class_num + target_one_hot.sum(dim=0).to(torch.float)
        self.correct = self.correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

        confidence_part, pred_class_part = torch.max(output, dim=1)
        self.confidence = np.append(self.confidence, confidence_part.cpu().numpy())
        self.pred_class = np.append(self.pred_class, pred_class_part.cpu().numpy())
        self.true_class = np.append(self.true_class, target.cpu().numpy())

    def get_shot_acc(self):
        acc_classes = self.correct / self.class_num
        head_acc = acc_classes[config.head_class_idx].mean() * 100
        med_acc = acc_classes[config.med_class_idx].mean() * 100
        tail_acc = acc_classes[config.tail_class_idx].mean() * 100
        return head_acc, med_acc, tail_acc

    def get_cal(self):
        cal = calibration(self.true_class, self.pred_class, self.confidence, num_bins=15)
        return cal

def adjust_learning_rate(optimizer, epoch, config):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if config.cos_lr:
        lr_min = 0
        lr_max = config.lr
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / config.epochs * 3.1415926535))
        warmup_epochs = 5
        if epoch < warmup_epochs:
            lr = config.lr / warmup_epochs * (epoch + 1)
    else:
        epoch = epoch + 1
        if epoch <= 5:
            lr = config.lr * epoch / 5
        elif epoch > 180:
            lr = config.lr * 0.0001
        elif epoch > 160:
            lr = config.lr * 0.01
        else:
            lr = config.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_rho(optimizer, epoch, config):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if config.rho_schedule == 'step':
        if epoch <= 5:
            rho = 0.05
        elif epoch > 180:
            rho = 0.6
        elif epoch > 160:
            rho = 0.5
        else:
            # rho = 0.1
            rho = 0.2
        for param_group in optimizer.param_groups:
            param_group['rho'] = rho
    if config.rho_schedule == 'linear':
        X = [1, config.epochs]
        Y = [config.min_rho, config.max_rho]
        y_interp = interp1d(X, Y)
        rho = y_interp(epoch)

        for param_group in optimizer.param_groups:
            param_group['rho'] = np.float16(rho)

    if config.rho_schedule == 'none':
        rho = config.rho
        for param_group in optimizer.param_groups:
            param_group['rho'] = rho


if __name__ == '__main__':
    main()
