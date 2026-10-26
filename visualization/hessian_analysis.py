import argparse
import random
import time
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from utils import *
from autoaug import CIFAR10Policy, Cutout
from hessian import hessian
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from util import config, update_config, accuracy, calibration, AverageMeter, ProgressMeter, create_logger
import pprint
import models
from datasets.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
import torchvision.datasets as datasets
from losses import LDAMLoss, FocalLoss, VSLoss, LALoss

def parse_args():
    parser = argparse.ArgumentParser(description='MiSLAS training (Stage-1)')
    parser.add_argument('--cfg',
                        default='/workspace/4-SAM/SAM/configs/cifar10/cifar10_imb001.yaml',
                        # default='/workspace/4-SAM/SAM/configs/cifar10/cifar10_imb001.yaml',
                        # default='/workspace/4-SAM/SAM/configs/cifar10/cifar10_imb001.yaml',
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    file_name = os.path.abspath(__file__)
    parser.add_argument('--file_name', default=file_name, type=str)
    args = parser.parse_args()
    update_config(config, args)  # logger.py中有很多共用的参数设置

    return args


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
    print("=> The GPU of model '{}'".format(config.gpu))

    # create model
    print("=> creating model '{}'".format(config.arch))
    num_classes = 100 if config.dataset == 'cifar100' else 10
    use_norm = True if config.loss_type == 'LDAM' else False
    model = models.__dict__[config.arch](num_classes=num_classes, use_norm=use_norm)

    # 载入你自己训练好的模型权重
    # weights_path = "/workspace/4-SAM/SAM/saved/model_best.pth.tar"
    weights_path = "/workspace/4-SAM/SAM/best_result/cifar_train_sam_5_1.py_cifar10_exp_0.01_LA_Tail_SAM_rho_schedule_none_seed_0_workers_8_64_cos_lr_True_resnet32_None/model_best.pth.tar"
    # weights_path = "/workspace/4-SAM/SAM/best_result/cifar_train_sam_5_1.py_cifar10_exp_0.01_LA_Tail_SAM_rho_schedule_none_seed_0_workers_8_64_cos_lr_True_resnet32_None/ckps/model_best_head.pth.tar"
    # weights_path = "/workspace/4-SAM/SAM/best_result/cifar_train_sam_5_1.py_cifar10_exp_0.01_LA_Tail_SAM_rho_schedule_none_seed_0_workers_8_64_cos_lr_True_resnet32_None/ckps/model_best_med.pth.tar"
    # weights_path = "/workspace/4-SAM/SAM/best_result/cifar_train_sam_5_1.py_cifar10_exp_0.01_LA_Tail_SAM_rho_schedule_none_seed_0_workers_8_64_cos_lr_True_resnet32_None/ckps/model_best_tail.pth.tar"
    save_dir = "/workspace/4-SAM/SAM/best_result/analysis_result/"
    save_dir = save_dir + config.dataset
    os.makedirs(save_dir, exist_ok=True)
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    weights = torch.load(weights_path, map_location='cpu')
    weights_model = weights["state_dict"]
    model.load_state_dict(weights_model)
    model.to(config.gpu)

    # Data loading code
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),  # add AutoAug
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if config.dataset == 'cifar10':
        train_dataset = IMBALANCECIFAR10(root=config.data_path, imb_type=config.imb_type, imb_factor=config.imb_factor,
                                         rand_number=config.rand_number, train=True, download=True,
                                         transform=transform_train)
        val_dataset = datasets.CIFAR10(root=config.data_path, train=False, download=True, transform=transform_val)
    elif config.dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(root=config.data_path, imb_type=config.imb_type, imb_factor=config.imb_factor,
                                          rand_number=config.rand_number, train=True, download=True,
                                          transform=transform_train)
        val_dataset = datasets.CIFAR100(root=config.data_path, train=False, download=True, transform=transform_val)

    train_sampler = config.train_sampler
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=(train_sampler is None),
        num_workers=config.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=config.workers, pin_memory=True)
    # hess_dataset = train_loader.dataset
    config.dataloader_hess = "val"
    hess_dataset = val_loader.dataset
    cls_num_list = []
    for i in range(config.num_classes):
        cls_num_list.append(hess_dataset.targets.count(i))
    per_cls_weights = None
    # criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(config.gpu)
    # criterion = FocalLoss(weight=per_cls_weights, gamma=2.0, reduction='mean')
    # criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(config.gpu)
    # criterion = LogitAdjust(cls_num_list=cls_num_list).cuda(config.gpu)
    criterion = LALoss(cls_num_list=cls_num_list, reduction='mean').cuda()
    all_trace = []
    all_eigenvalues = []
    print('clas_num_list',cls_num_list)


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

    for idx, cls_num in enumerate(cls_num_list):
        if cls_num < few_judge:
            few_shot_class.append(idx)
        elif cls_num <= medium_judge:
            medium_shot_class.append(idx)
        else:
            many_shot_class.append(idx)
    config.head_class_idx = many_shot_class
    config.med_class_idx = medium_shot_class
    config.tail_class_idx = few_shot_class


    for class_idx, num_samples in enumerate(cls_num_list):
        if config.dataloader_hess=="train":
            print(f"Number of samples in class{class_idx}: {num_samples}")
        cls_indices = [i for i in range(len(hess_dataset.targets)) if hess_dataset.targets[i]==class_idx]
        class_idx_dataset = torch.utils.data.Subset(hess_dataset, cls_indices)
        if config.dataloader_hess=="val":
            print(f"Number of samples in class{class_idx}: {len(class_idx_dataset)}")
        class_loader = torch.utils.data.DataLoader(
                class_idx_dataset, batch_size=config.batch_size, shuffle=False,
                num_workers=config.workers, pin_memory=True)
        class_loader = val_loader
        validate(class_loader, model, criterion, config, cls_num_list)

        hessian_comp = hessian(model, criterion, dataloader=class_loader, cuda=True)
        top_eigenvalues, _ = hessian_comp.eigenvalues()
        trace = hessian_comp.trace()
        density_eigen, density_weight = hessian_comp.density()
        # save_numpy(config, class_idx, density_eigen, density_weight)

        print("The type of density_eigen is ", type(density_eigen))
        print("The type of density_weight is ", type(density_weight))
        print("The density eigen ", density_eigen)
        print("The density weight ", density_weight)
        print('\n***Top Eigenvalues: ', top_eigenvalues)
        print('\n***Trace: ', np.mean(trace))
        all_trace.append(np.mean(trace))
        all_eigenvalues.append(top_eigenvalues[0])
        get_esd_plot(density_eigen, density_weight, class_idx)
    
    print("Trace:", all_trace)
    print("Eigenvalues:", all_eigenvalues)


    return


class AccMeter:
    def __init__(self):
        self.top1 = AverageMeter('Acc@1', ':6.3f')
        self.top5 = AverageMeter('Acc@5', ':6.3f')

        self.class_num = torch.zeros(config.num_classes).cuda(config.gpu)
        self.correct = torch.zeros(config.num_classes).cuda(config.gpu)

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


best_acc1 = 0.0

def validate(val_loader, model, criterion, config, cls_num_list):
    batch_time = AverageMeter('Time', ':6.3f')
    acc_meter = {
        'classifier1': AccMeter(),}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if config.gpu is not None:
                input = input.cuda(config.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(config.gpu, non_blocking=True)

            # compute output
            output1 = model(input)
            # loss = criterion(output1, target)

            # measure accuracy and record loss
            acc_meter['classifier1'].update(output1, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        global best_acc1
        is_classifier1_best = False

        for name in acc_meter.keys():
            entry = acc_meter[name]

            acc1, acc5 = entry.top1.avg, entry.top5.avg
            if config.name == 'Inaturalist2018':
                head_acc, med_acc, tail_acc = entry.get_shot_acc(cls_num_list)
            else:
                head_acc, med_acc, tail_acc = entry.get_shot_acc()

            print(('* ({name})  Acc@1 {acc1:.3f}  HAcc {head_acc:.3f}  MAcc {med_acc:.3f}  TAcc {tail_acc:.3f}' ).format(
                name=name, acc1=acc1, acc5=acc5, head_acc=head_acc, med_acc=med_acc, tail_acc=tail_acc))

    return is_classifier1_best

def get_esd_plot(eigenvalues, weights, class_idx):
    density, grids = density_generate(eigenvalues, weights)
    plt.semilogy(grids, density + 1.0e-7)
    plt.ylabel('Density (Log Scale)', fontsize=14, labelpad=10)
    plt.xlabel('Eigenvalue', fontsize=14, labelpad=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])
    plt.tight_layout()
    lambda_min = np.min(eigenvalues)
    lambda_max = np.max(eigenvalues)
    lambda_ratio = lambda_max/lambda_min
    # 添加文本到右上角
    textstr = '\n'.join((
        r'$\lambda_{min}=%.4f$' % lambda_min,
        r'$\lambda_{max}=%.4f$' % lambda_max,
        # r'$\lambda_{max}/\lambda_{min}=%.2f$' % lambda_ratio
    ))

    # 设置在右上角位置（用 axes 坐标系 [0,1]）
    plt.gca().text(
        0.98, 0.98, textstr,
        transform=plt.gca().transAxes,
        fontsize=24,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.6)
    )
    # wandb.log({"esd plot{class_idx}":plt})
    # wandb.log({f"lambda_min-{class_iCBReweightdx}":lambda_min, f"lambda_max-{class_idx}":lambda_max, f"lambda_ratio-{class_idx}":lambda_ratio})
    savepath = '/workspace/1-Invariant/CSA/CSA/test/CE/CE_class_' + str(class_idx)
    # plt.savefig(savepath+'.PDF')
    plt.savefig(savepath+'.eps', format='eps', bbox_inches='tight')
    plt.savefig(savepath+'.jpg', format='jpg', bbox_inches='tight')
    plt.close()


def density_generate(eigenvalues,
                     weights,
                     num_bins=10000,
                     sigma_squared=1e-5,
                     overhead=0.01):

    eigenvalues = np.array(eigenvalues)
    weights = np.array(weights)

    lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
    lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    num_runs = eigenvalues.shape[0]
    density_output = np.zeros((num_runs, num_bins))

    for i in range(num_runs):
        for j in range(num_bins):
            x = grids[j]
            tmp_result = gaussian(eigenvalues[i, :], x, sigma)
            density_output[i, j] = np.sum(tmp_result * weights[i, :])
    density = np.mean(density_output, axis=0)
    normalization = np.sum(density) * (grids[1] - grids[0])
    density = density / normalization
    return density, grids


def gaussian(x, x0, sigma_squared):
    return np.exp(-(x0 - x)**2 /
                  (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)

if __name__ == '__main__':
    main()
