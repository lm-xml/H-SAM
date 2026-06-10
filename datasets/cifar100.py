import numpy as np
from .autoaug import CIFAR10Policy, Cutout
import torch
import torchvision
from torchvision import transforms
import torchvision.datasets

class IMBALANCECIFAR100(torchvision.datasets.CIFAR100):
    cls_num = 100

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR100, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)
        self.ori_cls_num_list = img_num_list

    def get_ori_cls_num_list(self):
        return self.ori_cls_num_list

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list



class CIFAR100_LT():

    def __init__(self, root='./data/cifar100', imb_type='exp', imb_factor=0.01, batch_size=64, num_works=8):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])

        train_dataset = IMBALANCECIFAR100(root=root, imb_type=imb_type, imb_factor=imb_factor, rand_number=0, train=True, download=True,
                                         transform=transform_train)
        eval_dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_val)
        
        self.cls_num_list = train_dataset.get_cls_num_list()

        self.train_instance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True)

        self.eval = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=100, shuffle=False,
            num_workers=num_works, pin_memory=True)


    def get_weighted_loader(self, weighted_alpha):
        
        train_dataset = self.train_instance.dataset
        targets = train_dataset.targets
        batch_size = self.train_instance.batch_size
        num_works = self.train_instance.num_workers

        cls_weight = 1.0 / (np.array(self.cls_num_list) ** weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(self.cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        
        weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(targets), replacement=True)
        weighted_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=weighted_sampler)
        return weighted_loader
