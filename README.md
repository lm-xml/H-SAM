## Installation

**Requirements**

* Python 3.8
* torch 2.0.0
* torchvision 0.15.0


## Dataset Preparation
* [CIFAR-10 & CIFAR-100 & ImageNet-LT]

`The Long-tailed setting of imagenet` can be found in `config/dataset_txt/`.

## Training

```
python train_cifar.py
```

`The training setting of Datasets` can be found in `config/cifar10`,  `config/cifar100`, and `config/imagenet`.


## Output
```
H-SAN
├── saved
│   ├── modelname_date
│   │   ├── ckps
│   │   │   ├── current.pth.tar
│   │   │   └── model_best.pth.tar
│   │   └── logs
│   │       └── modelname.txt
│   ...   
```