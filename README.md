## Installation

**Requirements**

* Python 3.8
* requirements.txt


## Dataset Preparation
* [CIFAR-10 & CIFAR-100 & ImageNet-LT & iNaturalist 2018]

`The Long-tailed setting of imagenet and iNaturalist 2018` can be found in `config/dataset_txt/ImageNet-LT` and `config/dataset_txt/iNaturalist`.

## Training

```
python train_cifar.py
```

`The training setting of Datasets` can be found in `config/cifar10`,  `config/cifar100`, and `config/imagenet`.


## Output
```
H-SAM
├── saved
│   ├── modelname_date
│   │   ├── ckps
│   │   │   ├── current.pth.tar
│   │   │   └── model_best.pth.tar
│   │   └── logs
│   │       └── modelname.txt
│   ...   
```