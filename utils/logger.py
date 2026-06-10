from pathlib import Path
from yacs.config import CfgNode as CN
import os
import time
import logging

_C = CN()

# ===== General settings =====
_C.dataset = 'cifar10'
_C.data_path = './data/cifar10'
_C.num_classes = 100
_C.arch = 'resnet32'
_C.loss_type = 'LA'              # LDAM, VS, LA, CE
_C.imb_type = 'exp'
_C.imb_factor = 0.005
_C.train_rule = 'None'           # data sampling strategy
_C.train_sampler = 'None'           # data sampling strategy
_C.workers = 8
_C.epochs = 200
_C.start_epoch = 0
_C.batch_size = 64
_C.lr = 0.1
_C.momentum = 0.9
_C.weight_decay = 2e-4
_C.print_freq = 10
_C.resume = ''
_C.evaluate = False
_C.pretrained = False
_C.seed = 0
_C.gpu = 0

# ===== VS loss parameters =====
_C.gamma = 0.3
_C.tau = 1.0

# ===== Logging =====
_C.log_dir = 'logs'
_C.model_dir = 'ckps'
_C.name = 'test'

# ===== SAM & optimization =====
_C.rho = 0.2
_C.rho_steps = [0.1, 0.2]
_C.min_rho = 0.05
_C.max_rho = 0.8
_C.rho_schedule = 'step'         # none, linear, step
_C.SAM_type = 'H_SAM'         # H_SAM, Focal-SAM, SAM, None
_C.lmbda = 0.6
_C.flat_gamma = 1.0
_C.sharpness = 0.5
_C.prec = 'fp32'
_C.split_threshold = 3
_C.split = 1

# ===== Learning rate schedule =====
_C.cos_lr = True
_C.end_lr_cos = 0.0


def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)


def create_logger(cfg, cfg_name, file_name):
    time_str = time.strftime('%Y%m%d%H%M%S')

    cfg_name = os.path.basename(cfg_name).split('.')[0]
    name_list = [file_name.split('/')[-1], cfg.dataset, cfg.imb_type, str(cfg.imb_factor), cfg.loss_type, cfg.SAM_type,
                 'rho_schedule', cfg.rho_schedule, 'seed', str(cfg.seed), 'workers', str(cfg.workers),
                 str(cfg.batch_size), 'cos_lr', str(cfg.cos_lr), cfg.arch, str(cfg.train_rule)]
    print(', '.join(f"{name}: {value}" for name, value in zip(
        ['dataset', 'arch', 'loss_type', 'train_rule', 'imb_type', 'imb_factor',
         'SAM_type', 'rho_schedule', 'seed_flag', 'seed', 'num_exper', 'workers'],
        [cfg.dataset, cfg.arch, cfg.loss_type, cfg.train_rule, cfg.imb_type, str(cfg.imb_factor),
         cfg.SAM_type, cfg.rho_schedule, 'seed', str(cfg.seed), cfg.workers]
    )))
    log_dir = Path("saved/") / '_'.join(name_list) / Path(cfg.log_dir)
    print('=> creating {}'.format(log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = '{}.txt'.format(cfg_name)
    final_log_file = log_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    model_dir = Path("saved/") / '_'.join(name_list) / Path(cfg.model_dir)
    print('=> creating {}'.format(model_dir))
    model_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(model_dir)
