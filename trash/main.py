import os
import logging

import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet

from monai.utils import set_determinism
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)


import torch
import argparse
from configs import get_config, get_transform
from utils.common import get_logger, str2bool
from utils.visualize import visualize_val_data
from data import get_dataset
from trainval import trainval
from inference import inference
from model.builder import get_loss_function, get_optimizer, get_scheduler, get_model


def parse_arguments():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--run', type=str, default='_test')
    # parser.add_argument('--train-config', type=str, default='00_brats')
    # parser.add_argument('--val-config', type=str, default='00_brats')
    # parser.add_argument('--train-tf-config', type=str, default='00_brats')
    # parser.add_argument('--val-tf-config', type=str, default='00_brats')
    #
    # parser.add_argument('--data-dir', type=str,
    #                     default='/home/soyeollee/workspace/data/')
    # parser.add_argument('--visualize', type=str2bool, default=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default='00_aihelp')
    parser.add_argument('--train-config', type=str, default='00_aihelp')
    parser.add_argument('--val-config', type=str, default='00_aihelp')
    parser.add_argument('--train-tf-config', type=str, default='00_aihelp')
    parser.add_argument('--val-tf-config', type=str, default='00_aihelp')

    parser.add_argument('--data-dir', type=str,
                        default='/home/soyeollee/workspace/data/aihelp/ImageData')
    parser.add_argument('--visualize', type=str2bool, default=True)

    return parser.parse_args()


def main(args):
    work_dir = os.path.join('../runs', args.run)
    data_dir = args.data_dir

    os.makedirs(work_dir, exist_ok=True)

    # get logger
    logger = get_logger(run=args.run, path=work_dir)

    train_config = get_config(args.train_config, phase='train')
    val_config = get_config(args.val_config, phase='val')
    train_transform = get_transform(args.train_tf_config, phase='train')
    val_transform = get_transform(args.val_tf_config, phase='val')
    val_origin_transform = get_transform(args.val_tf_config, phase='val', org=True)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # set random seed
    logger.info('get dataset...')
    set_determinism(seed=0)
    train_loader, val_loader = get_dataset(
        dataset=train_config.dataset,
        data_dir=data_dir,
        work_dir=work_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        visualize=args.visualize,
        phases=['train', 'val']
       )

    logger.info('get model...')
    model = get_model(name=train_config.model['name'],
                      kwargs=train_config.model['args'],
                      num_classes=train_config.num_classes,
                      pretrained=train_config.pretrained)

    loss_function = get_loss_function(loss_name=train_config.loss['name'],
                                      args=train_config.loss['args'])
    optimizer = get_optimizer(
        model=model,
        optimizer_name=train_config.optimizer['name'],
        args=train_config.optimizer['args'])

    lr_scheduler = get_scheduler(optimizer,
                                 train_config.scheduler['name'],
                                 train_config.scheduler['args'])

    model = trainval(model=model,
                     train_loader=train_loader,
                     val_loader=val_loader,
                     optimizer=optimizer,
                     loss_function=loss_function,
                     lr_scheduler=lr_scheduler,
                     data_dir=data_dir,
                     work_dir=work_dir,
                     post_trans=post_trans,
                     visualize=args.visualize,
                     train_config=train_config,
                     val_org_transform=val_origin_transform,
                     logger=logger)

    inference(model)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)