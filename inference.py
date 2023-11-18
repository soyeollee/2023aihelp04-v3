import time
import torch
import os

from monai.metrics import DiceMetric
from monai.data import DataLoader, decollate_batch
from model.inference import inference
from utils.visualize import visualize_train_result
from data import get_dataset
from configs import get_transform

import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
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
from monai.utils import set_determinism

import torch

from utils.visualize import visualize_best_model
import torch
from monai.inferers import sliding_window_inference
from monai.inferers import SimpleInferer, SlidingWindowInferer

import argparse
from utils.common import str2bool
from model.builder import get_model
from configs import get_config, get_transform, post_pred, post_label
from data import get_dataset
import nibabel as nib
from tqdm import tqdm
from monai.engines import EnsembleEvaluator, SupervisedEvaluator, SupervisedTrainer
from monai.handlers import MeanDice, StatsHandler, ValidationHandler, from_engine
import copy
import numpy as np
from monai.transforms.utils import allow_missing_keys_mode




def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default='00')
    parser.add_argument('--test-config', type=str, default='00')
    parser.add_argument('--test-tf-config', type=str, default='00')
    parser.add_argument('--val-tf-config', type=str, default='00')

    parser.add_argument('--data-dir', type=str, default='/home/soyeollee/workspace/data/aihelp/ImageData')
    parser.add_argument('--validation', type=str2bool, default=False)
    parser.add_argument('--visualize', type=str2bool, default=True)
    return parser.parse_args()


def _inference(_input, _model, _roi_size):
    def _compute(_input):
        return sliding_window_inference(
            inputs=_input,
            roi_size=_roi_size,
            sw_batch_size=1,
            predictor=_model,
            overlap=0.5,
        )

    with torch.cuda.amp.autocast():
        return _compute(_input)


def inference(models, dataloader, test_config, work_dir, invertd_transform, visualize, data_dir, validation):
    os.makedirs(os.path.join(work_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(work_dir, 'visualize'), exist_ok=True)

    [model.eval() for model in models]
    _model = models[0]
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    with torch.no_grad():
        for val_data in tqdm(dataloader):
            _val_data = copy.deepcopy(val_data)
            _val_inputs = _val_data["image"].to('cuda')
            _val_data['pred'] = _inference(_val_inputs, _model, test_config.roi_size)
            _val_invertd_data = [invertd_transform(i) for i in decollate_batch(_val_data)]

            # write batch result to nii.gz file
            _file_path = _val_data['path'][0].split('/')[-3]
            save_path = os.path.join(work_dir, 'results', _file_path + '.nii.gz')

            _affine = val_data['image'][0].affine.cpu().numpy()

            if validation:
                _val_pred = _val_data['pred'][0].cpu()  # c, h, w
                _val_pred = post_pred(_val_pred)
                _val_labels = copy.deepcopy(val_data['label'])
                _val_labels = post_label(_val_labels[0]).cpu()
                dice_metric([_val_pred], [_val_labels])

            _output = post_pred(_val_invertd_data[0]['pred']).argmax(0).cpu().numpy().astype(float)
            _output_nii = nib.Nifti1Image(_output, affine=_affine)
            nib.save(_output_nii, save_path)

            if visualize:
                if validation:
                    label_sum = (_val_labels != 0)[1].sum(dim=(0, 1))
                    if label_sum.sum().item() == 0:
                        continue
                    max_channel = label_sum.argmax().item()
                else:
                    max_channel = (_output != 0).sum((0, 1)).argmax().item()

                image = nib.load(os.path.join(data_dir, _file_path, 'T1_space', 'T1_FOV.nii.gz')).get_fdata()

                # visualize image
                plt.figure("image", (6*1, 6))
                plt.subplot(1, 1, 1)
                plt.title(f"image channel 1")
                plt.imshow(image[:, :, max_channel], cmap="gray")
                plt.savefig(os.path.join(work_dir, 'visualize', _file_path + '_image.png'))

                # visualize pred
                plt.figure("pred", (6*1, 6))
                plt.subplot(1, 1, 1)
                plt.title(f"pred channel 1")
                plt.imshow(_output[:, :, max_channel], cmap="gray")
                plt.savefig(os.path.join(work_dir, 'visualize', _file_path + '_pred.png'))

                if validation:
                    label = nib.load(os.path.join(data_dir, _file_path, 'T1_space', 'ROI.nii.gz')).get_fdata()

                    # visualize label
                    plt.figure("label", (6*1, 6))
                    plt.subplot(1, 1, 1)
                    plt.title(f"label channel 1")
                    plt.imshow(label[:, :, max_channel], cmap="gray")
                    plt.savefig(os.path.join(work_dir, 'visualize', _file_path + '_label.png'))



        if validation:
            metric_org = dice_metric.aggregate().item()
            metric_org = round(metric_org, 5)
            with open(os.path.join(work_dir, f'metric_{str(metric_org)}.txt'), 'w') as f:
                pass


if __name__ == '__main__':
    args = parse_arguments()
    test_config = get_config(args.test_config, phase='test')
    work_dir = os.path.join('runs', 'pred', args.run)

    models = [get_model(name=test_config.model['name'],
                      kwargs=test_config.model['args'],
                      num_classes=test_config.num_classes,
                      pretrained=_pretrained) for _pretrained in test_config.pretrained]

    if args.validation:
        test_transform, invertd_transform = get_transform(args.val_tf_config, phase='val', org=True, return_post=True)
        os.makedirs(os.path.join(work_dir, 'visualize'), exist_ok=True)
    else:
        test_transform, invertd_transform = get_transform(args.test_tf_config, phase='test', org=True, return_post=True)

    if args.validation:
        dataloader = get_dataset(dataset=test_config.dataset,
                                 data_dir=args.data_dir,
                                 work_dir=os.path.join(work_dir, 'visualize'),
                                 train_transform=None,
                                 val_transform=test_transform,
                                 test_transform=None,
                                 batch_size=1,
                                 num_workers=4,
                                 visualize=args.visualize,
                                 phases=['val'],
                                 use_val_data=False,
                                 )
    else:
        dataloader = get_dataset(dataset=test_config.dataset,
                                 data_dir=args.data_dir,
                                 work_dir=None,
                                 train_transform=None,
                                 val_transform=None,
                                 test_transform=test_transform,
                                 batch_size=1,
                                 num_workers=4,
                                 visualize=False,
                                 phases=['test'])

    inference(models, dataloader[0], test_config, work_dir, invertd_transform, args.visualize, args.data_dir,
              args.validation)