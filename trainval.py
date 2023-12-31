import json
import time
import torch
import os

from monai.metrics import DiceMetric
from monai.data import DataLoader, decollate_batch
from model.inference import inference
from utils.visualize import visualize_train_result
from data import get_dataset
from configs import get_transform
import yaml
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
from configs import get_config, get_transform

import torch
import argparse
from model.builder import get_loss_function, get_optimizer, get_scheduler, get_model

from utils.visualize import visualize_best_model
import torch
from monai.inferers import sliding_window_inference
from utils.common import get_logger, str2bool


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default='00_aihelp')
    parser.add_argument('--train-config', type=str, default='01_aihelp')
    parser.add_argument('--val-config', type=str, default='00_aihelp')
    parser.add_argument('--train-tf-config', type=str, default='01_aihelp')
    parser.add_argument('--val-tf-config', type=str, default='00_aihelp')

    parser.add_argument('--data-dir', type=str,
                        default='/home/soyeollee/workspace/data/aihelp/ImageData')
    parser.add_argument('--visualize', type=str2bool, default=True)

    return parser.parse_args()


def trainval(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_function,
        lr_scheduler,
        data_dir,
        work_dir,
        post_trans,
        visualize,
        train_config=None,
        val_org_transform=None,
        logger=None
    ):

    def inference(input):
        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(240, 240, 160),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )

        with torch.cuda.amp.autocast():
            return _compute(input)

    max_epochs = train_config.max_epochs
    val_interval = train_config.val_interval
    num_classes = train_config.num_classes

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    device = 'cuda'
    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_all = [[] for _ in range(num_classes)]


    total_start = time.time()
    for epoch in range(max_epochs):
        epoch_start = time.time()
        logger.info("-" * 10)
        logger.info(f"epoch {epoch + 1}/{max_epochs}")

        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            #
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            #
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            if step % 10 == 0:
                logger.info(
                    f"{step}/{len(train_loader) // train_loader.batch_size}"
                    f", train_loss: {loss.item():.4f}"
                    f", step time: {(time.time() - step_start):.4f}"
                )

        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    val_outputs = inference(val_inputs)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                metric_batch = dice_metric_batch.aggregate()
                for i in range(num_classes):
                    _metric = metric_batch[i].item()
                    metric_values_all[i].append(_metric)
                dice_metric.reset()
                dice_metric_batch.reset()

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(time.time() - total_start)
                    torch.save(
                        model.state_dict(),
                        os.path.join(work_dir, "best_metric_model.pth"),
                    )
                    logger.info("saved new best metric model")

                metric_line = [f"#cls{i}: {metric_batch[i].item()}" for i in range(num_classes)]
                metric_line = ' '.join(metric_line)
                logger.info(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" {metric_line}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
        logger.info(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")

    total_time = time.time() - total_start

    logger.info(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")

    if visualize:
        visualize_train_result(
            epoch_loss_values,
            metric_values,
            val_interval,
            metric_values_all,
            work_dir
        )

    # Evaluation on original image spacings
    val_org_loader = get_dataset(train_config.dataset,
                                 data_dir,
                                 work_dir,
                                 train_transform=None,
                                 val_transform=val_org_transform,
                                 test_transform=None,
                                 batch_size=1,
                                 num_workers=train_config.num_workers,
                                 visualize=False,
                                 phases=['val'])



    model.load_state_dict(torch.load(os.path.join(work_dir, "best_metric_model.pth")))
    model.eval()

    # visualize best model
    model = visualize_best_model(
        model=model,
        work_dir=work_dir,
        val_loader=val_loader,
        post_trans=Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)]),
        num_classes=num_classes
    )

    post_transforms_org = Compose(
        [
            Invertd(
                keys="pred",
                transform=val_org_transform,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
                device="cpu",
            ),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
        ]
    )

    # Evaluation on original image spacings
    with torch.no_grad():
        for val_data in val_org_loader[0]:
            val_inputs = val_data["image"].to(device)
            val_data["pred"] = inference(val_inputs)
            val_data = [post_transforms_org(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            dice_metric(y_pred=val_outputs, y=val_labels)
            dice_metric_batch(y_pred=val_outputs, y=val_labels)

        metric_org = dice_metric.aggregate().item()
        metric_batch_org = dice_metric_batch.aggregate()

        dice_metric.reset()
        dice_metric_batch.reset()

    logger.info(f"Metric on original image spacing: {metric_org}")
    metric_org = round(metric_org, 5)

    with open(os.path.join(work_dir, f'metric_{str(metric_org)}.txt'), 'w') as f:
        pass

    for cls in range(len(metric_batch_org)):
        _metric = metric_batch_org[cls].item()
        logger.info(f"metric_cls{cls}: {_metric:.4f}")

    return model  # best model


if __name__ == '__main__':
    args = parse_arguments()
    work_dir = os.path.join('runs', 'train', args.run)
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

    # save config
    yaml.dump(train_config, open(os.path.join(work_dir, 'train_config.yaml'), 'w'))
    yaml.dump(val_config, open(os.path.join(work_dir, 'val_config.yaml'), 'w'))

    # save args
    with open(os.path.join(work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # set random seed
    logger.info('get dataset...')
    set_determinism(seed=0)
    train_loader, val_loader = get_dataset(
        dataset=train_config.dataset,
        data_dir=data_dir,
        work_dir=work_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=None,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        visualize=args.visualize,
        phases=['train', 'val'],
        num_folds=train_config.num_folds,
        fold_id=train_config.fold_id,
        use_val_data=train_config.use_val_data
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

    # inference(model)
