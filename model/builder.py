import torch
from monai.losses import DiceLoss, DiceCELoss
from monai.networks.nets import SegResNet, SwinUNETR
from utils.common import load_state_dict_force


def get_optimizer(model, optimizer_name, args):
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), **args)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), **args)

    return optimizer


def get_loss_function(loss_name, args):
    if loss_name == 'DiceLoss':
        loss_function = DiceLoss(**args)
    elif loss_name == 'DiceCELoss':
        loss_function = DiceCELoss(**args)
    else:
        raise NotImplementedError

    return loss_function


def get_scheduler(optimizer, scheduler_name, args):
    if scheduler_name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **args)
    else:
        raise NotImplementedError

    return scheduler


def get_model(name, num_classes, kwargs, pretrained=None):
    if name == 'SegResNet':
        model = SegResNet(
            out_channels=num_classes,
            **kwargs
        )

    elif name == 'SwinUNETR':
        model = SwinUNETR(
            out_channels=num_classes,
            **kwargs
        )

    if pretrained != 'None':
        try:
            model.load_state_dict(torch.load(pretrained))
        except:
            load_state_dict_force(model, pretrained)

    return model.to('cuda')

