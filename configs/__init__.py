import yaml
import os
import importlib
from types import SimpleNamespace
from configs.val_transform import *

post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)])

def get_config(config, phase='train'):
    config_path = os.path.join('./configs', phase, config + '.yaml')
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return SimpleNamespace(**config)


def get_transform(config, phase='train', org=False, test_threshold=0.5, return_post=False):
    module_path = '.'.join(['configs', f'{phase}_transform', config])
    module = importlib.import_module(module_path)

    if org:
        transform = module.org_transforms
    else:
        transform = module.transform


    if return_post:
        keys = "pred"

        post_transforms = Compose(
            [
                Invertd(
                    keys=keys,
                    transform=transform,
                    orig_keys="image",
                    meta_keys="pred_meta_dict",
                    orig_meta_keys="image_meta_dict",
                    meta_key_postfix="meta_dict",
                    nearest_interp=False,
                    to_tensor=True,
                    device="cpu",
                    allow_missing_keys=True
                )
            ]
        )
        return transform, post_transforms
    else:
        return transform