import yaml
import os
import importlib
from types import SimpleNamespace
from configs.val_transform import *

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

    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=transform,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
                device="cpu",
                allow_missing_keys=True
            ),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=test_threshold),
        ]
    )

    if return_post:
        return transform, post_transforms
    else:
        return transform