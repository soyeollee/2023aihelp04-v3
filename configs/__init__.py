import yaml
import os
import importlib
from types import SimpleNamespace

def get_config(config, phase='train'):
    config_path = os.path.join('./configs', phase, config + '.yaml')
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return SimpleNamespace(**config)


def get_transform(config, phase='train', org=False):
    module_path = '.'.join(['configs', f'{phase}_transform', config])
    module = importlib.import_module(module_path)

    if org:
        return module.org_transforms
    else:
        return module.transform
    