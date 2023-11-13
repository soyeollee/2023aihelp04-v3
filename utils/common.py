import logging
import os
import torch
from collections import OrderedDict

def get_logger(run, path):
    logger = logging.getLogger(run)
    logger.setLevel(logging.DEBUG)  # 또는 원하는 로깅 레벨로 설정

    # init file handler
    file_handler = logging.FileHandler(os.path.join(path, 'log.txt'))
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # init console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # 핸들러를 로거에 추가합니다.
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def load_state_dict_force(model, ckpt_path):
    # load model with "load_state_dict" and "strict=False"
    # if size mismatch, just ignore it
    pretrained_dict = torch.load(ckpt_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
