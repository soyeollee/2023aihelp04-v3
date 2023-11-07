import logging
import os
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