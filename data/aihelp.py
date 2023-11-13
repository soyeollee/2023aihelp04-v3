import os
import numpy as np
import nibabel as nib

from sklearn.model_selection import KFold
from monai.data import CacheDataset
from monai.transforms import Compose

class AIHelp4Dataset(CacheDataset):
    """
    Custom dataset to load NIfTI files and cache them for efficient reading.
    Uses MONAI's CacheDataset for caching.

    Args:
        image_file_list (list of str): List of file paths to the image files.
        label_file_list (list of str): List of file paths to the label files, can be None for inference.
        transforms (dict of monai.transforms.Compose): Dictionary of transformations to apply for each phase.
        phase (str): The phase for which the dataset is being used. One of ['train', 'val', 'inference'].
    """

    def __init__(self, image_file_list, transforms, phase):
        # Ensure the number of images and labels match, except for inference
        if phase != 'test':
            label_file_list = [x.replace('T1_FOV.nii.gz', 'ROI.nii.gz') for x in image_file_list]
        # Prepare a list of dictionaries, each containing the 'image' and 'label' file paths

        # For inference, only use image paths
        if phase == 'test':
            # path_list = [x.split('/')[-3] for x in image_file_list]
            data_dicts = [{'image': img, 'path': path} for img, path in zip(image_file_list, image_file_list)]
        else:
            data_dicts = [{'image': img, 'label': lbl, 'path': path} for img, lbl, path in zip(image_file_list, label_file_list, image_file_list)]

        # Initialize the CacheDataset with the data_dicts and the provided transforms for the specific phase
        super().__init__(data_dicts, transforms, cache_rate=1.0, num_workers=4)


def get_aihelp_datalist(data_path:str = '',
                        num_train:int = 50,
                        num_val:int = 10,  # train 데이터에 포함되어있음
                        num_test:int = 30,
                        num_folds:int = 1,
                        seed:int = 42,
                        use_val_data:bool = True,
                        ):
    """
    데이터셋의 train/val/test 데이터 경로를 반환하는 함수

    ...
    :param num_train: label 데이터가 있는 path의 수 (aihelp4 기준 50)
    :param num_val: train 데이터에서 validation으로 사용할 데이터의 수
    :param use_val_data: val 데이터를 train할 때 포함할지 여부 (fold > 1일때만 사용되는 arg)
    if val_data: k개의 fold가 동일한 validation 데이터를 사용
    if !val_data: k개의 fold가 서로 다른 validation (fold) 데이터를 사용

    :return: train/validation/test data path list
    """
    data_path_list = os.listdir(os.path.join(data_path))
    data_path_list.sort()
    train_data_list = data_path_list[:num_train-num_val]
    val_data_list = data_path_list[num_train-num_val:num_train]

    if num_folds == 1:
        train_data_list = [train_data_list, ]
        val_data_list = [val_data_list, ]
    if num_folds > 1:
        _train_data_list = []
        _val_data_list = []

        cv = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

        if use_val_data:
            train_data_list += val_data_list

        for t, v in cv.split(train_data_list):
            _train_data_list.append(np.array(train_data_list)[t])
            _val_data_list.append(np.array(train_data_list)[v]) \
                if use_val_data else _val_data_list.append(val_data_list)
        train_data_list = _train_data_list
        val_data_list = _val_data_list

    for fold in range(num_folds):
        train_data_list[fold] = [os.path.join(data_path, x, 'T1_space', 'T1_FOV.nii.gz') for x in train_data_list[fold]]
        val_data_list[fold] = [os.path.join(data_path, x, 'T1_space', 'T1_FOV.nii.gz') for x in val_data_list[fold]]

    test_data_list = data_path_list[num_train: num_train + num_test]

    test_data_list = [os.path.join(data_path, x, 'T1_space', 'T1_FOV.nii.gz') for x in test_data_list]

    return train_data_list, val_data_list, test_data_list