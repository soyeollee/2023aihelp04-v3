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

    def __init__(self, image_file_list, label_file_list, transforms, phase):
        # Ensure the number of images and labels match, except for inference
        if phase != 'inference':
            assert len(image_file_list) == len(label_file_list), "Image and label lists must have the same length."

        # Prepare a list of dictionaries, each containing the 'image' and 'label' file paths
        # For inference, only use image paths
        if phase == 'inference':
            data_dicts = [{'image': img} for img in image_file_list]
        else:
            data_dicts = [{'image': img, 'label': lbl} for img, lbl in zip(image_file_list, label_file_list)]

        # Initialize the CacheDataset with the data_dicts and the provided transforms for the specific phase
        super().__init__(data_dicts, transforms[phase], cache_rate=1.0, num_workers=4)
