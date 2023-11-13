from configs.train_transform import *
from model.transform import ConvertToMultiChannelBasedOnBratsClassesd, ConvertToForegroundBackground
import numpy as np


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
    RandAffined,
    OneOf,
    RandGridDistortiond,
    RandCoarseDropoutd,

)

transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToForegroundBackground(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=np.pi / 12,
            translate_range=(320 * 0.0625, 320 * 0.0625),
            scale_range=(0.1, 0.1),
            mode="nearest",
            padding_mode="reflection",
        ),
        OneOf(
            [
                RandGridDistortiond(keys=("image", "label"), prob=0.5, distort_limit=(-0.05, 0.05), mode="nearest", padding_mode="reflection"),
                RandCoarseDropoutd(
                    keys=("image", "label"),
                    holes=5,
                    max_holes=8,
                    spatial_size=(1, 1, 1),
                    max_spatial_size=(12, 12, 12),
                    fill_value=0.0,
                    prob=0.5,
                ),
            ]
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
)