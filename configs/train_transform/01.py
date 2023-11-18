from configs.train_transform import *
from model.transform import ConvertToMultiChannelBasedOnBratsClassesd, ConvertToForegroundBackground


transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToForegroundBackground(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),  #################
        ScaleIntensityRanged(keys="image", a_min=0, a_max=4096, b_min=0.0, b_max=1.0, clip=True),
        RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        RandGaussianNoised(keys="image", prob=0.5),

    ]
)