from configs.val_transform import *
from model.transform import ConvertToMultiChannelBasedOnBratsClassesd, ConvertToForegroundBackground

transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToForegroundBackground(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),  ###############################
        ScaleIntensityRanged(keys="image", a_min=0, a_max=4096, b_min=0.0, b_max=1.0, clip=True),
    ]
)

org_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
        ConvertToForegroundBackground(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),  #################
        ScaleIntensityRanged(keys="image", a_min=0, a_max=4096, b_min=0.0, b_max=1.0, clip=True),
    ]
)