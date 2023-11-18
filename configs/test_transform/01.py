from configs.val_transform import *
import os
from monai.transforms import Transform, ScaleIntensityRanged


org_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(keys="image", a_min=0, a_max=4096, b_min=0.0, b_max=1.0, clip=True),
    ]
)


