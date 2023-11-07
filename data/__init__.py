from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch
from data.aihelp import AIHelp4Dataset
from utils.visualize import visualize_val_data


def get_dataset(dataset,
                data_dir,
                work_dir,
                train_transform,
                val_transform,
                batch_size=1,
                num_workers=4,
                visualize=False,
                phases=['train', 'val']):
    dataloader_list = []

    if dataset == 'brats':
        if 'train' in phases:
            train_ds = DecathlonDataset(
                root_dir=data_dir,
                task="Task01_BrainTumour",
                transform=train_transform,
                section="training",
                download=True,
                cache_rate=0.0,
                num_workers=4,
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            dataloader_list.append(train_loader)
        if 'val' in phases:
            val_ds = DecathlonDataset(
                root_dir=data_dir,
                task="Task01_BrainTumour",
                transform=val_transform,
                section="validation",
                download=False,
                cache_rate=0.0,
                num_workers=4,
            )
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)
            dataloader_list.append(val_loader)

    elif dataset == 'aihelp':
        if 'train' in phases:
            train_ds = AIHelp4Dataset(
                ...
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            dataloader_list.append(train_loader)
        elif 'val' in phases:
            val_ds = AIHelp4Dataset(
                ...
            )
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)
            dataloader_list.append(val_loader)
    # visualize
    if visualize and 'val' in phases:
        visualize_val_data(val_ds, work_dir)

    return dataloader_list