from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch
from data.aihelp import AIHelp4Dataset, get_aihelp_datalist
from utils.visualize import visualize_val_data


def get_dataset(dataset,
                data_dir,
                work_dir,
                train_transform,
                val_transform,
                test_transform,
                batch_size=1,
                num_workers=4,
                visualize=False,
                phases=['train', 'val'],
                num_folds=4,
                fold_id=0,
                use_val_data=False
                ):
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
        train_data_list, val_data_list, test_data_list = get_aihelp_datalist(data_path=data_dir,
                                                                             num_folds=num_folds,
                                                                             use_val_data=use_val_data)
        if 'train' in phases:
            train_ds = AIHelp4Dataset(
                train_data_list[fold_id],
                transforms=train_transform,
                phase='train'
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            dataloader_list.append(train_loader)
        if 'val' in phases:
            val_ds = AIHelp4Dataset(
                val_data_list[fold_id],
                transforms=val_transform,
                phase='val'
            )
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)
            dataloader_list.append(val_loader)
        if 'test' in phases:
            test_ds = AIHelp4Dataset(
                test_data_list,
                transforms=test_transform,
                phase='test'
            )
            test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)
            dataloader_list.append(test_loader)
    # visualize
    if visualize and 'val' in phases:
        visualize_val_data(val_ds, work_dir)

    return dataloader_list