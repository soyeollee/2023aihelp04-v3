import os

import torch
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference

pallet = ['red', 'green', 'blue', 'yellow', 'purple', 'brown', 'pink', 'orange', 'cyan', 'magenta', 'lime', 'teal', 'lavender', 'maroon', 'navy', 'olive', 'grey', 'black', 'white']

def visualize_val_data(val_ds, work_dir):
    # pick one image from DecathlonDataset to visualize and check the 4 channels
    val_data_example = val_ds[2]
    input_dim = val_data_example['image'].shape[0]
    output_dim = val_data_example['label'].shape[0]

    print(f"image shape: {val_data_example['image'].shape}")
    plt.figure("image", (6*input_dim, 6))


    label_sum = (val_data_example['label'] != 0).sum(dim=(0, 1, 2))
    max_channel = label_sum.argmax().item()

    for i in range(input_dim):
        plt.subplot(1, input_dim, i + 1)
        plt.title(f"image channel {i}")
        plt.imshow(val_data_example["image"][i, :, :, max_channel].detach().cpu(), cmap="gray")
    plt.savefig(os.path.join(work_dir, 'input0_image.png'))
    # also visualize the 3 channels label corresponding to this image
    print(f"label shape: {val_data_example['label'].shape}")
    plt.figure("label", (6*output_dim, 6))
    for i in range(output_dim):
        plt.subplot(1, output_dim, i + 1)
        plt.title(f"label channel {i}")
        plt.imshow(val_data_example["label"][i, :, :, max_channel].detach().cpu())
    plt.savefig(os.path.join(work_dir, 'input0_label.png'))


def visualize_train_result(
        epoch_loss_values,
        metric_values,
        val_interval,
        metric_values_all,
        work_dir
    ):
    num_classes = len(metric_values_all)
    plt.figure("train_mean_loss", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="red")
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="green")
    plt.savefig(os.path.join(work_dir, 'train_mean_dice.png'))

    plt.figure("train_cls_loss", (6*num_classes, 6))
    for cls in range(num_classes):
        plt.subplot(1, num_classes, cls+1)
        plt.title(f"Val Mean Dice Class{cls}")
        x = [val_interval * (i + 1) for i in range(len(metric_values_all[cls]))]
        y = metric_values_all[cls]
        plt.xlabel("epoch")
        plt.plot(x, y, color=pallet[cls])
    plt.savefig(os.path.join(work_dir, f'train_cls_dice.png'))


def visualize_best_model(model, work_dir, val_loader, post_trans, num_classes):
    num_classes -= 1

    device = torch.device("cuda")
    model.load_state_dict(torch.load(os.path.join(work_dir, "best_metric_model.pth")))
    model.eval()

    def inference(input):
        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(240, 240, 160),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )

        with torch.cuda.amp.autocast():
            return _compute(input)

    with torch.no_grad():
        for data in val_loader:
            val_input = data["image"].to(device)  # b n_input_c h w d
            val_label = data["label"].to(device)  # b n_class h w d

            input_channel = val_input.shape[1]

            label_sum = (val_label != 0).sum(dim=(0,1,2,3))
            if label_sum.sum().item() == 0:
                continue

            max_channel = label_sum.argmax().item()

            # select one image to evaluate and visualize the model output
            # val_input = val_input.unsqueeze(0).to(device)
            roi_size = (128, 128, 64)
            sw_batch_size = 4
            val_output = inference(val_input).to(device)
            val_output = post_trans(val_output[0])
            plt.figure("image", (6*input_channel, 6))

            for i in range(input_channel):
                plt.subplot(1, input_channel, i + 1)
                plt.title(f"image channel {i}")
                plt.imshow(val_input[0, i, :, :, max_channel].detach().cpu(), cmap="gray")
            plt.savefig(os.path.join(work_dir, 'input1_image.png'))
            # visualize the 3 channels label corresponding to this image
            plt.figure("label", (6 * num_classes, 6))
            for i in range(num_classes-1):
                plt.subplot(1, num_classes, i + 1)
                plt.title(f"label channel {i}")
                plt.imshow(val_label[0, i, :, :, max_channel].detach().cpu())
            plt.savefig(os.path.join(work_dir, 'input1_label.png'))
            # visualize the 3 channels model output corresponding to this image
            plt.figure("output", (6 * num_classes, 6))
            for i in range(num_classes):
                plt.subplot(1, num_classes, i + 1)
                plt.title(f"output channel {i}")
                plt.imshow(val_output[i, :, :, max_channel].detach().cpu())
            plt.savefig(os.path.join(work_dir, 'input1_pred.png'))
            break
