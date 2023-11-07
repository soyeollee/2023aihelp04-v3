import torch
from monai.inferers import sliding_window_inference


def inference(input, model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            # roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    with torch.cuda.amp.autocast():
        return _compute(input)