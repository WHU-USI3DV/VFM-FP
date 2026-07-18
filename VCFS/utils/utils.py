"""General utility functions shared by VCFS scripts."""

import os
import random

import numpy as np
import torch
from PIL import Image


def cvtColor(image):
    """Convert PIL images to RGB before model inference/training."""
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    return image.convert("RGB")


def resize_image(image, size):
    """Resize an image to ``size`` and keep the legacy return signature."""
    width, height = size
    return image.resize((width, height), Image.BICUBIC), width, height


def resize_img(image, size, label):
    """Resize an image/mask pair using interpolation suitable for each type."""
    width, height = size
    image = image.resize((width, height), Image.BICUBIC)
    label = label.resize((width, height), Image.NEAREST)
    return image, label


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    raise ValueError("Optimizer has no parameter groups.")


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def preprocess_input(image):
    image /= 255.0
    return image


def show_config(**kwargs):
    print("Configurations:")
    print("-" * 70)
    print("|%25s | %40s|" % ("keys", "values"))
    print("-" * 70)
    for key, value in kwargs.items():
        print("|%25s | %40s|" % (str(key), str(value)))
    print("-" * 70)


def download_weights(backbone, model_dir="./model_data"):
    from torch.hub import load_state_dict_from_url

    download_urls = {
        "mobilenet": "https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar",
        "xception": "https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth",
    }
    if backbone not in download_urls:
        raise ValueError(f"Unsupported backbone for weight download: {backbone}")

    os.makedirs(model_dir, exist_ok=True)
    load_state_dict_from_url(download_urls[backbone], model_dir)
