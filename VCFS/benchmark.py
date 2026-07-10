"""Report VCFS model complexity and optional CUDA inference speed."""

from __future__ import annotations

import os
import time

import torch
from thop import clever_format, profile

from nets.deeplabv3_plus import DeepLab
from utils.class_config import load_class_config


def _env_int(name, current):
    value = os.environ.get(name)
    return current if value in (None, "") else int(value)


def _env_shape(name, current):
    value = os.environ.get(name)
    if value in (None, ""):
        return current
    parts = [int(part.strip()) for part in value.replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} must be like 512,512 or 512x512")
    return parts


def evaluate_model_complexity():
    cuda_devices = os.environ.get("VCFS_CUDA_VISIBLE_DEVICES")
    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    class_config = load_class_config(os.environ.get("VCFS_CLASS_CONFIG"))
    num_classes = _env_int("VCFS_NUM_CLASSES", class_config["num_classes"])
    backbone = os.environ.get("VCFS_BACKBONE", "mobilenet")
    downsample_factor = _env_int("VCFS_DOWNSAMPLE_FACTOR", 8)
    input_shape = _env_shape("VCFS_INPUT_SHAPE", [512, 512])
    warmup_iters = _env_int("VCFS_BENCHMARK_WARMUP", 50)
    test_iters = _env_int("VCFS_BENCHMARK_ITERS", 200)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: DeepLabV3+ ({backbone})")
    print(f"Input: {input_shape[0]}x{input_shape[1]}, classes: {num_classes}")

    model = DeepLab(
        num_classes=num_classes,
        backbone=backbone,
        downsample_factor=downsample_factor,
        pretrained=False,
    ).to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)

    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops = macs * 2
    flops_str, params_str = clever_format([flops, params], "%.2f")
    print(f"Parameters: {params / 1e6:.2f} M ({params_str})")
    print(f"FLOPs: {flops / 1e9:.2f} G ({flops_str})")

    if not torch.cuda.is_available():
        print("CUDA is unavailable; skipped GPU memory and FPS measurements.")
        return

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(dummy_input)
    max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"Max GPU memory: {max_memory:.2f} MB")

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(test_iters):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    total_time = time.time() - start_time

    avg_time_ms = (total_time / test_iters) * 1000
    fps = 1.0 / (total_time / test_iters)
    print(f"Average time: {avg_time_ms:.2f} ms/image")
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    evaluate_model_complexity()