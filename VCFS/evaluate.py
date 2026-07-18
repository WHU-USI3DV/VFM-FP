"""Evaluate one VCFS checkpoint without writing per-image prediction files."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nets.deeplabv3_plus import DeepLab
from utils.class_config import load_class_config
from utils.utils import preprocess_input
from utils.utils_metrics import fast_hist, per_Accuracy, per_class_PA_Recall, per_class_Precision, per_class_iu


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for candidate in (Path.cwd() / path, REPO_ROOT / path, REPO_ROOT / "VCFS" / path):
        if candidate.exists():
            return candidate
    return REPO_ROOT / path


def parse_shape(value):
    parts = [int(part.strip()) for part in value.replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("input shape must be WIDTH,HEIGHT")
    return parts


def read_ids(path):
    return [
        line.strip().replace(",", " ").split()[0]
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_model(checkpoint, num_classes, backbone, downsample_factor, device):
    model = DeepLab(
        num_classes=num_classes,
        backbone=backbone,
        downsample_factor=downsample_factor,
        pretrained=False,
    )
    state = torch.load(checkpoint, map_location=device)
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def predict(model, image, input_shape, device):
    width, height = input_shape
    image = image.convert("RGB").resize((width, height), Image.BICUBIC)
    array = preprocess_input(np.asarray(image, dtype=np.float32))
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(tensor)[0].argmax(dim=0).cpu().numpy().astype(np.uint8)


def evaluate(dataset_path, split_file, checkpoint, class_config, input_shape, backbone, downsample_factor, device):
    class_info = load_class_config(class_config)
    num_classes = class_info["num_classes"]
    image_ids = read_ids(split_file)
    model = load_model(checkpoint, num_classes, backbone, downsample_factor, device)
    hist = np.zeros((num_classes, num_classes), dtype=np.float64)

    for index, image_id in enumerate(image_ids, start=1):
        image = Image.open(dataset_path / "JPEGImages" / f"{image_id}.jpg")
        mask = Image.open(dataset_path / "SegmentationClass" / f"{image_id}.png")
        prediction = predict(model, image, input_shape, device)
        target = np.asarray(mask.resize(tuple(input_shape), Image.NEAREST))
        if target.ndim == 3:
            target = target[:, :, 0]
        hist += fast_hist(target.reshape(-1), prediction.reshape(-1), num_classes)
        if index % 25 == 0 or index == len(image_ids):
            print(f"Evaluated {index}/{len(image_ids)}")

    ious = per_class_iu(hist)
    recalls = per_class_PA_Recall(hist)
    precisions = per_class_Precision(hist)
    ignored = class_info.get("eval_ignore_class_ids", [])
    paper_ious = [value for index, value in enumerate(ious) if index not in ignored]
    return {
        "dataset_path": str(dataset_path.resolve()),
        "split_file": str(Path(split_file).resolve()),
        "checkpoint": str(Path(checkpoint).resolve()),
        "sample_lines": len(image_ids),
        "unique_samples": len(set(image_ids)),
        "class_names": class_info["classes"],
        "eval_ignore_class_ids": ignored,
        "iou_percent": {
            name: float(value * 100) for name, value in zip(class_info["classes"], ious)
        },
        "recall_percent": {
            name: float(value * 100) for name, value in zip(class_info["classes"], recalls)
        },
        "precision_percent": {
            name: float(value * 100) for name, value in zip(class_info["classes"], precisions)
        },
        "miou_including_background": float(np.nanmean(ious) * 100),
        "miou_excluding_background": float(np.nanmean(ious[1:]) * 100),
        "paper_miou": float(np.nanmean(paper_ious) * 100),
        "pixel_accuracy": float(per_Accuracy(hist) * 100),
        "confusion_matrix": hist.astype(np.int64).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--split-file", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--class-config", default="configs/classes.facadewhu.json")
    parser.add_argument("--input-shape", type=parse_shape, default=[512, 512])
    parser.add_argument("--backbone", default="mobilenet")
    parser.add_argument("--downsample-factor", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_path = resolve_path(args.dataset_path)
    split_file = resolve_path(args.split_file) if args.split_file else dataset_path / "txt" / "val.txt"
    checkpoint = resolve_path(args.checkpoint)
    output = resolve_path(args.output)
    result = evaluate(
        dataset_path,
        split_file,
        checkpoint,
        resolve_path(args.class_config),
        args.input_shape,
        args.backbone,
        args.downsample_factor,
        torch.device(args.device),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"Paper mIoU: {result['paper_miou']:.4f}; "
        f"including background: {result['miou_including_background']:.4f}"
    )
    print(f"Evaluation report: {output}")


if __name__ == "__main__":
    main()
