"""Create VOC-style train/val/test split files for a segmentation dataset."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def _env_float(name, current):
    value = os.environ.get(name)
    return current if value in (None, "") else float(value)


def main():
    dataset_path = Path(os.environ.get("VCFS_DATASET_PATH", "facadewhu_extend"))
    trainval_percent = _env_float("VCFS_TRAINVAL_PERCENT", 1.0)
    train_percent = _env_float("VCFS_TRAIN_PERCENT", 0.9)

    seg_dir = dataset_path / "SegmentationClass"
    split_dir = dataset_path / "txt"
    split_dir.mkdir(parents=True, exist_ok=True)

    if not seg_dir.exists():
        raise FileNotFoundError(f"SegmentationClass directory not found: {seg_dir}")

    random.seed(0)
    masks = sorted(path for path in seg_dir.iterdir() if path.suffix.lower() == ".png")
    total = len(masks)
    trainval_count = int(total * trainval_percent)
    train_count = int(trainval_count * train_percent)
    indices = list(range(total))
    trainval_indices = set(random.sample(indices, trainval_count))
    train_indices = set(random.sample(sorted(trainval_indices), train_count))

    splits = {
        "trainval.txt": [],
        "train.txt": [],
        "val.txt": [],
        "test.txt": [],
    }
    for index, mask_path in enumerate(masks):
        image_id = mask_path.stem
        if index in trainval_indices:
            splits["trainval.txt"].append(image_id)
            target = "train.txt" if index in train_indices else "val.txt"
            splits[target].append(image_id)
        else:
            splits["test.txt"].append(image_id)

    for name, values in splits.items():
        (split_dir / name).write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")

    print(f"Dataset: {dataset_path}")
    print(f"Masks: {total}, trainval: {len(splits['trainval.txt'])}, train: {len(splits['train.txt'])}, val: {len(splits['val.txt'])}, test: {len(splits['test.txt'])}")

    class_counts = np.zeros([256], dtype=np.int64)
    for mask_path in tqdm(masks, desc="Checking masks"):
        mask = np.array(Image.open(mask_path), np.uint8)
        if mask.ndim > 2:
            raise ValueError(f"Mask must be single-channel id image: {mask_path}")
        class_counts += np.bincount(mask.reshape(-1), minlength=256)

    used_ids = [index for index, count in enumerate(class_counts) if count > 0]
    print("Used label ids:", used_ids)
    if class_counts[255] > 0 and class_counts[0] > 0 and np.sum(class_counts[1:255]) == 0:
        print("Warning: masks contain only 0 and 255. Convert binary labels to class ids 0 and 1 before training.")
    elif class_counts[0] > 0 and np.sum(class_counts[1:]) == 0:
        print("Warning: masks contain only background pixels.")


if __name__ == "__main__":
    main()