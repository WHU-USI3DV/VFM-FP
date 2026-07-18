"""Create safe VOC-style split files for VCFS training.

The SDA reproduction uses a held-out 10 percent split from the original images.
Synthetic samples must only be appended to the training split and must never be
written to val.txt or test.txt.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def _env_float(name, current):
    value = os.environ.get(name)
    return current if value in (None, "") else float(value)


def _env_int(name, current):
    value = os.environ.get(name)
    return current if value in (None, "") else int(value)


def write_names(path, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")


def split_original_ids(names, test_ratio, seed):
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    rng = random.Random(seed)
    shuffled = list(names)
    rng.shuffle(shuffled)
    test_count = int(round(len(shuffled) * test_ratio))
    train_count = len(shuffled) - test_count
    if train_count <= 0 or test_count <= 0:
        raise ValueError(f"Invalid split: train={train_count}, test={test_count}")

    train = sorted(shuffled[:train_count])
    test = sorted(shuffled[train_count:])
    return train, test


def check_masks(mask_paths):
    class_counts = np.zeros([256], dtype=np.int64)
    for mask_path in tqdm(mask_paths, desc="Checking masks"):
        mask = np.array(Image.open(mask_path), np.uint8)
        if mask.ndim > 2:
            raise ValueError(f"Mask must be single-channel id image: {mask_path}")
        class_counts += np.bincount(mask.reshape(-1), minlength=256)

    used_ids = [index for index, count in enumerate(class_counts) if count > 0]
    print("Used label ids:", used_ids)
    if class_counts[255] > 0 and class_counts[0] > 0 and np.sum(class_counts[1:255]) == 0:
        print("Warning: masks contain only 0 and 255. Convert binary labels to class ids before training.")
    elif class_counts[0] > 0 and np.sum(class_counts[1:]) == 0:
        print("Warning: masks contain only background pixels.")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=os.environ.get("VCFS_DATASET_PATH", "facadewhu_extend"))
    parser.add_argument("--test-ratio", type=float, default=_env_float("VCFS_TEST_RATIO", 0.1))
    parser.add_argument("--seed", type=int, default=_env_int("VCFS_SPLIT_SEED", 11))
    parser.add_argument("--synthetic-prefix", default=os.environ.get("VCFS_SYNTHETIC_PREFIX", "syn_"))
    parser.add_argument("--skip-mask-check", action="store_true")
    return parser


def main(args=None):
    args = build_parser().parse_args(args)
    dataset_path = Path(args.dataset_path)
    seg_dir = dataset_path / "SegmentationClass"
    split_dir = dataset_path / "txt"
    split_dir.mkdir(parents=True, exist_ok=True)

    if not seg_dir.exists():
        raise FileNotFoundError(f"SegmentationClass directory not found: {seg_dir}")

    masks = sorted(path for path in seg_dir.iterdir() if path.suffix.lower() == ".png")
    all_ids = [path.stem for path in masks]
    synthetic_ids = sorted(image_id for image_id in all_ids if image_id.startswith(args.synthetic_prefix))
    original_ids = sorted(image_id for image_id in all_ids if not image_id.startswith(args.synthetic_prefix))

    train_original, held_out = split_original_ids(original_ids, args.test_ratio, args.seed)
    augmented_train = train_original + synthetic_ids

    write_names(split_dir / "train.txt", train_original)
    write_names(split_dir / "trainval.txt", train_original)
    write_names(split_dir / "val.txt", held_out)
    write_names(split_dir / "test.txt", held_out)
    write_names(split_dir / "train_1601.txt", augmented_train)

    print(f"Dataset: {dataset_path}")
    print(f"Masks: {len(masks)} original: {len(original_ids)} synthetic: {len(synthetic_ids)}")
    print(
        "Wrote train/trainval: {} val/test: {} train_1601: {}".format(
            len(train_original), len(held_out), len(augmented_train)
        )
    )

    if not args.skip_mask_check:
        check_masks(masks)


if __name__ == "__main__":
    main()
