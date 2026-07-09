"""Prepare SCF-retained SDA samples for VCFS training.

The diffusion stage writes synthetic image ids and their source image ids to
``synthetic_pairs.csv``. This helper copies retained synthetic images into a
VOC-style VCFS dataset and assigns each synthetic image the source mask used by
ControlNet generation.
"""

import argparse
import csv
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def read_keep_ids(path):
    keep_ids = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            keep_ids.append(line.replace(",", " ").split()[0])
    return keep_ids


def read_pair_record(path):
    pairs = {}
    with open(path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 2:
                continue
            syn_id, source_id = row[0].strip(), row[1].strip()
            if syn_id and source_id and syn_id != "syn_id":
                pairs[syn_id] = source_id
    return pairs


def find_image(image_dir, image_id):
    image_dir = Path(image_dir)
    for ext in IMAGE_EXTENSIONS:
        path = image_dir / f"{image_id}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find image for id {image_id} under {image_dir}")


def read_names(path):
    path = Path(path)
    if not path.exists():
        return []
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if parts:
            names.append(parts[0])
    return names


def write_names(path, names):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")


def prepare_dataset(args):
    keep_ids = read_keep_ids(args.scf_keep)
    pairs = read_pair_record(args.pair_record)

    target = Path(args.target_dataset)
    image_out = target / "JPEGImages"
    mask_out = target / "SegmentationClass"
    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    copied = []
    missing_pairs = []
    for syn_id in keep_ids:
        source_id = pairs.get(syn_id)
        if source_id is None:
            missing_pairs.append(syn_id)
            continue

        syn_image = find_image(args.synthetic_image_dir, syn_id)
        source_mask = Path(args.source_mask_dir) / f"{source_id}.png"
        if not source_mask.exists():
            raise FileNotFoundError(f"Could not find source mask: {source_mask}")

        shutil.copy2(syn_image, image_out / f"{syn_id}{syn_image.suffix.lower()}")
        shutil.copy2(source_mask, mask_out / f"{syn_id}.png")
        copied.append(syn_id)

    if missing_pairs:
        preview = ", ".join(missing_pairs[:5])
        raise RuntimeError(
            f"{len(missing_pairs)} retained ids were not found in {args.pair_record}: {preview}"
        )

    split_dir = target / "txt"
    base_train = read_names(split_dir / args.base_train_name)
    val_names = set(read_names(split_dir / args.val_name))
    test_names = set(read_names(split_dir / args.test_name))
    generated = [name for name in copied if name not in val_names and name not in test_names]

    if args.write_train_split:
        merged = []
        seen = set()
        for name in base_train + generated:
            if name not in seen:
                merged.append(name)
                seen.add(name)
        write_names(split_dir / args.output_train_name, merged)

    write_names(split_dir / args.retained_split_name, generated)
    print(f"Copied {len(copied)} SCF-retained synthetic samples into {target}")
    print(f"Wrote retained split: {split_dir / args.retained_split_name}")
    if args.write_train_split:
        print(f"Wrote augmented train split: {split_dir / args.output_train_name}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-image-dir", default="SDA_output/syn_image")
    parser.add_argument("--source-mask-dir", default="FacadeWHU_origin/SegmentationClass")
    parser.add_argument("--scf-keep", default="SDA_output/scf/scf_keep.txt")
    parser.add_argument("--pair-record", default="SDA_output/txt/synthetic_pairs.csv")
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--base-train-name", default="train.txt")
    parser.add_argument("--val-name", default="val.txt")
    parser.add_argument("--test-name", default="test.txt")
    parser.add_argument("--retained-split-name", default="sda_retained.txt")
    parser.add_argument("--output-train-name", default="train_1601.txt")
    parser.add_argument("--write-train-split", action="store_true")
    return parser


if __name__ == "__main__":
    prepare_dataset(build_parser().parse_args())
