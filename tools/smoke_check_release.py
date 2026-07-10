"""Validate the public VFM-FP release tree.

The check is intentionally dependency-light. It verifies required public files,
JSON syntax, and absence of datasets, unexpected checkpoints, generated media, caches, and
shell-specific public tooling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = [
    "README.md",
    ".gitignore",
    ".gitattributes",
    "configs/classes.facadewhu.json",
    "configs/classes.ecp.json",
    "configs/paths.example.json",
    "requirements/segmentation.txt",
    "requirements/sda.txt",
    "docs/citation_template.md",
    "docs/data_and_weights.md",
    "docs/legacy_notes.md",
    "docs/licensing.md",
    "SDA/README.md",
    "SDA/prepare_vcfs_augmented_dataset.py",
    "SDA/DINO_extract/dino_rank_generated.py",
    "SDA/DINO_extract/semantic_consistency_filter.py",
    "SDA/diffusion/semantic_diffusion_augmentation.py",
    "SDA/diffusion/Mul_Ab_norway.py",
    "VCFS/README.md",
    "VCFS/README_VFMFP.md",
    "VCFS/train.py",
    "VCFS/predict.py",
    "VCFS/get_miou.py",
    "VCFS/deeplab.py",
    "VCFS/benchmark.py",
    "VCFS/model_data/deeplab_mobilenetv2.pth",
    "VCFS/nets/deeplabv3_plus.py",
    "VCFS/utils/dataloader_latest.py",
    "VCFS/utils/split_utils.py",
    "VCFS/utils/class_config.py",
    "VCFS/utils/fea_upscale.py",
    "VCFS/voc_annotation.py",
    "tools/audit_release.py",
    "tools/smoke_check_release.py",
    "tools/syntax_check_release.py",
]

JSON_FILES = [
    "configs/classes.facadewhu.json",
    "configs/classes.ecp.json",
    "configs/paths.example.json",
]

FORBIDDEN_DIR_NAMES = {
    "__pycache__",
    ".ipynb_checkpoints",
    "logs",
    "debug",
    "VOCdevkit",
    "FacadeWHU_origin",
    "facadewhu",
    "facadewhu_extend",
    "wuhan_test",
    "4_nyl_zip",
}

FORBIDDEN_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pth",
    ".pt",
    ".ckpt",
    ".onnx",
    ".bin",
    ".safetensors",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".npy",
    ".npz",
    ".p" + "s1",
}

TEXT_EXTENSIONS = {".md", ".txt", ".json"}

ALLOWED_BINARY_FILES = {
    Path("VCFS/model_data/deeplab_mobilenetv2.pth"),
}


def is_under_git(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return relative.parts[:1] == (".git",)


def validate(root: Path) -> list[str]:
    failures: list[str] = []

    for rel in REQUIRED_FILES:
        if not (root / rel).is_file():
            failures.append(f"Missing file: {rel}")

    for rel in JSON_FILES:
        path = root / rel
        if path.exists():
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 - report JSON read/parse failures.
                failures.append(f"Invalid JSON: {rel}: {exc}")

    for path in root.rglob("*"):
        if is_under_git(path, root):
            continue
        if path.is_dir() and path.name in FORBIDDEN_DIR_NAMES:
            failures.append(f"Forbidden directory: {path.relative_to(root)}")
        rel_path = path.relative_to(root) if path.is_file() else None
        if (
            path.is_file()
            and path.suffix.lower() in FORBIDDEN_EXTENSIONS
            and rel_path not in ALLOWED_BINARY_FILES
        ):
            failures.append(f"Forbidden file: {path.relative_to(root)}")
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            shell_name = "power" + "shell"
            if shell_name in text:
                failures.append(f"Shell-specific reference in public text: {path.relative_to(root)}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    failures = validate(root)
    if failures:
        print("Smoke check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

