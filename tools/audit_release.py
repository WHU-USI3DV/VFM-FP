"""Audit the workspace for files that should not be published.

This script is intentionally read-only. It reports likely local artifacts,
large files, model weights, datasets, caches, and generated outputs that should
stay outside the public repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
}

ARTIFACT_DIR_NAMES = {
    "__pycache__",
    ".ipynb_checkpoints",
    ".vscode",
    "logs",
    "debug",
    "model_data",
    "VOCdevkit",
    "FacadeWHU_origin",
    "facadewhu",
    "facadewhu_extend",
    "wuhan_test",
    "deeplab_features",
    "DINO_feature",
    "pca_visualizations",
    "low_quality_pic",
    "check_quality",
    "Paper_save",
}

LARGE_OR_BINARY_EXTENSIONS = {
    ".pth",
    ".pt",
    ".ckpt",
    ".bin",
    ".tar",
    ".zip",
    ".rar",
    ".7z",
    ".safetensors",
}

GENERATED_MEDIA_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def is_inside_named_dir(path: Path, names: set[str]) -> str | None:
    for part in path.parts:
        if part in names:
            return part
    return None


def iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in DEFAULT_SKIP_DIRS for part in path.parts):
            continue
        yield path


def audit(root: Path, large_mb: int) -> list[tuple[str, Path, int]]:
    findings: list[tuple[str, Path, int]] = []
    large_bytes = large_mb * 1024 * 1024

    for path in iter_files(root):
        rel = path.relative_to(root)
        size = path.stat().st_size
        suffix = path.suffix.lower()

        artifact_dir = is_inside_named_dir(rel, ARTIFACT_DIR_NAMES)
        if artifact_dir:
            findings.append((f"artifact-dir:{artifact_dir}", rel, size))
            continue

        if suffix in LARGE_OR_BINARY_EXTENSIONS:
            findings.append((f"large-binary:{suffix}", rel, size))
            continue

        if suffix in GENERATED_MEDIA_EXTENSIONS:
            findings.append((f"media:{suffix}", rel, size))
            continue

        if size >= large_bytes:
            findings.append((f"large-file:>={large_mb}MB", rel, size))

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument("--large-mb", type=int, default=10, help="Large file threshold.")
    parser.add_argument("--limit", type=int, default=200, help="Maximum findings to print.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    findings = audit(root, args.large_mb)

    print(f"Scanned: {root}")
    print(f"Findings: {len(findings)}")
    for reason, rel, size in findings[: args.limit]:
        print(f"{reason}\t{size}\t{rel}")

    if len(findings) > args.limit:
        print(f"... {len(findings) - args.limit} more findings omitted")

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
