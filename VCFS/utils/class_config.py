"""Class-configuration helpers for VCFS scripts."""

import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLASS_CONFIG = REPO_ROOT / "configs" / "classes.facadewhu.json"


def resolve_config_path(path=None):
    value = path or os.environ.get("VCFS_CLASS_CONFIG")
    if not value:
        return DEFAULT_CLASS_CONFIG
    config_path = Path(value)
    if not config_path.is_absolute():
        cwd_candidate = Path.cwd() / config_path
        repo_candidate = REPO_ROOT / config_path
        config_path = cwd_candidate if cwd_candidate.exists() else repo_candidate
    return config_path


def load_class_config(path=None):
    config_path = resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    classes = config.get("classes") or config.get("name_classes")
    if not classes:
        raise ValueError(f"Class config has no classes list: {config_path}")

    num_classes = int(config.get("num_classes", len(classes)))
    if num_classes != len(classes):
        raise ValueError(
            f"num_classes ({num_classes}) does not match classes length ({len(classes)}) in {config_path}"
        )

    colors = config.get("colors_rgb") or []
    if colors and len(colors) < num_classes:
        raise ValueError(
            f"colors_rgb has {len(colors)} colors but num_classes is {num_classes} in {config_path}"
        )

    config["num_classes"] = num_classes
    config["classes"] = classes
    config["path"] = str(config_path)
    return config