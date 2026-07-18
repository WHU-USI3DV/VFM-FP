"""Training configuration helpers for VCFS."""

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

from .class_config import load_class_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs" / "vcfs.facadewhu.json"
TRUE_VALUES = {"1", "true", "yes", "y", "on"}


@dataclass
class TrainConfig:
    class_config: str | None = None
    dataset_path: str = "facadewhu_extend"
    model_path: str = "model_data/deeplab_mobilenetv2.pth"
    num_classes: int | None = None

    backbone: str = "mobilenet"
    pretrained: bool = True
    downsample_factor: int = 8
    input_shape: list[int] | None = None

    seed: int = 11
    cuda: bool = True
    distributed: bool = False
    sync_bn: bool = False
    fp16: bool = False

    init_epoch: int = 0
    freeze_epoch: int = 0
    freeze_batch_size: int = 2
    epochs: int = 200
    batch_size: int = 4
    freeze_train: bool = False

    init_lr: float = 2e-4
    min_lr_ratio: float = 0.01
    lr_scaling_mode: str = "legacy_batch"
    optimizer_type: str = "adam"
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_decay_type: str = "cos"

    save_period: int = 5
    save_dir: str = "results"
    eval_flag: bool = True
    eval_period: int = 5

    train_split: str = "train_1601.txt"
    train_fallback_split: str | None = "train.txt"
    val_split: str = "val.txt"
    val_fallback_split: str | None = "test.txt"
    auto_create_train_split: bool = True
    require_unique_split_ids: bool = False
    expected_original_train_samples: int | None = None
    expected_val_samples: int | None = None
    minimum_synthetic_train_samples: int = 0
    synthetic_prefix: str = "syn_"

    dice_loss: bool = False
    focal_loss: bool = False
    inverfreq_loss: bool = False
    augmentation_profile: str = "legacy_resize_hsv"
    num_workers: int = 4

    def __post_init__(self):
        if self.input_shape is None:
            self.input_shape = [512, 512]

    @property
    def min_lr(self):
        return self.init_lr * self.min_lr_ratio

    @property
    def unfreeze_epoch(self):
        return self.epochs

    @property
    def unfreeze_batch_size(self):
        return self.batch_size


def resolve_repo_path(path):
    if path in (None, ""):
        return path
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)

    cwd_candidate = Path.cwd() / candidate
    repo_candidate = REPO_ROOT / candidate
    vcfs_candidate = REPO_ROOT / "VCFS" / candidate
    for resolved in (cwd_candidate, repo_candidate, vcfs_candidate):
        if resolved.exists():
            return str(resolved)
    return str(candidate)


def resolve_vcfs_path(path):
    if path in (None, ""):
        return path
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(REPO_ROOT / "VCFS" / candidate)


def load_json_config(path=None):
    config_path = Path(resolve_repo_path(path or os.environ.get("VCFS_TRAIN_CONFIG") or DEFAULT_TRAIN_CONFIG))
    with config_path.open("r", encoding="utf-8") as file:
        values = json.load(file)
    return values, str(config_path)


def parse_bool(value, default=None):
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in TRUE_VALUES


def parse_shape(value):
    if value in (None, ""):
        return None
    if isinstance(value, (list, tuple)):
        parts = [int(part) for part in value]
    else:
        parts = [int(part.strip()) for part in str(value).replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("input_shape must be like [512, 512], 512,512, or 512x512")
    return parts


def _coerce_value(field_name, value):
    bool_fields = {
        "cuda",
        "distributed",
        "sync_bn",
        "fp16",
        "pretrained",
        "freeze_train",
        "eval_flag",
        "auto_create_train_split",
        "require_unique_split_ids",
        "dice_loss",
        "focal_loss",
        "inverfreq_loss",
    }
    int_fields = {
        "seed",
        "num_classes",
        "downsample_factor",
        "init_epoch",
        "freeze_epoch",
        "freeze_batch_size",
        "epochs",
        "batch_size",
        "save_period",
        "eval_period",
        "num_workers",
        "expected_original_train_samples",
        "expected_val_samples",
        "minimum_synthetic_train_samples",
    }
    float_fields = {"init_lr", "min_lr_ratio", "momentum", "weight_decay"}

    if field_name == "input_shape":
        return parse_shape(value)
    if field_name in bool_fields:
        return parse_bool(value)
    if field_name in int_fields and value is not None:
        return int(value)
    if field_name in float_fields and value is not None:
        return float(value)
    return value


def apply_overrides(cfg, values):
    field_names = {field.name for field in fields(TrainConfig)}
    aliases = {
        "unfreeze_epoch": "epochs",
        "unfreeze_batch_size": "batch_size",
    }
    for key, value in values.items():
        field_name = aliases.get(key, key)
        if field_name not in field_names or value is None:
            continue
        setattr(cfg, field_name, _coerce_value(field_name, value))
    return cfg


def cli_env_overrides(args):
    raw = {
        "class_config": args.class_config or os.environ.get("VCFS_CLASS_CONFIG"),
        "dataset_path": args.dataset_path if args.dataset_path is not None else os.environ.get("VCFS_DATASET_PATH"),
        "model_path": args.model_path if args.model_path is not None else os.environ.get("VCFS_MODEL_PATH"),
        "input_shape": args.input_shape if args.input_shape is not None else os.environ.get("VCFS_INPUT_SHAPE"),
        "epochs": args.epochs if args.epochs is not None else os.environ.get("VCFS_EPOCHS"),
        "batch_size": args.batch_size if args.batch_size is not None else os.environ.get("VCFS_BATCH_SIZE"),
        "freeze_batch_size": args.freeze_batch_size if args.freeze_batch_size is not None else os.environ.get("VCFS_FREEZE_BATCH_SIZE"),
        "save_dir": args.save_dir if args.save_dir is not None else os.environ.get("VCFS_SAVE_DIR"),
        "num_workers": args.num_workers if args.num_workers is not None else os.environ.get("VCFS_NUM_WORKERS"),
        "num_classes": args.num_classes if args.num_classes is not None else os.environ.get("VCFS_NUM_CLASSES"),
        "eval_flag": args.eval_flag if args.eval_flag is not None else os.environ.get("VCFS_EVAL_FLAG"),
        "fp16": args.fp16 if args.fp16 is not None else os.environ.get("VCFS_FP16"),
    }
    return {key: value for key, value in raw.items() if value not in (None, "")}


def load_train_config(args):
    values, train_config_path = load_json_config(args.config)
    cfg = apply_overrides(TrainConfig(), values)
    cfg = apply_overrides(cfg, cli_env_overrides(args))

    if cfg.lr_scaling_mode not in {"legacy_batch", "fixed"}:
        raise ValueError("lr_scaling_mode must be 'legacy_batch' or 'fixed'")
    if cfg.augmentation_profile not in {"legacy_resize_hsv", "paper_facadewhu"}:
        raise ValueError(
            "augmentation_profile must be 'legacy_resize_hsv' or 'paper_facadewhu'"
        )

    class_config = load_class_config(resolve_repo_path(cfg.class_config))
    if cfg.num_classes is None:
        cfg.num_classes = class_config["num_classes"]

    cfg.dataset_path = resolve_repo_path(cfg.dataset_path)
    cfg.model_path = resolve_repo_path(cfg.model_path)
    cfg.save_dir = resolve_vcfs_path(cfg.save_dir)
    cfg.train_config_path = train_config_path
    return cfg, class_config
