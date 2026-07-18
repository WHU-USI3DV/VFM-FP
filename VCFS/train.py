import argparse
import datetime
import hashlib
import json
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

if os.environ.get("VCFS_CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["VCFS_CUDA_VISIBLE_DEVICES"]

from nets.deeplabv3_plus import DeepLab, VCFS_ARCHITECTURE_VERSION
from nets.deeplabv3_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader_latest import DeeplabDataset, deeplab_dataset_collate
from utils.split_utils import load_split_lines
from utils.train_config import load_train_config
from utils.utils import download_weights, seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="Train the VCFS facade parser.")
    parser.add_argument("--config", default=None, help="Training config JSON. Defaults to ../configs/vcfs.facadewhu.json.")
    parser.add_argument("--class-config", default=None, help="Override dataset class JSON from the training config.")
    parser.add_argument("--dataset-path", default=None, help="Override VOC-style dataset directory.")
    parser.add_argument("--model-path", default=None, help="Override checkpoint path. Use an empty string to skip loading.")
    parser.add_argument("--input-shape", default=None, help="Override input size, for example 512,512 or 512x512.")
    parser.add_argument("--epochs", type=int, default=None, help="Override total training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override unfreeze-stage batch size.")
    parser.add_argument("--freeze-batch-size", type=int, default=None, help="Override freeze-stage batch size.")
    parser.add_argument("--save-dir", default=None, help="Override training output directory.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader worker count.")
    parser.add_argument("--num-classes", type=int, default=None, help="Override class count.")
    parser.add_argument("--eval-flag", choices=("true", "false"), default=None, help="Override periodic mIoU evaluation.")
    parser.add_argument("--fp16", choices=("true", "false"), default=None, help="Override mixed precision training.")
    return parser.parse_known_args()[0]


def setup_device(cfg):
    ngpus = torch.cuda.device_count()
    if cfg.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] rank={rank}, local_rank={local_rank}, gpu_count={ngpus}")
        return device, local_rank, rank, ngpus

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, 0, 0, ngpus


def load_partial_weights(model, model_path, device, local_rank):
    if not model_path:
        return

    if local_rank == 0:
        print(f"Load weights {model_path}.")

    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    load_key, no_load_key, temp_dict = [], [], {}

    for key, value in pretrained_dict.items():
        if key in model_dict and np.shape(model_dict[key]) == np.shape(value):
            temp_dict[key] = value
            load_key.append(key)
        else:
            no_load_key.append(key)

    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)

    if local_rank == 0:
        print(f"Successful Load Key Num: {len(load_key)}")
        print(f"Fail To Load Key Num: {len(no_load_key)}")
        if no_load_key:
            print("Unloaded keys are expected when the class head shape differs.")


def prepare_model(cfg, device, local_rank, ngpus):
    if cfg.pretrained:
        if cfg.distributed:
            if local_rank == 0:
                download_weights(cfg.backbone)
            dist.barrier()
        else:
            download_weights(cfg.backbone)

    model = DeepLab(
        num_classes=cfg.num_classes,
        backbone=cfg.backbone,
        downsample_factor=cfg.downsample_factor,
        pretrained=cfg.pretrained,
    )
    if not cfg.pretrained:
        weights_init(model)

    load_partial_weights(model, cfg.model_path, device, local_rank)

    model_train = model.train()
    if cfg.sync_bn and ngpus > 1 and cfg.distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif cfg.sync_bn:
        print("SyncBatchNorm requires distributed training with more than one GPU.")

    if cfg.cuda:
        if cfg.distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train,
                device_ids=[local_rank],
                find_unused_parameters=True,
            )
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    return model, model_train


def validate_split_contract(cfg, train_lines, val_lines):
    train_ids = [line.strip().split()[0] for line in train_lines if line.strip()]
    val_ids = [line.strip().split()[0] for line in val_lines if line.strip()]

    if cfg.require_unique_split_ids:
        duplicate_train = len(train_ids) - len(set(train_ids))
        duplicate_val = len(val_ids) - len(set(val_ids))
        if duplicate_train or duplicate_val:
            raise ValueError(
                "Split files must contain unique ids for this experiment; "
                f"train duplicates={duplicate_train}, val duplicates={duplicate_val}."
            )

    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        preview = ", ".join(overlap[:10])
        raise ValueError(
            f"Train split and val split overlap: {len(overlap)} ids, e.g. {preview}. "
            "Regenerate splits so held-out val/test ids are not used for training."
        )

    synthetic_count = sum(image_id.startswith(cfg.synthetic_prefix) for image_id in train_ids)
    original_count = len(train_ids) - synthetic_count
    if (
        cfg.expected_original_train_samples is not None
        and original_count != cfg.expected_original_train_samples
    ):
        raise ValueError(
            "Original training sample count does not match the configured split contract: "
            f"expected {cfg.expected_original_train_samples}, got {original_count}."
        )
    if cfg.expected_val_samples is not None and len(val_ids) != cfg.expected_val_samples:
        raise ValueError(
            "Validation sample count does not match the configured split contract: "
            f"expected {cfg.expected_val_samples}, got {len(val_ids)}."
        )
    if synthetic_count < cfg.minimum_synthetic_train_samples:
        raise ValueError(
            "SDA training split contains too few synthetic samples: "
            f"required at least {cfg.minimum_synthetic_train_samples}, got {synthetic_count}."
        )
    return {
        "original_train": original_count,
        "synthetic_train": synthetic_count,
        "val": len(val_ids),
    }


def load_dataset_splits(cfg, local_rank):
    train_lines, train_split = load_split_lines(
        cfg.dataset_path,
        cfg.train_split,
        fallback_name=cfg.train_fallback_split,
        auto_create_train=cfg.auto_create_train_split,
    )
    val_lines, val_split = load_split_lines(
        cfg.dataset_path,
        cfg.val_split,
        fallback_name=cfg.val_fallback_split,
    )

    split_counts = validate_split_contract(cfg, train_lines, val_lines)

    train_limit = os.environ.get("VCFS_TRAIN_LIMIT")
    val_limit = os.environ.get("VCFS_VAL_LIMIT")
    if train_limit:
        train_lines = train_lines[: int(train_limit)]
    if val_limit:
        val_lines = val_lines[: int(val_limit)]

    if local_rank == 0:
        print(f"Train split: {train_split}")
        print(f"Val split: {val_split}")
        print(
            "Full split contract: "
            f"original={split_counts['original_train']}, "
            f"synthetic={split_counts['synthetic_train']}, val={split_counts['val']}"
        )
        print(f"Train samples: {len(train_lines)}, val samples: {len(val_lines)}")

    return train_lines, val_lines, train_split, val_split


def sha256_file(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_training_manifest(cfg, class_config, train_split, val_split, num_train, num_val):
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_audit = Path(cfg.dataset_path) / "txt" / "sda_audit.json"
    files = {
        "train_config": cfg.train_config_path,
        "class_config": class_config["path"],
        "initial_checkpoint": cfg.model_path,
        "train_split": train_split,
        "val_split": val_split,
        "vcfs_network": Path(__file__).resolve().parent / "nets" / "deeplabv3_plus.py",
        "vcfs_attention": Path(__file__).resolve().parent / "nets" / "my_attention.py",
        "vcfs_dino_projection": Path(__file__).resolve().parent / "utils" / "fea_upscale.py",
    }
    if dataset_audit.exists():
        files["dataset_audit"] = str(dataset_audit)

    manifest = {
        "manifest_version": 1,
        "vcfs_architecture_version": VCFS_ARCHITECTURE_VERSION,
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "resolved_config": asdict(cfg),
        "class_names": class_config["classes"],
        "eval_ignore_class_ids": class_config.get("eval_ignore_class_ids", []),
        "num_train": num_train,
        "num_val": num_val,
        "files": {
            name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
            for name, path in files.items()
            if path and Path(path).exists()
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
    }
    destination = save_dir / "training_manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(f"Training manifest: {destination}")


def scaled_lr(cfg, batch_size):
    if cfg.lr_scaling_mode == "fixed":
        return cfg.init_lr, cfg.min_lr

    nominal_batch_size = 16
    lr_limit_max = 5e-4 if cfg.optimizer_type == "adam" else 1e-1
    lr_limit_min = 3e-4 if cfg.optimizer_type == "adam" else 5e-4
    if cfg.backbone == "xception":
        lr_limit_max = 1e-4 if cfg.optimizer_type == "adam" else 1e-1
        lr_limit_min = 1e-4 if cfg.optimizer_type == "adam" else 5e-4

    init_lr_fit = min(max(batch_size / nominal_batch_size * cfg.init_lr, lr_limit_min), lr_limit_max)
    min_lr_fit = min(
        max(batch_size / nominal_batch_size * cfg.min_lr, lr_limit_min * 1e-2),
        lr_limit_max * 1e-2,
    )
    return init_lr_fit, min_lr_fit


def make_optimizer(cfg, model, init_lr_fit):
    if cfg.optimizer_type == "adam":
        return optim.Adam(model.parameters(), init_lr_fit, betas=(cfg.momentum, 0.999), weight_decay=cfg.weight_decay)
    if cfg.optimizer_type == "sgd":
        return optim.SGD(model.parameters(), init_lr_fit, momentum=cfg.momentum, nesterov=True, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer_type}")


def make_loaders(cfg, train_dataset, val_dataset, train_sampler, val_sampler, batch_size, shuffle, rank):
    common = {
        "num_workers": cfg.num_workers,
        "pin_memory": True,
        "drop_last": True,
        "collate_fn": deeplab_dataset_collate,
        "worker_init_fn": partial(worker_init_fn, rank=rank, seed=cfg.seed),
    }
    train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, sampler=train_sampler, **common)
    val_loader = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, sampler=val_sampler, **common)
    return train_loader, val_loader


def show_training_config(cfg, class_config, num_train, num_val, local_rank):
    if local_rank != 0:
        return
    show_config(
        train_config=cfg.train_config_path,
        vcfs_architecture_version=VCFS_ARCHITECTURE_VERSION,
        num_classes=cfg.num_classes,
        class_config=class_config["path"],
        backbone=cfg.backbone,
        model_path=cfg.model_path,
        input_shape=cfg.input_shape,
        Init_Epoch=cfg.init_epoch,
        Freeze_Epoch=cfg.freeze_epoch,
        UnFreeze_Epoch=cfg.epochs,
        Freeze_batch_size=cfg.freeze_batch_size,
        Unfreeze_batch_size=cfg.batch_size,
        Freeze_Train=cfg.freeze_train,
        Init_lr=cfg.init_lr,
        Min_lr=cfg.min_lr,
        lr_scaling_mode=cfg.lr_scaling_mode,
        augmentation_profile=cfg.augmentation_profile,
        optimizer_type=cfg.optimizer_type,
        momentum=cfg.momentum,
        lr_decay_type=cfg.lr_decay_type,
        save_period=cfg.save_period,
        save_dir=cfg.save_dir,
        num_workers=cfg.num_workers,
        num_train=num_train,
        num_val=num_val,
    )


def warn_if_too_few_steps(cfg, num_train):
    wanted_step = 1.5e4 if cfg.optimizer_type == "sgd" else 0.5e4
    steps_per_epoch = num_train // cfg.batch_size
    total_step = steps_per_epoch * cfg.epochs
    if total_step > wanted_step:
        return
    if steps_per_epoch == 0:
        raise ValueError("Dataset is too small for training. Please add more data.")
    wanted_epoch = wanted_step // steps_per_epoch + 1
    print(f"\n[Warning] With optimizer {cfg.optimizer_type}, recommended total steps are at least {int(wanted_step)}.")
    print(
        "[Warning] Current run: "
        f"num_train={num_train}, batch_size={cfg.batch_size}, epochs={cfg.epochs}, total_steps={total_step}."
    )
    print(f"[Warning] Consider increasing epochs to about {int(wanted_epoch)} for a full experiment.")


def build_training_state(cfg, model, rank, ngpus):
    if cfg.freeze_train:
        for param in model.backbone.parameters():
            param.requires_grad = False

    batch_size = cfg.freeze_batch_size if cfg.freeze_train else cfg.batch_size
    init_lr_fit, min_lr_fit = scaled_lr(cfg, batch_size)
    optimizer = make_optimizer(cfg, model, init_lr_fit)
    lr_scheduler = get_lr_scheduler(cfg.lr_decay_type, init_lr_fit, min_lr_fit, cfg.epochs)

    return batch_size, optimizer, lr_scheduler


def build_datasets_and_loaders(cfg, train_lines, val_lines, batch_size, rank, ngpus):
    train_dataset = DeeplabDataset(
        train_lines,
        cfg.input_shape,
        cfg.num_classes,
        True,
        cfg.dataset_path,
        augmentation_profile=cfg.augmentation_profile,
    )
    val_dataset = DeeplabDataset(
        val_lines,
        cfg.input_shape,
        cfg.num_classes,
        False,
        cfg.dataset_path,
        augmentation_profile=cfg.augmentation_profile,
    )

    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        batch_size = batch_size // ngpus
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    epoch_step = len(train_lines) // batch_size
    epoch_step_val = len(val_lines) // batch_size
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("Dataset is too small for training. Please add more data.")

    train_loader, val_loader = make_loaders(
        cfg, train_dataset, val_dataset, train_sampler, val_sampler, batch_size, shuffle, rank
    )
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_sampler": train_sampler,
        "val_sampler": val_sampler,
        "shuffle": shuffle,
        "batch_size": batch_size,
        "epoch_step": epoch_step,
        "epoch_step_val": epoch_step_val,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


def rebuild_loaders_after_unfreeze(cfg, model, loader_state, rank, ngpus):
    batch_size = cfg.batch_size
    init_lr_fit, min_lr_fit = scaled_lr(cfg, batch_size)
    lr_scheduler = get_lr_scheduler(cfg.lr_decay_type, init_lr_fit, min_lr_fit, cfg.epochs)

    for param in model.backbone.parameters():
        param.requires_grad = True

    effective_batch_size = batch_size // ngpus if cfg.distributed else batch_size
    epoch_step = len(loader_state["train_dataset"]) // effective_batch_size
    epoch_step_val = len(loader_state["val_dataset"]) // effective_batch_size
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("Dataset is too small for training. Please add more data.")

    train_loader, val_loader = make_loaders(
        cfg,
        loader_state["train_dataset"],
        loader_state["val_dataset"],
        loader_state["train_sampler"],
        loader_state["val_sampler"],
        effective_batch_size,
        loader_state["shuffle"],
        rank,
    )
    loader_state.update(
        {
            "batch_size": effective_batch_size,
            "epoch_step": epoch_step,
            "epoch_step_val": epoch_step_val,
            "train_loader": train_loader,
            "val_loader": val_loader,
        }
    )
    return lr_scheduler


def create_callbacks(cfg, class_config, model, val_lines, local_rank):
    if local_rank != 0:
        return None, None
    time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
    log_dir = str(Path(cfg.save_dir) / f"loss_{time_str}")
    loss_history = LossHistory(log_dir, model, input_shape=cfg.input_shape)
    eval_callback = EvalCallback(
        model,
        cfg.input_shape,
        cfg.num_classes,
        val_lines,
        cfg.dataset_path,
        log_dir,
        cfg.cuda,
        eval_flag=cfg.eval_flag,
        period=cfg.eval_period,
        ignore_class_ids=class_config.get("eval_ignore_class_ids", []),
    )
    return loss_history, eval_callback


def train(cfg, class_config):
    print(f"Using train config: {cfg.train_config_path}")
    print(f"Using class config: {class_config['path']}")
    seed_everything(cfg.seed)

    device, local_rank, rank, ngpus = setup_device(cfg)
    train_lines, val_lines, train_split, val_split = load_dataset_splits(cfg, local_rank)
    model, model_train = prepare_model(cfg, device, local_rank, ngpus)
    show_training_config(cfg, class_config, len(train_lines), len(val_lines), local_rank)
    if local_rank == 0:
        warn_if_too_few_steps(cfg, len(train_lines))
        write_training_manifest(
            cfg,
            class_config,
            train_split,
            val_split,
            len(train_lines),
            len(val_lines),
        )

    loss_history, eval_callback = create_callbacks(cfg, class_config, model, val_lines, local_rank)
    scaler = torch.cuda.amp.GradScaler() if cfg.fp16 else None

    batch_size, optimizer, lr_scheduler = build_training_state(cfg, model, rank, ngpus)
    loader_state = build_datasets_and_loaders(cfg, train_lines, val_lines, batch_size, rank, ngpus)
    class_weights = np.ones([cfg.num_classes], np.float32)
    unfrozen = False

    for epoch in range(cfg.init_epoch, cfg.epochs):
        if epoch >= cfg.freeze_epoch and not unfrozen and cfg.freeze_train:
            lr_scheduler = rebuild_loaders_after_unfreeze(cfg, model, loader_state, rank, ngpus)
            unfrozen = True

        if cfg.distributed:
            loader_state["train_sampler"].set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler, epoch)
        fit_one_epoch(
            model_train,
            model,
            loss_history,
            eval_callback,
            optimizer,
            epoch,
            loader_state["epoch_step"],
            loader_state["epoch_step_val"],
            loader_state["train_loader"],
            loader_state["val_loader"],
            cfg.epochs,
            cfg.cuda,
            cfg.dice_loss,
            cfg.focal_loss,
            cfg.inverfreq_loss,
            class_weights,
            cfg.num_classes,
            cfg.fp16,
            scaler,
            cfg.save_period,
            cfg.save_dir,
            local_rank,
        )

        if cfg.distributed:
            dist.barrier()

    if local_rank == 0:
        loss_history.writer.close()


def main():
    cfg, class_config = load_train_config(parse_args())
    train(cfg, class_config)


if __name__ == "__main__":
    main()
