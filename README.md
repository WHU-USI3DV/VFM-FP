# VFM-FP


**Leveraging Visual Foundation Model Priors for Facade Parsing from Street View Images**

Zhe Chen, Qingwen Tan, Fuxun Liang, Chen Long, Pangyin Li, Zhiming Liu, Junjie Chen, and Zhen Dong

[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1566253526004719) | [DOI](https://doi.org/10.1016/j.inffus.2026.104593)

## Current Release Scope

This repository currently releases **VCFS**, the VFM-FP semantic segmentation network architecture.

## Repository Layout

```text
VCFS/
|-- nets/
|   |-- deeplabv3_plus.py      # VCFS architecture
|   |-- my_attention.py        # cross-attention fusion block
|   |-- mobilenetv2.py         # MobileNetV2 backbone
|   `-- xception.py            # legacy DeepLabv3+ backbone
|-- utils/
|   |-- fea_upscale.py         # DINOv2 token projection
|   |-- train_config.py        # training config loader
|   |-- class_config.py        # class metadata loader
|   |-- dataloader_latest.py   # VOC-style dataset loader
|   `-- utils_metrics.py       # segmentation metrics
|-- train.py                   # training entry point
|-- evaluate.py                # compact checkpoint evaluator
|-- predict.py                 # interactive/image/video inference
|-- get_miou.py                # legacy mIoU evaluator
`-- benchmark.py               # parameter/FLOP/FPS report

configs/
|-- classes.facadewhu.json     # FacadeWHU class metadata
|-- classes.ecp.json           # ECP class metadata example
`-- vcfs.facadewhu.json        # default VCFS training config
```

## Environment

Install the Python dependencies for VCFS:

```bash
pip install -r VCFS/requirements.txt
```

VCFS loads DINOv2 through `torch.hub` on first use. Make sure the runtime can
access the DINOv2 weights or has them cached locally.

## Dataset Format

Training uses a VOC-style semantic segmentation dataset:

```text
<dataset_name>/
|-- JPEGImages/
|-- SegmentationClass/
`-- txt/
    |-- train.txt
    |-- val.txt
    |-- trainval.txt
    `-- test.txt
```

The default FacadeWHU class order is:

```text
background, window, door, facade, balcony, roof, shop
```

Background is ignored for the paper mIoU average through
`configs/classes.facadewhu.json`.

## Training

From the repository root:

```bash
cd VCFS
python train.py --config ../configs/vcfs.facadewhu.json
```

Common overrides:

```bash
python train.py \
  --config ../configs/vcfs.facadewhu.json \
  --dataset-path facadewhu_extend \
  --class-config ../configs/classes.facadewhu.json \
  --batch-size 4 \
  --fp16 true
```

## Evaluation

Use `evaluate.py` for compact JSON evaluation without keeping per-image
prediction masks:

```bash
cd VCFS
python evaluate.py \
  --dataset-path facadewhu_extend \
  --checkpoint results/run_name/best_epoch_weights.pth \
  --class-config ../configs/classes.facadewhu.json \
  --output results/run_name/eval_best.json
```