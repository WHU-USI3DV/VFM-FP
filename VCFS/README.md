# VCFS: VFM-FP Segmentation Network

This directory contains the VCFS code released for the VFM-FP facade parsing
network. VCFS keeps the DeepLabv3+ decoder and MobileNetV2 backbone, then fuses
frozen DINOv2 visual foundation model features into multiple backbone stages
through trainable projection and cross-attention blocks.

SDA data generation code is intentionally kept out of this public entry point.
Use this folder for model definition, training, inference, and evaluation.

## Directory Layout

```text
VCFS/
|-- nets/
|   |-- deeplabv3_plus.py      # VCFS architecture
|   |-- my_attention.py        # cross-attention fusion block
|   |-- mobilenetv2.py         # MobileNetV2 backbone
|   `-- xception.py            # legacy DeepLabv3+ backbone
|-- utils/
|   |-- fea_upscale.py         # DINOv2 token projection
|   |-- train_config.py        # config loader
|   |-- class_config.py        # class metadata loader
|   |-- dataloader_latest.py   # VOC-style dataset loader
|   |-- callbacks.py           # loss and mIoU callbacks
|   `-- utils_metrics.py       # segmentation metrics
|-- train.py                   # training entry point
|-- evaluate.py                # compact checkpoint evaluator
|-- predict.py                 # interactive/image/video inference
|-- get_miou.py                # legacy mIoU evaluation path
|-- voc_annotation.py          # split-file generation
`-- benchmark.py               # parameter/FLOP/FPS report
```

## Architecture

The active VCFS architecture is defined in `nets/deeplabv3_plus.py`:

- `MobileNetV2` extracts CNN features at four stages.
- A frozen DINOv2 ViT-S/14 model provides visual foundation model tokens.
- `DINOFeatureProjection` reshapes DINO tokens to each CNN feature scale and
  projects channels with a trainable 1x1 convolution, BatchNorm, and ReLU.
- `Attention_cross` uses CNN features as queries and projected DINO features as
  keys/values, then adds the fused representation back to the CNN stream.
- The DeepLabv3+ ASPP and decoder head produce dense facade parsing logits.

The current architecture marker is:

```python
VCFS_ARCHITECTURE_VERSION = "registered_dino_projection_v1"
```

## Class Setup

FacadeWHU experiments use this class order:

```text
background, window, door, facade, balcony, roof, shop
```

Class metadata lives in `../configs/classes.facadewhu.json`. The evaluator reads
`eval_ignore_class_ids` from the class config; the paper setting ignores
background when reporting paper mIoU while still recording all-class metrics.

## Dataset Format

Use a VOC-style dataset directory:

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

`train.py` selects the dataset and split names from `../configs/vcfs.*.json`,
then applies command-line or `VCFS_*` environment overrides. It refuses to train
when train and validation splits overlap.

## Training

From this directory:

```bash
python train.py --config ../configs/vcfs.facadewhu.json
```

Useful overrides:

```bash
python train.py \
  --config ../configs/vcfs.facadewhu.json \
  --dataset-path facadewhu_extend \
  --class-config ../configs/classes.facadewhu.json \
  --batch-size 4 \
  --fp16 true
```

## Evaluation

`evaluate.py` is the preferred evaluator because it writes a compact JSON report
without leaving per-image prediction masks:

```bash
python evaluate.py \
  --dataset-path facadewhu_extend \
  --checkpoint results/run_name/best_epoch_weights.pth \
  --class-config ../configs/classes.facadewhu.json \
  --output results/run_name/eval_best.json
```

`get_miou.py` remains available for the legacy workflow that saves prediction
masks and metric plots.

## Release Notes

Do not commit local datasets, generated images, result folders, TensorBoard
events, or experiment checkpoints. The repository keeps only the lightweight
baseline checkpoint required by the original DeepLabv3+ initialization path.
