# VFM-FP Segmentation Code

This folder contains the DeepLabv3+ training, evaluation, and inference code used by VFM-FP.

## Entrypoints

- `train.py`: train the facade parsing model.
- `predict.py`: run single-image, folder, video, FPS, or ONNX export modes.
- `get_miou.py`: evaluate mIoU on a VOC-style dataset.
- `voc_annotation.py`: generate VOC-style split files.

## Class Setup

The current facade class order is:

```text
background, window, door, facade, balcony, roof, shop
```

The same metadata is also recorded in `../configs/facade_classes.json`.

## Dataset Format

Use a VOC-style directory:

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

`train.py` currently reads:

```text
facadewhu_extend/txt/train_1601.txt
facadewhu_extend/txt/val.txt
```

Keep this behavior for reproducibility. A later refactor can expose these paths as command-line arguments while preserving the default values.

## Release Notes

The inherited `README.md` comes from the upstream DeepLabv3+ baseline. This VFM-FP-specific README is the preferred public entry for this folder.

Do not commit local `logs/`, `.pth` checkpoints, mIoU outputs, VOC data folders, or generated images.