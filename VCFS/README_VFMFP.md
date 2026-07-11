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

The same metadata is recorded in `../configs/classes.facadewhu.json`. ECP/native-class examples are in `../configs/classes.ecp.json`.

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

`train.py` reads the dataset selected by `--dataset-path`; if unset, it uses `facadewhu_extend`. It looks for:

```text
<dataset>/txt/train_1601.txt
<dataset>/txt/val.txt
```

If `train_1601.txt` is missing but `train.txt` exists, the split loader falls back to `train.txt`. Use `--class-config` to select the matching class metadata before training.

Training example:

```bash
python train.py --class-config ../configs/classes.ecp.json --dataset-path ecp_0619_refine
```

## Release Notes

The inherited `README.md` comes from the upstream DeepLabv3+ baseline. This VFM-FP-specific README is the preferred public entry for this folder.

Do not commit local `logs/`, extra `.pth` checkpoints, mIoU outputs, VOC data folders, or generated images. The included `model_data/deeplab_mobilenetv2.pth` is the only checkpoint kept in git.
