# Data and Weights

This repository intentionally excludes local experiment data and generated training artifacts. Keep datasets, generated images, logs, and caches outside git.

The public release includes `VCFS/model_data/deeplab_mobilenetv2.pth`, the default MobileNet DeepLab checkpoint used by the VCFS training and inference entrypoints.

## Expected Local Paths

Segmentation training code expects VOC-style data with id-mask labels:

```text
VCFS/<dataset_name>/
|-- JPEGImages/
|-- SegmentationClass/
`-- txt/
    |-- train.txt
    |-- val.txt
    |-- trainval.txt
    `-- test.txt
```

SDA scripts use the original dataset and write generated outputs under these default paths:

```text
FacadeWHU_origin/          # original images, masks, and split files for SDA
SDA_output/                # generated images, SCF records, and retained ids
VCFS/facadewhu_extend/     # augmented VOC-style dataset prepared for VCFS
```

## Dataset Class Configs

Class metadata is configured through JSON files under `configs/`:

- `configs/classes.facadewhu.json`: default FacadeWHU-style 7-class setup.
- `configs/classes.ecp.json`: example native ECP setup.

The default FacadeWHU class order is:

```text
0 background
1 window
2 door
3 facade
4 balcony
5 roof
6 shop
```

ECP training should use the original ECP class ids directly; remapping is only needed if you intentionally convert ECP into the default FacadeWHU 7-class setup.

For VCFS, set:

```bash
export VCFS_CLASS_CONFIG=configs/classes.ecp.json
export VCFS_DATASET_PATH=ecp_0619_refine
```

For SDA LTP on ECP or a custom dataset, pass the same class config:

```bash
python SDA/diffusion/semantic_diffusion_augmentation.py --class-config configs/classes.ecp.json
```

For a custom dataset, copy one of the class JSON files and update `num_classes`, `classes`, `colors_rgb`, and `dominant_class_id`. The `dominant_class_id` controls SDA long-tail preference allocation.

Current local path conventions are summarized in `configs/paths.example.json`.

## Files to Exclude From Git

- `*.pth`, `*.pt`, `*.ckpt`, `*.bin`, `*.safetensors`, except the included `VCFS/model_data/deeplab_mobilenetv2.pth`
- raw images, generated synthetic images, visualization images, and masks
- `logs/`, `miou_out*/`, `debug/`, and paper visualization outputs
- dataset zips and other archives
- `__pycache__/`, `.ipynb_checkpoints/`, and IDE state

## Release Recommendation

For the public repository, publish one of these:

1. Download links for datasets and additional trained weights, with checksums.
2. A small sample dataset containing only a few permissively shareable examples.
3. Scripts that create required directory placeholders.

Do not publish datasets or additional pretrained weights unless their licenses and redistribution permissions are clear.

Before publishing repository updates, run the audit scripts from the repository root to check for accidental datasets, weights, generated media, caches, or local paths.

