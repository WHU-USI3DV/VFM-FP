# VFM-FP

<a href="https://github.com/" target="_blank">Leveraging Visual Foundation Model Priors for Facade Parsing from Street View Images</a>

This repository contains the source code for VFM-FP, a street-view facade parsing workflow that combines data-level synthetic data augmentation with semantic segmentation training and inference.

> **Leveraging Visual Foundation Model Priors for Facade Parsing from Street View Images**  
> Zhe Chen; Qingwen Tan; Fuxun Liang; Chen Long; Pangyin Li; Zhiming Liu; Junjie Chen; Zhen Dong  
> Accepted manuscript. Official DOI and publication metadata are pending in `docs/citation_template.md`.

## Introduction

VFM-FP is designed for facade parsing from street-view images. The public code is organized around two parts:

- **SDA**: data-level expansion and sample selection using diffusion/ControlNet generation and DINO feature-distance ranking.
- **VCFS**: VFM-CNN fusion segmentor code for training, inference, and mIoU evaluation.

This release preserves the accepted-paper algorithmic behavior while keeping local datasets, checkpoints, generated images, and experiment outputs outside the repository.

## News

- 2026-07-09: Public source code released on GitHub.
- 2026-07-04: The segmentation module was renamed to `VCFS`, short for VFM-CNN fusion segmentor.
- 2026-07-04: Final PDF metadata was used to fill the citation draft title and author list.

## Repository Layout

```text
.
|-- SDA/                         # Data-level augmentation and sample selection
|   |-- diffusion/               # ControlNet / diffusion image generation
|   `-- DINO_extract/            # DINO feature extraction and generated-sample ranking
|-- VCFS/                        # VFM-CNN fusion segmentor
|   |-- nets/                    # Network definitions
|   |-- utils/                   # Dataloaders, callbacks, metrics, training helpers
|   |-- train.py                 # Training entrypoint
|   |-- predict.py               # Image/folder/video/FPS/ONNX inference entrypoint
|   `-- get_miou.py              # mIoU evaluation entrypoint
|-- configs/                     # Class metadata and path examples
|-- docs/                        # Data, citation, and licensing notes
|-- requirements/                # Lightweight dependency lists
`-- tools/                       # Release generation and audit tools
```

## Environment

The code was cleaned to keep dependency lists small and readable. Install the PyTorch build that matches your CUDA version.

For VCFS segmentation training and inference:

```bash
conda create -n vfmfp python=3.10
conda activate vfmfp
pip install -r requirements/segmentation.txt
```

For SDA generation and DINO ranking:

```bash
conda activate vfmfp
pip install -r requirements/sda.txt
```

Python 3.10 is the recommended public-release baseline. `VCFS` loads DINOv2 through `torch.hub`, and the current upstream DINOv2 hub code uses Python 3.10 syntax. Python 3.8 may still work only if you already have a compatible local Torch Hub cache snapshot.

The original upstream environment files under `VCFS/` are preserved only as experiment references. Prefer the curated files under `requirements/` for a public release.

## Data Preparation

Large datasets, model weights, generated images, logs, and caches are intentionally excluded from this repository.

The recommended workflow is:

1. Convert the original dataset to VOC-style folders.
2. Run SDA on the original training set to generate and filter augmented samples.
3. Build the augmented VCFS dataset and train VCFS on that augmented split.

All datasets must use single-channel label masks. Each pixel value in `SegmentationClass/*.png` must be a class id, not an RGB color:

```text
<dataset_root>/
|-- JPEGImages/
|   |-- xxx.jpg
|-- SegmentationClass/
|   |-- xxx.png
`-- txt/
    |-- train.txt
    |-- val.txt
    |-- trainval.txt
    `-- test.txt
```

### Class Configs

Class-dependent settings are stored in JSON files under `configs/`:

- `configs/classes.facadewhu.json`: default FacadeWHU-style 7-class setup.
- `configs/classes.ecp.json`: ECP native-class example.

`VCFS/train.py`, `VCFS/deeplab.py`, `VCFS/predict.py`, and `VCFS/get_miou.py` read `VCFS_CLASS_CONFIG`. If the variable is not set, the default is `configs/classes.facadewhu.json`.

PowerShell example:

```powershell
$env:VCFS_CLASS_CONFIG="configs/classes.facadewhu.json"
$env:VCFS_DATASET_PATH="facadewhu_extend"
```

Bash example:

```bash
export VCFS_CLASS_CONFIG=configs/classes.facadewhu.json
export VCFS_DATASET_PATH=facadewhu_extend
```

You may also override only the class count with `VCFS_NUM_CLASSES`, but the safer public workflow is to edit or create a matching class JSON file.

### Default FacadeWHU Setup

The public defaults are configured for the FacadeWHU-style 7-class facade parser:

```text
0 background
1 window
2 door
3 facade
4 balcony
5 roof
6 shop
```

Place the original data used by SDA at:

```text
FacadeWHU_origin/
|-- JPEGImages/
|-- SegmentationClass/
`-- txt/
    |-- train.txt
    |-- trainval.txt
    `-- val.txt
```

The default SDA output is `SDA_output/`. After SCF, run `SDA/prepare_vcfs_augmented_dataset.py` to create the augmented VCFS dataset under `VCFS/facadewhu_extend/`, including `txt/train_1601.txt`.

### ECP or Another Native-Class Dataset

ECP does not need to be remapped before training. Keep the original ECP class ids and train directly on native ECP labels. The example native ECP config is:

```text
configs/classes.ecp.json
background, wall, window, door, balcony, roof, shop, sky, chimney
```

For ECP training, set the ECP class config and dataset path before running VCFS:

```powershell
$env:VCFS_CLASS_CONFIG="configs/classes.ecp.json"
$env:VCFS_DATASET_PATH="ecp_0619_refine"
cd VCFS
python train.py
```

Only remap ECP masks if you deliberately want to merge ECP into the default FacadeWHU 7-class setup. In that case, convert masks into ids `0..6` following the FacadeWHU order before running SDA or VCFS, and use `configs/classes.facadewhu.json`.

For a new dataset, copy one of the class config files, update `num_classes`, `classes`, `colors_rgb`, and `dominant_class_id`, then set `VCFS_CLASS_CONFIG` to that file. `dominant_class_id` is used by SDA LTP; for FacadeWHU it is `3` because `facade` is class id 3.

### Where To Change Runtime Settings

Use these environment variables when possible, so you do not have to edit source files for every dataset:

```text
VCFS_CLASS_CONFIG      class metadata JSON used by training, inference, and mIoU
VCFS_DATASET_PATH      dataset folder under VCFS/ or an absolute dataset path
VCFS_MODEL_PATH        checkpoint path for training resume or inference defaults
VCFS_INPUT_SHAPE       input size, for example 512,512 or 512x512
VCFS_BATCH_SIZE        unfreeze-stage batch size
VCFS_EPOCHS            total training epochs
VCFS_SAVE_DIR          training log/checkpoint output folder
VCFS_NUM_WORKERS       dataloader worker count
VCFS_MIOU_OUT_PATH     mIoU prediction/output folder
```

The algorithmic defaults in `VCFS/train.py` remain the code baseline: `Init_lr=2e-4`, `Unfreeze_batch_size=4`, Adam, cosine decay, and 200 epochs.

Source-only release packages do not include datasets or weights. See `docs/data_and_weights.md` for expected paths and redistribution guidance.

## SDA: Data-Level Expansion

SDA is the first stage of the VFM-FP workflow. It expands the original facade parsing data, filters generated samples, and prepares the augmented VOC-style dataset used by VCFS training.

### 1. Diffusion / ControlNet generation with DDE + LTP

The main SDA generation entrypoint is:

```bash
python SDA/diffusion/semantic_diffusion_augmentation.py --help
```

Example command using the paper's high-diversity DDE setting, run from the repository root:

```bash
python SDA/diffusion/semantic_diffusion_augmentation.py \
  --class-config configs/classes.facadewhu.json \
  --allocation-mode ltp \
  --prompt-profile paper_high \
  --target-total 1601
```

By default, the script reads `FacadeWHU_origin/JPEGImages`, `FacadeWHU_origin/SegmentationClass`, and `FacadeWHU_origin/txt/trainval.txt`, then writes generated images and records under `SDA_output/`. Add `--dry-run` to inspect the LTP allocation plan without loading diffusion models. For ECP or another native-class dataset, pass the matching class config, for example `--class-config configs/classes.ecp.json`. `paper_high` combines the location, time, and weather prompts from the paper: France/USA/China/Italy, noon/afternoon, and sunny/cloudy. `paper_limited` keeps the five-prompt ablation setting, while `Mul_Ab_norway.py` remains as a compatibility wrapper for the original Norway defaults.

This script initializes Stable Diffusion, ControlNet, depth estimation, and semantic segmentation models at runtime. Keep Hugging Face caches and downloaded model weights outside git.

### 2. Semantic consistency filtering with DINOv2

The SCF entrypoint is:

```bash
python SDA/DINO_extract/semantic_consistency_filter.py --help
```

Example command, run from the repository root:

```bash
python SDA/DINO_extract/semantic_consistency_filter.py --output-mode filtered_ids
```

The generation step writes `SDA_output/txt/syn_trainval.txt` and `SDA_output/txt/source_trainval_for_syn.txt` as aligned synthetic/source id lists. SCF uses DINOv2 feature distance and keeps generated samples whose score is not above `mean + std`. The lower-level `dino_rank_generated.py` script also supports sorted-score outputs for manual inspection.

Compatibility presets are also kept:

```bash
python SDA/DINO_extract/Extract_Cul.py
python SDA/DINO_extract/Extract_Cul_facadewhu.py
python SDA/DINO_extract/Extract_Cul_ecp.py
```

### 3. Prepare SCF-retained samples for VCFS

Copy retained synthetic images and their inherited source masks into a VOC-style VCFS dataset. Add `--copy-source-train` when the VCFS dataset does not already contain the original training images and labels:

```bash
python SDA/prepare_vcfs_augmented_dataset.py \
  --copy-source-train \
  --write-train-split
```

By default, this reads `SDA_output/scf/scf_keep.txt` and `SDA_output/txt/synthetic_pairs.csv`, writes retained synthetic samples into `VCFS/facadewhu_extend`, and creates `txt/train_1601.txt` from `txt/train.txt` plus retained `syn_*` samples. The script validates that every split id has paired image and mask files before writing the augmented training split. `SDA/diffusion/voc_annotation.py` is still kept as a simple split-file helper when needed.

## VCFS: Train and Use the Facade Parsing Segmentor

### 1. Confirm augmented split files

After SDA, `VCFS/facadewhu_extend/txt/train_1601.txt` should contain the original training samples plus the SCF-retained `syn_*` samples. If your dataset already has `txt/train_1601.txt`, `txt/val.txt`, `txt/trainval.txt`, and related split files, you can skip this step.

Otherwise, edit `VOCdevkit_path` in `VCFS/voc_annotation.py` to point to your dataset folder, then run:

```bash
cd VCFS
python voc_annotation.py
```

### 2. Configure training

Prefer configuring dataset-specific settings with environment variables:

```powershell
$env:VCFS_CLASS_CONFIG="configs/classes.facadewhu.json"
$env:VCFS_DATASET_PATH="facadewhu_extend"
$env:VCFS_BATCH_SIZE="4"
$env:VCFS_EPOCHS="200"
```

For ECP native-class training:

```powershell
$env:VCFS_CLASS_CONFIG="configs/classes.ecp.json"
$env:VCFS_DATASET_PATH="ecp_0619_refine"
```

The main source locations are still easy to find when you want to change defaults permanently:

- `configs/classes.*.json`: class count, class names, colors, and SDA `dominant_class_id`.
- `VCFS/train.py`: learning rate, optimizer, scheduler, epoch, batch-size, and dataset defaults.
- `VCFS/deeplab.py`: inference checkpoint defaults such as `model_path`, `backbone`, and `input_shape`.
- `VCFS/predict.py`: prediction mode, input folder, and output folder.
- `VCFS/get_miou.py`: mIoU mode and output folder; `VCFS_DATASET_PATH` and `VCFS_MIOU_OUT_PATH` can override paths.

These defaults are kept close to the accepted code for reproducibility.

### 3. Train

```bash
cd VCFS
python train.py
```

Training outputs are written under `VCFS/logs/` by default. Logs and checkpoints are ignored by git and excluded from release packages.

### 4. Predict

Edit `mode` and related paths in `VCFS/predict.py`, then run:

```bash
cd VCFS
python predict.py
```

Useful modes inside `predict.py` include:

- `predict`: single-image interactive prediction.
- `dir_predict`: folder prediction.
- `video`: video or camera prediction.
- `fps`: FPS measurement.
- `export_onnx`: ONNX export.

For folder prediction, edit:

```python
mode = "dir_predict"
dir_origin_path = "path/to/JPEGImages"
dir_save_path = "path/to/output"
```

### 5. Evaluate mIoU

Edit the dataset and output settings in `VCFS/get_miou.py`:

- `VOCdevkit_path`
- `image_ids`
- `gt_dir`
- `miou_out_path`
- `num_classes`
- `name_classes`

Then run:

```bash
cd VCFS
python get_miou.py
```

### 6. Model complexity and FPS utility

The optional benchmark script is kept in the final package:

```bash
cd VCFS
python benchmark.py
```

Use it when you need parameters, FLOPs, GPU memory, or FPS statistics.

## Repository Checks

Run these checks before publishing changes:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/smoke_check.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File tools/audit_release.ps1 -Root . -Limit 100
powershell -NoProfile -ExecutionPolicy Bypass -File tools/content_audit.ps1 -Root . -Limit 100
python tools/syntax_check_release.py .
```

The latest checked public code passed:

- release asset audit
- content audit
- Python syntax check
- static local-import check

## Citation

The citation draft is stored in `docs/citation_template.md`. It already contains the paper title and author list extracted from the final PDF metadata.

Do not add a root `CITATION.cff` until official year, journal issue details, pages, and DOI are available.

## License

This repository does not currently include a root project `LICENSE`. The upstream DeepLabv3+ baseline notice is preserved at `VCFS/LICENSE`.

See `docs/licensing.md` for details.

## Related Projects

This repository builds on or interfaces with the following open-source ecosystems:

- DeepLabv3+ PyTorch baseline by Bubbliiiing.
- DINOv2 from Facebook Research.
- Hugging Face diffusers and transformers.
- ControlNet models used through diffusers.
