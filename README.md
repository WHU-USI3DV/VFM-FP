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

VCFS expects VOC-style facade parsing data:

```text
VCFS/<dataset_name>/
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

The facade class order used by the cleaned metadata is:

```text
background, window, door, facade, balcony, roof, shop
```

The same class metadata is recorded in `configs/facade_classes.json`.

Source-only release packages do not include datasets or weights. See `docs/data_and_weights.md` for expected paths and redistribution guidance.

## VCFS: Facade Parsing Segmentor

### 1. Generate VOC split files

If your dataset already has `txt/train.txt`, `txt/val.txt`, `txt/trainval.txt`, and `txt/test.txt`, you can skip this step.

Otherwise, edit `VOCdevkit_path` in `VCFS/voc_annotation.py` to point to your dataset folder, then run:

```bash
cd VCFS
python voc_annotation.py
```

### 2. Configure training

Before training, check the following settings in `VCFS/train.py`:

- `VOCdevkit_path`: dataset folder.
- Training split path: currently `txt/train_1601.txt`.
- Validation split path: currently `txt/val.txt`.
- `num_classes`: current facade setup uses 7 classes.
- `model_path`: pretrained or resumed checkpoint path.
- `input_shape`, `backbone`, `downsample_factor`, batch size, optimizer, and epoch settings.

These defaults are kept unchanged for reproducibility.

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

## SDA: Data-Level Expansion

SDA contains the data expansion and generated-sample ranking workflow used before segmentation training.

### 1. Diffusion / ControlNet generation with DDE + LTP

The main SDA generation entrypoint is:

```bash
cd SDA/diffusion
python semantic_diffusion_augmentation.py --help
```

Example command using the paper's high-diversity DDE setting:

```bash
python semantic_diffusion_augmentation.py \
  --image-dir FacadeWHU_origin/JPEGImages \
  --mask-dir FacadeWHU_origin/SegmentationClass \
  --split-file FacadeWHU_origin/txt/trainval.txt \
  --output-dir SDA_output/syn_image \
  --allocation-mode ltp \
  --prompt-profile paper_high \
  --target-total 1601
```

Add `--dry-run` to inspect the LTP allocation plan without loading diffusion models. `paper_high` combines the location, time, and weather prompts from the paper: France/USA/China/Italy, noon/afternoon, and sunny/cloudy. `paper_limited` keeps the five-prompt ablation setting, while `Mul_Ab_norway.py` remains as a compatibility wrapper for the original Norway defaults.

This script initializes Stable Diffusion, ControlNet, depth estimation, and semantic segmentation models at runtime. Keep Hugging Face caches and downloaded model weights outside git.

### 2. Semantic consistency filtering with DINOv2

The SCF entrypoint is:

```bash
cd SDA/DINO_extract
python semantic_consistency_filter.py --help
```

Example command:

```bash
python semantic_consistency_filter.py \
  --ori-jpeg-path FacadeWHU_origin/JPEGImages \
  --syn-jpeg-path SDA_output/syn_image \
  --ori-txt SDA_output/txt/source_trainval_for_syn.txt \
  --syn-txt SDA_output/txt/syn_trainval.txt \
  --save-path SDA_output/scf \
  --output-mode filtered_ids
```

The generation step writes `SDA_output/txt/syn_trainval.txt` and `SDA_output/txt/source_trainval_for_syn.txt` as aligned synthetic/source id lists. SCF uses DINOv2 feature distance and keeps generated samples whose score is not above `mean + std`. The lower-level `dino_rank_generated.py` script also supports sorted-score outputs for manual inspection.

Compatibility presets are also kept:

```bash
python Extract_Cul.py
python Extract_Cul_facadewhu.py
python Extract_Cul_ecp.py
```

### 3. Prepare SCF-retained samples for VCFS

Copy retained synthetic images and their inherited source masks into a VOC-style VCFS dataset:

```bash
cd SDA
python prepare_vcfs_augmented_dataset.py \
  --synthetic-image-dir SDA_output/syn_image \
  --source-mask-dir FacadeWHU_origin/SegmentationClass \
  --scf-keep SDA_output/scf/scf_keep.txt \
  --pair-record SDA_output/txt/synthetic_pairs.csv \
  --target-dataset ../VCFS/facadewhu_extend \
  --write-train-split
```

This writes `txt/sda_retained.txt` and, with `--write-train-split`, creates `txt/train_1601.txt` from `txt/train.txt` plus retained `syn_*` samples. `SDA/diffusion/voc_annotation.py` is still kept as a simple split-file helper when needed.

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
