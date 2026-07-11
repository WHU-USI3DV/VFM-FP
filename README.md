# VFM-FP

<a href="https://doi.org/10.1016/j.inffus.2026.104593" target="_blank">Leveraging Visual Foundation Model Priors for Facade Parsing from Street View Images</a>

This is the PyTorch implementation about VFM-FP, a street-view facade parsing workflow that leverages prior knowledge of VFMs for data augmentation and feature fusion.

> **Leveraging Visual Foundation Model Priors for Facade Parsing from Street View Images**  
> [Zhe Chen](https://chenzhe-code.github.io/); [Qingwen Tan](https://github.com/TanQingw); [Fuxun Liang](https://liangfxwhu.github.io/); [Chen Long](https://chenlongwhu.github.io/); [Pangyin Li](https://dongzhenwhu.github.io/team/index.html); [Zhiming Liu](https://liesmars.whu.edu.cn/yjry/gdyjry.htm); [Junjie Chen](https://www.arch.hku.hk/staff/rec/chen-junjie/); [Zhen Dong](https://dongzhenwhu.github.io/)  

## 🔭 Introduction

VFM-FP is designed for facade parsing from street-view images. The public code is organized around two parts:

- **SDA**: data-level expansion and sample selection using diffusion/ControlNet generation and DINO feature-distance ranking.
- **VCFS**: VFM-CNN fusion segmentor code for training, inference, and mIoU evaluation.

Local datasets, generated images, and experiment outputs are kept outside the repository. A default MobileNet DeepLab checkpoint is included for training and inference.

## 🆕 Repository Layout

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

## 🔧 Environment

Recommended tested environment:

- Python 3.10
- PyTorch 2.3.0
- torchvision 0.18.0
- CUDA 11.8

Create the environment and install the recommended PyTorch build:

```bash
conda create -n vfmfp python=3.10
conda activate vfmfp
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118
```

For VCFS segmentation training and inference:

```bash
pip install -r requirements/segmentation.txt
```

For SDA generation and DINO ranking:

```bash
pip install -r requirements/sda.txt
```

If your CUDA version is different, install the matching PyTorch and torchvision build first, then install the remaining dependencies from `requirements/`.

Python 3.10 is the recommended public-release baseline. `VCFS` loads DINOv2 through `torch.hub`, and the current upstream DINOv2 hub code uses Python 3.10 syntax. Python 3.8 may still work only if you already have a compatible local Torch Hub cache snapshot.

Use the dependency files under `requirements/` for installation.

## 💾 Data Preparation
Large datasets, generated images, logs, and caches are intentionally excluded from this repository. The default `VCFS/model_data/deeplab_mobilenetv2.pth` checkpoint is included for VCFS training and inference.

The recommended workflow is:

1. Organize the dataset into the VOC-style folder structure required by VCFS.
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

### Dataset Configs

Dataset-dependent settings are stored in JSON files under `configs/`:

- `configs/classes.facadewhu.json`: FacadeWHU 7-class setup.
- `configs/classes.ecp.json`: ECP 9-class setup.

`VCFS/train.py` can read the dataset and class config directly from command-line arguments. If no class config is provided, the default is `configs/classes.facadewhu.json`.

Training example:

```bash
cd VCFS
python train.py --class-config ../configs/classes.facadewhu.json --dataset-path facadewhu_extend
```

You may also pass `--num-classes`, but the safer workflow is to edit or create a matching class JSON file.

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

For ECP training, pass the ECP class config and dataset path in the same command:

```bash
cd VCFS
python train.py --class-config ../configs/classes.ecp.json --dataset-path ecp_0619_refine
```

Only remap ECP masks if you deliberately want to merge ECP into the default FacadeWHU 7-class setup. In that case, convert masks into ids `0..6` following the FacadeWHU order before running SDA or VCFS, and use `configs/classes.facadewhu.json`.

For a new dataset, copy one of the class config files, update `num_classes`, `classes`, `colors_rgb`, and `dominant_class_id`, then pass that file with `--class-config`. `dominant_class_id` is used by SDA LTP; for FacadeWHU it is `3` because `facade` is class id 3.

### Where To Change Runtime Settings

Use command-line arguments for training, so you do not have to edit source files for every dataset:

```text
--class-config         class metadata JSON
--dataset-path         dataset folder under VCFS/ or an absolute dataset path
--model-path           checkpoint path for training resume
--input-shape          input size, for example 512,512 or 512x512
--batch-size           unfreeze-stage batch size
--epochs               total training epochs
--save-dir             training log/checkpoint output folder
--num-workers          DataLoader worker count
```

The algorithmic defaults in `VCFS/train.py` remain the code baseline: `Init_lr=2e-4`, `Unfreeze_batch_size=4`, Adam, cosine decay, and 200 epochs.

Release packages do not include datasets or generated training outputs. See `docs/data_and_weights.md` for expected paths and redistribution guidance.

## 🔦 SDA: Data-Level Expansion

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

## ✏️ VCFS: Train and Use the Facade Parsing Segmentor

### 1. Confirm augmented split files

After SDA, `VCFS/facadewhu_extend/txt/train_1601.txt` should contain the original training samples plus the SCF-retained `syn_*` samples. If your dataset already has `txt/train_1601.txt`, `txt/val.txt`, `txt/trainval.txt`, and related split files, you can skip this step.

Otherwise, run the split helper from `VCFS/`:

```bash
cd VCFS
python voc_annotation.py
```

### 2. Configure training

Pass dataset-specific settings directly to `train.py`:

```bash
python train.py --class-config ../configs/classes.facadewhu.json --dataset-path facadewhu_extend --batch-size 4 --epochs 200
```

For ECP native-class training:

```bash
python train.py --class-config ../configs/classes.ecp.json --dataset-path ecp_0619_refine --batch-size 4 --epochs 200
```

The main source locations are still easy to find when you want to change defaults permanently:

- `configs/classes.*.json`: class count, class names, colors, and SDA `dominant_class_id`.
- `VCFS/train.py`: learning rate, optimizer, scheduler, epoch, batch-size, and dataset defaults.
- `VCFS/deeplab.py`: inference checkpoint defaults such as `model_path`, `backbone`, and `input_shape`.
- `VCFS/predict.py`: prediction mode, input folder, and output folder.
- `VCFS/get_miou.py`: mIoU mode, dataset path, class config, and output folder.

These defaults are kept close to the accepted code for reproducibility.

### 3. Train

```bash
cd VCFS
python train.py --class-config ../configs/classes.facadewhu.json --dataset-path facadewhu_extend
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

Pass the dataset, class config, and output settings directly:

```bash
cd VCFS
python get_miou.py --class-config ../configs/classes.facadewhu.json --dataset-path facadewhu_extend --miou-out-path miou_out
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

```bash
python tools/smoke_check_release.py .
python tools/audit_release.py --root . --limit 100
python tools/syntax_check_release.py .
```

The latest checked public code passed:

- release asset audit
- content audit
- Python syntax check
- static local-import check

## 💡 Citation

If you use this code, please cite:

```bibtex
@article{chen2026vfmfp,
  title = {Leveraging Visual Foundation Model Priors for Facade Parsing from Street View Images},
  author = {Chen, Zhe and Tan, Qingwen and Liang, Fuxun and Long, Chen and Li, Pangyin and Liu, Zhiming and Chen, Junjie and Dong, Zhen},
  journal = {Information Fusion},
  year = {2026},
  doi = {10.1016/j.inffus.2026.104593}
}
```

## License

This repository does not currently include a root project `LICENSE`. The upstream DeepLabv3+ baseline notice is preserved at `VCFS/LICENSE`.

See `docs/licensing.md` for details.

## Related Projects

This repository builds on or interfaces with the following open-source ecosystems:

- DeepLabv3+ PyTorch baseline by Bubbliiiing.
- DINOv2 from Facebook Research.
- Hugging Face diffusers and transformers.
- ControlNet models used through diffusers.
