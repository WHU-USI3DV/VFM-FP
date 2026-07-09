# SDA Code

SDA contains the Semantic Diffusion Augmentation workflow used before training the facade parser.

The public SDA code follows the paper design:

1. Diverse Domain Extension (DDE): generate style-diverse facade images with location, time, and weather prompts.
2. Long-Tail Preference (LTP): allocate more generated images to source images containing long-tail classes.
3. Semantic Consistency Filtering (SCF): use DINOv2 feature distance to keep semantically consistent generated samples.

## Main Scripts

- `diffusion/semantic_diffusion_augmentation.py`: main paper-aligned SDA generation entrypoint.
- `diffusion/Mul_Ab_norway.py`: compatibility wrapper for the original Norway defaults.
- `diffusion/voc_annotation.py`: split-file helper for generated outputs.
- `DINO_extract/semantic_consistency_filter.py`: SCF wrapper using DINOv2 mean-plus-std filtering.
- `DINO_extract/dino_rank_generated.py`: configurable DINOv2 feature-distance ranking and filtering CLI.
- `DINO_extract/Extract_Cul.py`, `Extract_Cul_facadewhu.py`, and `Extract_Cul_ecp.py`: compatibility ranking presets.

## 1. Generate Synthetic Images With DDE + LTP

From `SDA/diffusion`:

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

Use `--dry-run` to inspect the LTP allocation plan without loading diffusion models. `paper_high` uses the 16 DDE prompt combinations described in the paper:

```text
locations: France, USA, China, Italy
times: noon, afternoon
weathers: sunny, cloudy
```

Useful alternatives:

```bash
# Five-prompt limited-diversity ablation setting.
python semantic_diffusion_augmentation.py --prompt-profile paper_limited

# Original Norway compatibility setting.
python Mul_Ab_norway.py

# Custom domain prompts.
python semantic_diffusion_augmentation.py --locations France,China --times noon --weathers sunny,cloudy
```

## 2. Run SCF With DINOv2

The generation script writes aligned lists under `SDA_output/txt/`: `syn_trainval.txt` contains generated ids, and `source_trainval_for_syn.txt` repeats the corresponding source ids. Then run:

```bash
cd SDA/DINO_extract
python semantic_consistency_filter.py \
  --ori-jpeg-path FacadeWHU_origin/JPEGImages \
  --syn-jpeg-path SDA_output/syn_image \
  --ori-txt SDA_output/txt/source_trainval_for_syn.txt \
  --syn-txt SDA_output/txt/syn_trainval.txt \
  --save-path SDA_output/scf \
  --output-mode filtered_ids
```

SCF writes:

```text
SDA_output/scf/scf_keep.txt
SDA_output/scf/scf_discard.txt
```

The threshold is `mean(Q) + std(Q)` by default, matching the paper's semantic consistency filtering rule.

## 3. Prepare Retained Samples For VCFS

Copy the SCF-retained synthetic images into a VOC-style VCFS dataset and copy each source mask as the synthetic label:

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

This writes `txt/sda_retained.txt` and, with `--write-train-split`, merges `txt/train.txt` plus retained `syn_*` samples into `txt/train_1601.txt`. The VCFS trainer also auto-generates `train_1601.txt` when paired `syn_*` images and masks are already present.

Generated images, DINO outputs, low-quality sample folders, and local model caches are excluded by the root `.gitignore`.

