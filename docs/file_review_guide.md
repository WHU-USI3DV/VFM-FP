# File Review Guide

This guide summarizes the six files that were reviewed and are kept in the current final lean release decision.

The full file-by-file inventory is generated at:

    release/VFM-FP-essential/FILE_INVENTORY.md

## Current Release Candidate

- `release/VFM-FP-essential` is the source-only candidate for manual review.
- The folder excludes datasets, generated images, model weights, caches, archives, and local run outputs.
- The latest inventory lists 53 `Keep`, 6 `Review`, and 84 `Optional` files.

## Review Items

| File | Evidence | Recommendation |
| --- | --- | --- |
| `VCFS/.gitignore` | Duplicates useful ignore rules inside the segmentation subfolder. The root `.gitignore` already covers the public repository. | Optional if the whole repo is published; keep if the segmentation folder may be released independently. |
| `VCFS/benchmark.py` | Standalone FLOPs/parameter/FPS utility. It is not imported by training, inference, or evaluation. | Optional for code release; keep if reporting or reproducing model-complexity numbers is useful. |
| `VCFS/README.md` | Upstream DeepLabv3+ README retained as attribution/context. The VFM-FP-specific entrypoint is `README_VFMFP.md` plus the root README. | Optional for a lean release; keep if you want upstream usage notes visible. |
| `VCFS/requirements.txt` | Original baseline dependency list. The curated public dependency list is `requirements/segmentation.txt`. | Optional; keep only as an upstream experiment reference. |
| `VCFS/utils/dataloader_latest_ecp.py` | ECP-specific dataloader variant. `train.py` currently imports `utils.dataloader_latest`, not this file. | Optional unless ECP-specific experiments need to be reproduced. |
| `SDA/diffusion/voc_annotation.py` | Standalone split-generation helper for `diversity/SegmentationClass` to `diversity/txt`. It is not imported by the main SDA generation script. | Optional unless you want to publish this helper for SDA output splits. |

## Final Decision

All six Review files are kept in the current final lean package. Optional files remain excluded from the lean/final release but preserved in the broader source-only copies.
