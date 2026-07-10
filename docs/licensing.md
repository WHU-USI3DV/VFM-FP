# Licensing Notes

This repository combines original VFM-FP code with code derived from an upstream DeepLabv3+ PyTorch implementation.

## Current State

- `VCFS/LICENSE` contains the upstream MIT license notice from Bubbliiiing.
- This repository does not currently include a root project `LICENSE`.
- Dataset files, generated images, additional pretrained weights, and model caches should not be redistributed until their licenses are confirmed. The default `VCFS/model_data/deeplab_mobilenetv2.pth` checkpoint is intentionally included in this release.

## Before Public Release

- Revisit whether a root `LICENSE` should be added before public GitHub release.
- Preserve the upstream MIT notice for the DeepLabv3+ baseline code.
- Add third-party attribution for any copied or modified model code.
- Confirm redistribution rights for all datasets, sample images, additional trained weights, and generated assets.
- Keep private datasets and additional checkpoints outside git if redistribution is unclear.

## Practical Recommendation

For the public release, keep datasets and generated assets outside git. Provide external download instructions for datasets and any additional weights after verifying redistribution rights.

Publishing without a root license means reuse rights are not explicitly granted for the project-owned code. Add a root license later if broader reuse permissions should be granted.

