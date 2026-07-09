# Licensing Notes

This repository combines original VFM-FP code with code derived from an upstream DeepLabv3+ PyTorch implementation.

## Current State

- `VCFS/LICENSE` contains the upstream MIT license notice from Bubbliiiing.
- This repository does not currently include a root project `LICENSE`.
- Dataset files, generated images, pretrained weights, and model caches should not be redistributed until their licenses are confirmed.

## Before Public Release

- Revisit whether a root `LICENSE` should be added before public GitHub release.
- Preserve the upstream MIT notice for the DeepLabv3+ baseline code.
- Add third-party attribution for any copied or modified model code.
- Confirm redistribution rights for all datasets, sample images, trained weights, and generated assets.
- Keep private datasets and checkpoints outside git if redistribution is unclear.

## Practical Recommendation

For the first public release, publish source code and documentation only. Provide external download instructions for datasets and weights after verifying redistribution rights.

Publishing without a root license means reuse rights are not explicitly granted for the project-owned code. Add a root license later if broader reuse permissions should be granted.
