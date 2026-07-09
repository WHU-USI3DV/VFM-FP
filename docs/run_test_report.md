# Run Test Report

Generated for the current source-only release candidate.

## Scope

- Release candidate: `release/VFM-FP-final`
- Test type: lightweight structural and entrypoint checks
- Heavy training, inference, and diffusion generation were not run because datasets, model weights, and several runtime dependencies are not installed in the current environment.

## Passed Checks

- Smoke check: passed
- Release asset audit: 0 findings
- Content audit: 0 findings
- Python syntax check: 34 Python files scanned, 0 syntax failures
- `SDA/DINO_extract/dino_rank_generated.py --help`: passed
- `SDA/diffusion/Mul_Ab_norway.py --help`: passed

## Current Python Environment

- Python: 3.8.8

Available packages detected:

- torchvision
- cv2
- PIL
- numpy
- scipy
- tqdm
- matplotlib
- h5py
- sklearn

Missing packages detected:

- torch
- tensorboard
- labelme
- thop
- torchsummary
- onnx
- onnxsim
- diffusers
- transformers
- accelerate
- safetensors

## Interpretation

The final release package is structurally runnable: scripts parse correctly, release checks pass, and lightweight CLI entrypoints start successfully.

Full VCFS training/inference requires installing `requirements/segmentation.txt`, preparing VOC-style data, and providing trained or pretrained weights.

Full SDA generation/ranking requires installing `requirements/sda.txt`, configuring local data paths, and providing model access/cache for DINOv2, Stable Diffusion, ControlNet, and related Hugging Face models.
