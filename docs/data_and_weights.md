# Data and Weights

This repository intentionally excludes local experiment data and trained artifacts. Keep datasets, checkpoints, generated images, logs, and caches outside git.

## Expected Local Paths

Segmentation training code expects VOC-style data:

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

SDA scripts currently expect local data under:

```text
SDA/DINO_extract/FacadeWHU_origin/
SDA/DINO_extract/norway/
SDA/diffusion/<dataset folders>
```

The DINO and diffusion scripts also reference locally cached Hugging Face models and downloaded checkpoints. Keep those outside git or provide download instructions. Current local path conventions are summarized in `configs/paths.example.json`.

## Files to Exclude From Git

- `*.pth`, `*.pt`, `*.ckpt`, `*.bin`, `*.safetensors`
- raw images, generated synthetic images, visualization images, and masks
- `logs/`, `miou_out*/`, `debug/`, and paper visualization outputs
- dataset zips and other archives
- `__pycache__/`, `.ipynb_checkpoints/`, and IDE state

## Release Recommendation

For the public repository, publish one of these:

1. Download links for datasets and trained weights, with checksums.
2. A small sample dataset containing only a few permissively shareable examples.
3. Scripts that create required directory placeholders.

Do not publish datasets or pretrained weights unless their licenses and redistribution permissions are clear.

Before publishing repository updates, run the audit scripts from the repository root to check for accidental datasets, weights, generated media, caches, or local paths.
