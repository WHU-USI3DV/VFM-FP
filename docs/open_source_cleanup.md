# Open Source Cleanup Notes

## Current Evidence

- The workspace is not currently a git repository.
- Code is mixed with datasets, model checkpoints, generated images, mIoU outputs, cache files, and paper visualization assets.
- There are about 88 Python files but more than 50,000 image files in the workspace.
- The main segmentation code lives in VCFS/.
- The main data augmentation and DINO ranking code lives in SDA/.
- The existing DeepLab README is inherited from the upstream baseline and displays with encoding issues in this environment.

## Preserved Algorithmic Boundaries

This cleanup pass does not modify:

- DeepLabv3+ network definitions in VCFS/nets/
- training loop logic in VCFS/utils/utils_fit.py
- metrics in VCFS/utils/utils_metrics.py
- DINO feature ranking formulas
- DINO ranking default paths and output file names in compatibility entrypoints
- ControlNet prompt, condition, and generation parameters
- inference resize and post-processing behavior

## Recommended Public Structure

VFM-FP/
+-- README.md
+-- configs/
+-- docs/
+-- SDA/
|   +-- README.md
|   +-- diffusion/
|   |   +-- legacy/            # diffusion data-maintenance helpers
|   +-- DINO_extract/
|       +-- legacy/            # DINO ranking and maintenance helpers
+-- VCFS/
    +-- README_VFMFP.md
    +-- train.py
    +-- predict.py
    +-- get_miou.py
    +-- legacy/                # one-off experiment helpers with clearer names
    +-- nets/
    +-- utils/
        +-- legacy/            # legacy dataset maintenance helpers

## Code Files Worth Keeping as Public Entrypoints

- VCFS/train.py
- VCFS/predict.py
- VCFS/get_miou.py
- VCFS/voc_annotation.py
- SDA/diffusion/Mul_Ab_norway.py
- SDA/DINO_extract/dino_rank_generated.py
- SDA/DINO_extract/Extract_Cul.py
- SDA/DINO_extract/Extract_Cul_facadewhu.py
- SDA/DINO_extract/Extract_Cul_ecp.py

## Files and Folders to Remove or Keep Local Before Publishing

- __pycache__/
- .ipynb_checkpoints/
- .vscode/
- logs/
- debug/
- miou_out*/
- deeplab_features/
- DINO_feature/
- pca_visualizations/
- raw datasets and dataset zips
- local model weights and Hugging Face model caches

## Completed Low-Risk Refactors

- Renamed dataloader_lastest.py to dataloader_latest.py and kept dataloader_lastest.py as a compatibility wrapper.
- Renamed dataloader_lastest_ecp.py to dataloader_latest_ecp.py and kept dataloader_lastest_ecp.py as a compatibility wrapper.
- Updated VCFS/train.py to import utils.dataloader_latest.
- Renamed benchimark.py to benchmark.py and kept benchimark.py as a compatibility wrapper.
- Moved root-level one-off helpers into VCFS/legacy/ with descriptive names and kept summary.py, turn.py, delete.py, and unsort.py as wrappers.
- Moved utility maintenance helpers into VCFS/utils/legacy/ with descriptive names and kept utils/delete.py, utils/downto.py, utils/Rename.py, and utils/turn1.py as wrappers.
- Moved legacy segmentation modules deeplab_ori.py, get_miou_ori.py, utils/dataloader.py, utils/dataloader_ori.py, utils/callbacks_ori.py, and utils/utils_ori.py into legacy folders with compatibility wrappers.
- Moved SDA diffusion one-off helpers into SDA/diffusion/legacy/ with descriptive names and kept the old script names as wrappers.
- Moved SDA DINO maintenance helpers into SDA/DINO_extract/legacy/ with descriptive names and kept the old script names as wrappers.
- Added small command-line interfaces to the moved maintenance helper scripts while preserving their original default paths and behavior.
- Extracted the shared DINOv2 generated-image ranking logic to SDA/DINO_extract/dino_rank_generated.py and kept Extract_Cul.py, Extract_Cul_facadewhu.py, and Extract_Cul_ecp.py as compatibility entrypoints with the original defaults.
- Parameterized SDA/diffusion/Mul_Ab_norway.py and delayed heavy diffusion/model imports until generation runtime; the original script is preserved at SDA/diffusion/legacy/mul_ab_norway_original.py.
- Moved earlier diffusion variants Mul_Ab_2.py and Mul_Ab_2_diversity.py into SDA/diffusion/legacy/ with compatibility wrappers.
- Moved DINO/DeepLab visualization helpers into SDA/DINO_extract/legacy/ with compatibility wrappers.

## Next Refactor Candidates

These are safe only after a behavior-preserving check:

- Replace hard-coded dataset paths with command-line arguments or config files in the remaining diffusion SDA entrypoints.
- Add smoke tests for diffusion public entrypoint argument parsing after parameterization.
- Convert the project-level README to include the final paper title, citation, and model download links.
- Use tools/prepare_release.ps1 to create a clean release copy, then run tools/audit_release.ps1 and tools/content_audit.ps1 against release/VFM-FP-open-source before initializing the public repository.






