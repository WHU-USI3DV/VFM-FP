# File Inventory

Generated for: VFM-FP-final
Generated at: 2026-07-04T16:28:54

Recommendation meanings:

- Keep: necessary for the public source release.
- Review: useful but worth checking before final publication.
- Optional: compatibility, legacy, or historical helper; remove only after confirming it is not needed.

| File | Category | Purpose | Recommendation |
| --- | --- | --- | --- |
| .gitattributes | Repo hygiene | Line-ending and text normalization rules. | Keep |
| .gitignore | Repo hygiene | Git ignore rules for data, weights, logs, caches, and release folders. | Keep |
| configs/facade_classes.json | Config | Facade class names, ids, and colors used by documentation and release metadata. | Keep |
| configs/paths.example.json | Config | Example path conventions and accepted default locations. | Keep |
| docs/citation_template.md | Documentation | Release, data, licensing, citation, cleanup, or file-role documentation. | Keep |
| docs/data_and_weights.md | Documentation | Release, data, licensing, citation, cleanup, or file-role documentation. | Keep |
| docs/file_review_guide.md | Documentation | Release, data, licensing, citation, cleanup, or file-role documentation. | Keep |
| docs/licensing.md | Documentation | Release, data, licensing, citation, cleanup, or file-role documentation. | Keep |
| docs/open_source_cleanup.md | Documentation | Release, data, licensing, citation, cleanup, or file-role documentation. | Keep |
| docs/owner_decisions.md | Documentation | Release, data, licensing, citation, cleanup, or file-role documentation. | Keep |
| docs/release_checklist.md | Documentation | Release, data, licensing, citation, cleanup, or file-role documentation. | Keep |
| docs/release_status.md | Documentation | Release, data, licensing, citation, cleanup, or file-role documentation. | Keep |
| docs/run_test_report.md | Documentation | Release, data, licensing, citation, cleanup, or file-role documentation. | Keep |
| docs/script_inventory.md | Documentation | Release, data, licensing, citation, cleanup, or file-role documentation. | Keep |
| LEAN_RELEASE_NOTES.md | Release note | Generated note describing how the lean release candidate was created. | Keep |
| README.md | Core | Project overview and entrypoint documentation. | Keep |
| RELEASE_COPY_NOTES.md | Release note | Generated note describing how the clean release copy was created. | Keep |
| requirements/sda.txt | Environment | Lightweight dependency list for segmentation or SDA workflows. | Keep |
| requirements/segmentation.txt | Environment | Lightweight dependency list for segmentation or SDA workflows. | Keep |
| SDA/diffusion/ImageNet_color.py | SDA core | Semantic color palette used by diffusion segmentation conditioning. | Keep |
| SDA/diffusion/Mul_Ab_norway.py | SDA core | Configurable ControlNet generation script with accepted-paper defaults. | Keep |
| SDA/diffusion/voc_annotation.py | SDA utility | VOC-style split generation helper for SDA outputs. | Review |
| SDA/DINO_extract/dino_rank_generated.py | SDA core | Configurable DINOv2 feature-distance ranking CLI. | Keep |
| SDA/DINO_extract/Extract_Cul.py | SDA core | Compatibility entrypoint for a DINO ranking preset. | Keep |
| SDA/DINO_extract/Extract_Cul_ecp.py | SDA core | Compatibility entrypoint for a DINO ranking preset. | Keep |
| SDA/DINO_extract/Extract_Cul_facadewhu.py | SDA core | Compatibility entrypoint for a DINO ranking preset. | Keep |
| SDA/DINO_extract/Extract_Cul2.py | SDA core | Compatibility entrypoint for a DINO ranking preset. | Keep |
| SDA/README.md | Documentation | Overview of SDA data expansion and DINO ranking scripts. | Keep |
| tools/audit_release.ps1 | Release tool | Checks the release folder for data, weights, media, archives, and large files. | Keep |
| tools/audit_release.py | Release tool | Checks the release folder for data, weights, media, archives, and large files. | Keep |
| tools/content_audit.ps1 | Release tool | Checks the release folder for local paths, placeholder markers, and sensitive text. | Keep |
| tools/generate_file_inventory.ps1 | Release tool | Generates this file-by-file inventory. | Keep |
| tools/prepare_lean_release.ps1 | Release tool | Creates a proposed lean release from Keep and Review inventory items. | Keep |
| tools/prepare_release.ps1 | Release tool | Creates the source-only clean release folder. | Keep |
| tools/smoke_check.ps1 | Release tool | Validates required files and cache-free release hygiene. | Keep |
| tools/syntax_check_release.py | Release tool | Syntax-checks release Python files without importing heavy dependencies. | Keep |
| VCFS/.gitignore | Repo hygiene | Original submodule ignore rules for datasets, logs, caches, and build artifacts. | Review |
| VCFS/benchmark.py | Segmentation utility | Benchmark/FPS utility for inference speed checks. | Review |
| VCFS/deeplab.py | Segmentation core | DeepLabv3+ inference class and preprocessing/postprocessing logic. | Keep |
| VCFS/get_miou.py | Segmentation core | mIoU evaluation entrypoint. | Keep |
| VCFS/json_to_dataset.py | Segmentation utility | Converts LabelMe-style annotations to segmentation masks. | Keep |
| VCFS/LICENSE | License | Upstream MIT license notice for the inherited DeepLabv3+ baseline code. | Keep |
| VCFS/nets/__init__.py | Segmentation model | Network architecture, backbone, attention, or training-loss module. | Keep |
| VCFS/nets/attention.py | Segmentation model | Network architecture, backbone, attention, or training-loss module. | Keep |
| VCFS/nets/deeplabv3_plus.py | Segmentation model | Network architecture, backbone, attention, or training-loss module. | Keep |
| VCFS/nets/deeplabv3_plus_1.py | Segmentation model | Network architecture, backbone, attention, or training-loss module. | Keep |
| VCFS/nets/deeplabv3_training.py | Segmentation model | Network architecture, backbone, attention, or training-loss module. | Keep |
| VCFS/nets/mobilenetv2.py | Segmentation model | Network architecture, backbone, attention, or training-loss module. | Keep |
| VCFS/nets/my_attention.py | Segmentation model | Network architecture, backbone, attention, or training-loss module. | Keep |
| VCFS/nets/plus_0630.py | Segmentation model | Network architecture, backbone, attention, or training-loss module. | Keep |
| VCFS/nets/xception.py | Segmentation model | Network architecture, backbone, attention, or training-loss module. | Keep |
| VCFS/predict.py | Segmentation core | Inference entrypoint for image, folder, video, FPS, and export workflows. | Keep |
| VCFS/README.md | Documentation | Upstream DeepLabv3+ README kept as baseline reference. | Review |
| VCFS/README_VFMFP.md | Documentation | VFM-FP-specific notes for the segmentation module. | Keep |
| VCFS/requirements.txt | Environment | Original lightweight requirements file for the segmentation baseline. | Review |
| VCFS/train.py | Segmentation core | Training entrypoint for the DeepLabv3+ facade parser. | Keep |
| VCFS/utils/__init__.py | Package marker | Marks the segmentation utils directory as a Python package. | Keep |
| VCFS/utils/callbacks.py | Segmentation core | Training callbacks, logging, checkpoints, and evaluation callbacks. | Keep |
| VCFS/utils/dataloader_latest.py | Segmentation core | Current training dataloader used by train.py. | Keep |
| VCFS/utils/dataloader_latest_ecp.py | Segmentation data | ECP-specific dataloader variant. | Review |
| VCFS/utils/fea_upscale.py | Segmentation model | Feature upscaling helper imported by some network variants. | Keep |
| VCFS/utils/utils.py | Segmentation core | Shared preprocessing, resizing, configuration, and utility functions. | Keep |
| VCFS/utils/utils_fit.py | Segmentation core | Training and validation loop helper functions. | Keep |
| VCFS/utils/utils_metrics.py | Segmentation core | Segmentation metrics and mIoU helpers. | Keep |
| VCFS/voc_annotation.py | Segmentation utility | Generates VOC-style split files. | Keep |
