# Script Inventory

This inventory separates public entrypoints from local experiment utilities. It is a conservative documentation pass: no core algorithmic behavior is changed.

## Segmentation Training and Inference

Public entrypoints in VCFS/:

- train.py: train the DeepLabv3+ facade parser.
- predict.py: run inference for single images, folders, videos, FPS checks, or ONNX export.
- get_miou.py: evaluate segmentation results.
- voc_annotation.py: generate VOC-style split files.
- json_to_dataset.py: convert LabelMe-style annotations to segmentation data.

Core model and training modules:

- nets/deeplabv3_plus.py
- nets/deeplabv3_training.py
- nets/mobilenetv2.py
- nets/xception.py
- nets/attention.py
- utils/dataloader_latest.py
- utils/utils_fit.py
- utils/utils_metrics.py
- utils/callbacks.py
- utils/utils.py

Compatibility wrappers:

- deeplab_ori.py
- get_miou_ori.py
- benchimark.py
- summary.py
- turn.py
- delete.py
- unsort.py
- utils/dataloader.py
- utils/dataloader_ori.py
- utils/callbacks_ori.py
- utils/utils_ori.py
- utils/dataloader_lastest.py
- utils/dataloader_lastest_ecp.py
- utils/delete.py
- utils/downto.py
- utils/Rename.py
- utils/turn1.py

Legacy experiment helpers moved out of the main entrypoint surface:

- legacy/deeplab_original.py
- legacy/get_miou_original.py
- legacy/model_summary.py
- legacy/remap_segmentation_colors.py
- legacy/filter_split_file.py
- legacy/shuffle_split_file.py
- utils/legacy/dataloader_baseline.py
- utils/legacy/dataloader_original.py
- utils/legacy/callbacks_original.py
- utils/legacy/utils_original.py
- utils/legacy/prune_image_triplets.py
- utils/legacy/random_downsample_folder.py
- utils/legacy/add_syn_prefix.py
- utils/legacy/remap_segmentation_colors_ecp.py

Legacy or experiment-specific segmentation scripts still not reorganized:

- benchmark.py
- utils/fea_upscale.py is retained in place because some network definitions still import it.

## SDA Diffusion Scripts

Public or near-public entrypoints:

- SDA/diffusion/Mul_Ab_norway.py: configurable structured ControlNet generation script with accepted-paper defaults.
- SDA/diffusion/voc_annotation.py: VOC split generation helper.
- SDA/diffusion/ImageNet_color.py: semantic color palette used by generation scripts.

Compatibility wrappers kept at their historical paths:

- SDA/diffusion/Mul_Ab_2.py
- SDA/diffusion/Mul_Ab_2_diversity.py
- SDA/diffusion/check_quality.py
- SDA/diffusion/delete.py
- SDA/diffusion/downto.py
- SDA/diffusion/Rename.py
- SDA/diffusion/small.py
- SDA/diffusion/small2.py
- SDA/diffusion/sort.py
- SDA/diffusion/unsort.py

Legacy diffusion maintenance helpers with clearer names:

- SDA/diffusion/legacy/mul_ab_norway_original.py
- SDA/diffusion/legacy/mul_ab_2_original.py
- SDA/diffusion/legacy/mul_ab_2_diversity_original.py
- SDA/diffusion/legacy/filter_zero_quality_scores.py
- SDA/diffusion/legacy/keep_first_generated_per_index.py
- SDA/diffusion/legacy/random_downsample_folder.py
- SDA/diffusion/legacy/copy_trainval_images_with_syn_prefix.py
- SDA/diffusion/legacy/count_facade_detail_pixels.py
- SDA/diffusion/legacy/count_facade_detail_pixels_with_header.py
- SDA/diffusion/legacy/sort_score_file_desc.py
- SDA/diffusion/legacy/shuffle_score_file.py

## SDA DINO Feature Scripts

Public or near-public entrypoints:

- SDA/DINO_extract/dino_rank_generated.py: configurable DINOv2 feature-distance ranking CLI.
- SDA/DINO_extract/Extract_Cul.py: compatibility entrypoint for the default generated-image ranking run.
- SDA/DINO_extract/Extract_Cul_facadewhu.py: compatibility entrypoint for the FacadeWHU ranking variant.
- SDA/DINO_extract/Extract_Cul_ecp.py: compatibility entrypoint for the ECP ranking variant.


Compatibility wrappers kept at their historical paths:

- SDA/DINO_extract/deeplabv3+.py
- SDA/DINO_extract/dino_extract.py
- SDA/DINO_extract/get_feature.py
- SDA/DINO_extract/Check_quality.py
- SDA/DINO_extract/chouqu.py
- SDA/DINO_extract/Com_Cmean.py
- SDA/DINO_extract/delete.py
- SDA/DINO_extract/Extract_Cul2.py
- SDA/DINO_extract/random_get.py
- SDA/DINO_extract/Rename.py
- SDA/DINO_extract/small.py
- SDA/DINO_extract/SORT_ascend.py

Legacy DINO maintenance helpers with clearer names:

- SDA/DINO_extract/legacy/deeplabv3_plus_visualization.py
- SDA/DINO_extract/legacy/dino_pca_single_image.py
- SDA/DINO_extract/legacy/deeplab_encoder_feature_visualization.py
- SDA/DINO_extract/legacy/make_quality_comparison_images.py
- SDA/DINO_extract/legacy/copy_ranked_generated_images.py
- SDA/DINO_extract/legacy/average_scores_by_source_id.py
- SDA/DINO_extract/legacy/keep_first_syn_image_per_index.py
- SDA/DINO_extract/legacy/extract_cul_default_original.py
- SDA/DINO_extract/legacy/extract_cul_facadewhu_original.py
- SDA/DINO_extract/legacy/extract_cul_ecp_original.py
- SDA/DINO_extract/legacy/extract_cul_norway_variant.py
- SDA/DINO_extract/legacy/random_downsample_folder.py
- SDA/DINO_extract/legacy/add_syn_prefix_to_images.py
- SDA/DINO_extract/legacy/count_roof_pixels.py
- SDA/DINO_extract/legacy/sort_score_file_asc.py

Other local or experiment-specific helpers still not reorganized:

- None currently identified in SDA/DINO_extract.

## Recommended Next Cleanup

- Continue moving model initialization inside main functions for the remaining diffusion entrypoints.
- Keep current defaults while documenting the accepted-paper reproduction paths.
- Decide whether to publish a clean release copy or remove local data, generated images, model weights, and feature visualizations from this workspace before creating the public repository.




