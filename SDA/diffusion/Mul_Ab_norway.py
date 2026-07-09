"""Compatibility entrypoint for the original Norway SDA generation run.

The general paper-aligned SDA implementation lives in
``semantic_diffusion_augmentation.py``. This wrapper preserves the previous
Norway defaults for reproducing the accepted-paper cross-domain setting.
"""

from semantic_diffusion_augmentation import main


NORWAY_DEFAULTS = {
    "image_dir": "FacadeWHU_origin/JPEGImages",
    "mask_dir": "FacadeWHU_origin/SegmentationClass",
    "split_file": "FacadeWHU_origin/txt/trainval.txt",
    "candidate_file": "Norway/small/output_sort_1.txt",
    "output_dir": "Norway/output",
    "allocation_mode": "legacy_score",
    "prompt_profile": "norway",
    "max_source_images": 2,
    "seed_record": "Seed_record.txt",
}


if __name__ == "__main__":
    main(defaults=NORWAY_DEFAULTS)
