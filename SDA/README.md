# SDA Code

SDA contains the data-level expansion workflow used before training the facade parser.

## Subdirectories

- diffusion/: diffusion and ControlNet generation scripts.
- diffusion/legacy/: local data-maintenance helpers with descriptive names and compatibility wrappers in diffusion/.
- DINO_extract/: DINO feature extraction, quality ranking, sample selection, and related visualization scripts.
- DINO_extract/legacy/: local DINO ranking and dataset-maintenance helpers with descriptive names and compatibility wrappers in DINO_extract/.

## Main Scripts

- diffusion/Mul_Ab_norway.py: configurable structured diffusion generation script with the original Norway defaults.
- diffusion/Mul_Ab_2.py: compatibility wrapper for an earlier generation variant kept in diffusion/legacy/.
- diffusion/Mul_Ab_2_diversity.py: compatibility wrapper for a diversity-oriented generation variant kept in diffusion/legacy/.
- DINO_extract/dino_rank_generated.py: configurable DINOv2 feature-distance ranking CLI.
- DINO_extract/Extract_Cul.py: compatibility entrypoint for the default ranking run.
- DINO_extract/Extract_Cul_facadewhu.py: compatibility entrypoint for the FacadeWHU ranking variant.
- DINO_extract/Extract_Cul_ecp.py: compatibility entrypoint for the ECP ranking variant.
- DINO_extract/dino_extract.py and DINO_extract/get_feature.py: compatibility wrappers for legacy visualization helpers.

## Notes for Public Release

The DINO ranking entrypoints and diffusion/Mul_Ab_norway.py now accept command-line paths while preserving the accepted-paper defaults. The remaining diffusion variants still contain hard-coded local dataset paths and should be parameterized with the same conservative pattern.

Generated images, DINO visualizations, low-quality sample folders, and local model caches are excluded by the root .gitignore.



