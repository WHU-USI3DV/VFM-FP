# Legacy Notes

This public release keeps the active VFM-FP implementation and documented compatibility entrypoints, while removing unreferenced local experiment variants from the source tree.

Removed legacy files included old DINO ranking experiments, inactive DeepLab variants, and a duplicate ECP dataloader variant. The active VCFS training path remains `VCFS/nets/deeplabv3_plus.py` with `VCFS/utils/dataloader_latest.py`.

The preserved compatibility entrypoints are:

- `SDA/diffusion/Mul_Ab_norway.py`
- `SDA/DINO_extract/Extract_Cul.py`
- `SDA/DINO_extract/Extract_Cul_facadewhu.py`
- `SDA/DINO_extract/Extract_Cul_ecp.py`