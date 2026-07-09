"""Compatibility wrapper for the renamed legacy DINO ranking variant."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("legacy") / "extract_cul_norway_variant.py"), run_name="__main__")
