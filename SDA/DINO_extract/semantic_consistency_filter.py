"""Semantic Consistency Filtering (SCF) entrypoint.

This wrapper runs the DINOv2 feature-distance scorer in filtered output mode.
The default original-id list is the generation-time source list, where each
entry is aligned one-to-one with the generated image id list.
"""

from dino_rank_generated import main


SCF_DEFAULTS = {
    "ori_txt": "SDA_output/txt/source_trainval_for_syn.txt",
    "syn_txt": "SDA_output/txt/syn_trainval.txt",
    "output_mode": "filtered_ids",
}


if __name__ == "__main__":
    main(defaults=SCF_DEFAULTS)
