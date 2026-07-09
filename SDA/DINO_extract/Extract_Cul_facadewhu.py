"""Compatibility entrypoint for the FacadeWHU DINO generated-image ranking run."""

from dino_rank_generated import main


if __name__ == "__main__":
    main({
        "ori_jpeg_path": "FacadeWHU_origin/JPEGImages",
        "syn_jpeg_path": "norway/syn_image",
        "ori_txt": "norway/txt/trainval_w.txt",
        "syn_txt": "norway/txt/trainval.txt",
        "save_path": "norway/low_result",
        "output_mode": "syn_scores",
        "scores_name": "re_dis_ynl_st3.txt",
        "echo_scores": True,
    })
