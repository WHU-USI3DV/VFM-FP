"""Run DINO ranking with the default preset."""

from dino_rank_generated import main


if __name__ == "__main__":
    main({
        "ori_jpeg_path": "FacadeWHU_origin/JPEGImages",
        "syn_jpeg_path": "norway/syn_image",
        "ori_txt": "norway/txt/trainval_w.txt",
        "syn_txt": "norway/txt/trainval.txt",
        "save_path": "norway/low_result",
        "sort_image_ids": True,
        "output_mode": "sorted_indices",
        "with_scores_name": "low_with.txt",
        "ids_name": "low_wout.txt",
    })
