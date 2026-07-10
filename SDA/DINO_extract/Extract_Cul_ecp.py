"""Run DINO ranking with the ECP preset."""

from dino_rank_generated import main


if __name__ == "__main__":
    main({
        "ori_jpeg_path": "ecp/ori_ecp",
        "syn_jpeg_path": "ecp/ST3",
        "ori_txt": "ecp/txt3/trainval_o2s3.txt",
        "syn_txt": "ecp/txt3/trainval_syn3.txt",
        "save_path": "ecp/Re_DINO",
        "sort_image_ids": True,
        "output_mode": "syn_scores",
        "scores_name": "re3.txt",
    })
