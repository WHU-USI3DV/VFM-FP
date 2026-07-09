"""Configurable DINOv2 feature-distance ranking for generated facade images.

The default presets used by Extract_Cul*.py preserve the accepted-paper
experiment paths and output formats. Imports for torch/torchvision/PIL happen
inside run_ranking so --help does not initialize or download models.
"""

import argparse
import os
from pathlib import Path


DEFAULTS = {
    "ori_jpeg_path": "FacadeWHU_origin/JPEGImages",
    "syn_jpeg_path": "norway/syn_image",
    "ori_txt": "norway/txt/trainval_w.txt",
    "syn_txt": "norway/txt/trainval.txt",
    "save_path": "norway/low_result",
    "patch_h": 40,
    "patch_w": 40,
    "feat_dim": 384,
    "model_repo": "facebookresearch/dinov2",
    "model_name": "dinov2_vits14",
    "output_mode": "sorted_indices",
    "with_scores_name": "low_with.txt",
    "ids_name": "low_wout.txt",
    "scores_name": "re_dis_ynl_st3.txt",
    "echo_scores": False,
}


def build_parser(defaults=None):
    cfg = dict(DEFAULTS)
    if defaults:
        cfg.update(defaults)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ori-jpeg-path", default=cfg["ori_jpeg_path"])
    parser.add_argument("--syn-jpeg-path", default=cfg["syn_jpeg_path"])
    parser.add_argument("--ori-txt", default=cfg["ori_txt"])
    parser.add_argument("--syn-txt", default=cfg["syn_txt"])
    parser.add_argument("--save-path", default=cfg["save_path"])
    parser.add_argument("--patch-h", default=cfg["patch_h"], type=int)
    parser.add_argument("--patch-w", default=cfg["patch_w"], type=int)
    parser.add_argument("--feat-dim", default=cfg["feat_dim"], type=int)
    parser.add_argument("--model-repo", default=cfg["model_repo"])
    parser.add_argument("--model-name", default=cfg["model_name"])
    parser.add_argument("--output-mode", default=cfg["output_mode"], choices=("sorted_indices", "syn_scores"))
    parser.add_argument("--with-scores-name", default=cfg["with_scores_name"])
    parser.add_argument("--ids-name", default=cfg["ids_name"])
    parser.add_argument("--scores-name", default=cfg["scores_name"])
    parser.add_argument("--echo-scores", action="store_true", default=cfg["echo_scores"])
    return parser


def load_image_ids(txt_path):
    with open(os.path.join("", txt_path), "r") as file:
        image_ids = file.read().splitlines()
    image_ids.sort()
    return image_ids


def build_transform(torchvision_transforms, patch_h, patch_w):
    return torchvision_transforms.Compose([
        torchvision_transforms.GaussianBlur(9, sigma=(0.1, 2.0)),
        torchvision_transforms.Resize((patch_h * 14, patch_w * 14)),
        torchvision_transforms.CenterCrop((patch_h * 14, patch_w * 14)),
        torchvision_transforms.ToTensor(),
        torchvision_transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def extract_features(image_ids, image_dir, model, transform, patch_h, patch_w, feat_dim, torch, Image, tqdm):
    features = torch.zeros(len(image_ids), patch_h * patch_w, feat_dim)
    images_tensor = torch.zeros(1, 3, patch_h * 14, patch_w * 14)

    pbar = tqdm(total=len(image_ids), desc="configurable", unit="B", unit_scale=True, unit_divisor=1024)
    for i, image_id in enumerate(image_ids):
        pbar.update(1)

        image_path = os.path.join(image_dir, image_id + ".jpg")
        image = Image.open(image_path)
        img = image.convert("RGB")

        images_tensor[0] = transform(img)[:3]
        images_tensor = images_tensor.cuda()

        with torch.no_grad():
            features_dict = model.forward_features(images_tensor)
            features[i] = features_dict["x_norm_patchtokens"]

    pbar.close()
    return features


def compute_batch_loss(ori_features, syn_features, patch_h, patch_w, feat_dim, torch):
    ori_features = ori_features.reshape(len(ori_features), feat_dim, patch_h, patch_w).cpu()
    syn_features = syn_features.reshape(len(syn_features), feat_dim, patch_h, patch_w).cpu()

    mean_o = torch.mean(ori_features, dim=(2, 3), keepdim=True)
    var_o = torch.var(ori_features, dim=(2, 3), keepdim=True)

    mean_s = torch.mean(syn_features, dim=(2, 3), keepdim=True)
    var_s = torch.var(syn_features, dim=(2, 3), keepdim=True)

    sigma_o = torch.sqrt(var_o + 1e-6)
    sigma_s = torch.sqrt(var_s + 1e-6)

    ada_ori = (ori_features - mean_o) / sigma_o
    ada_syn = (syn_features - mean_s) / sigma_s

    content_loss = torch.mean((ada_ori - ada_syn) ** 2, dim=(2, 3))
    return torch.sum(content_loss, dim=1)


def write_sorted_indices(batch_loss, save_path, with_scores_name, ids_name, torch):
    sorted_loss, indices = torch.sort(batch_loss, descending=False)
    indices = indices + 1

    with open(os.path.join(save_path, with_scores_name), "w") as file:
        for index, dis in zip(indices, sorted_loss):
            file.write(f"{index.item()}, {dis.item()}\n")

    with open(os.path.join(save_path, ids_name), "w") as file:
        for index, _dis in zip(indices, sorted_loss):
            file.write(f"{index.item()}\n")


def write_syn_scores(batch_loss, syn_image_ids, save_path, scores_name, echo_scores=False):
    app = []
    for i, image_id in enumerate(syn_image_ids):
        app.append((image_id, batch_loss[i]))

    with open(os.path.join(save_path, scores_name), "w") as file:
        for image_id, loss in app:
            file.write(image_id)
            file.write(",")
            file.write(f"{loss.item()}\n")
            if echo_scores:
                print(image_id + "," + str(loss.item))
                print("\n")


def run_ranking(args):
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from tqdm import tqdm

    transform = build_transform(T, args.patch_h, args.patch_w)
    dinov2_model = torch.hub.load(args.model_repo, args.model_name, source="github").cuda()

    ori_image_ids = load_image_ids(args.ori_txt)
    syn_image_ids = load_image_ids(args.syn_txt)

    print("Start Train")
    ori_features = extract_features(
        ori_image_ids,
        args.ori_jpeg_path,
        dinov2_model,
        transform,
        args.patch_h,
        args.patch_w,
        args.feat_dim,
        torch,
        Image,
        tqdm,
    )

    print("Start Train")
    syn_features = extract_features(
        syn_image_ids,
        args.syn_jpeg_path,
        dinov2_model,
        transform,
        args.patch_h,
        args.patch_w,
        args.feat_dim,
        torch,
        Image,
        tqdm,
    )

    batch_loss = compute_batch_loss(ori_features, syn_features, args.patch_h, args.patch_w, args.feat_dim, torch)

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    if args.output_mode == "sorted_indices":
        write_sorted_indices(batch_loss, args.save_path, args.with_scores_name, args.ids_name, torch)
    else:
        write_syn_scores(batch_loss, syn_image_ids, args.save_path, args.scores_name, args.echo_scores)

    print("Sorted losses and indices have been written to ...")


def main(defaults=None):
    parser = build_parser(defaults)
    args = parser.parse_args()
    run_ranking(args)


if __name__ == "__main__":
    main()
