"""Rank generated facade images with DINOv2 feature distance."""

import argparse
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

def repo_path(*parts):
    return str(REPO_ROOT.joinpath(*parts))


DEFAULTS = {
    "ori_jpeg_path": repo_path("FacadeWHU_origin", "JPEGImages"),
    "syn_jpeg_path": repo_path("SDA_output", "syn_image"),
    "ori_txt": repo_path("SDA_output", "txt", "source_trainval_for_syn.txt"),
    "syn_txt": repo_path("SDA_output", "txt", "syn_trainval.txt"),
    "save_path": repo_path("SDA_output", "scf"),
    "patch_h": 40,
    "patch_w": 40,
    "feat_dim": 384,
    "model_repo": "facebookresearch/dinov2",
    "model_name": "dinov2_vits14",
    "output_mode": "sorted_indices",
    "with_scores_name": "low_with.txt",
    "ids_name": "low_wout.txt",
    "scores_name": "re_dis_ynl_st3.txt",
    "filtered_name": "scf_keep.txt",
    "discarded_name": "scf_discard.txt",
    "threshold_scale": 1.0,
    "sort_image_ids": False,
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
    parser.add_argument("--output-mode", default=cfg["output_mode"], choices=("sorted_indices", "syn_scores", "filtered_ids"))
    parser.add_argument("--with-scores-name", default=cfg["with_scores_name"])
    parser.add_argument("--ids-name", default=cfg["ids_name"])
    parser.add_argument("--scores-name", default=cfg["scores_name"])
    parser.add_argument("--filtered-name", default=cfg["filtered_name"])
    parser.add_argument("--discarded-name", default=cfg["discarded_name"])
    parser.add_argument("--threshold-scale", default=cfg["threshold_scale"], type=float)
    parser.add_argument("--sort-image-ids", action="store_true", default=cfg["sort_image_ids"])
    parser.add_argument("--echo-scores", action="store_true", default=cfg["echo_scores"])
    return parser


def load_image_ids(txt_path, sort_ids=False):
    with open(os.path.join("", txt_path), "r") as file:
        image_ids = [line.strip().replace(",", " ").split()[0] for line in file if line.strip()]
    if sort_ids:
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
                print(image_id + "," + str(loss.item()))
                print("\n")


def write_filtered_ids(batch_loss, syn_image_ids, save_path, filtered_name, discarded_name, threshold_scale, torch):
    threshold = torch.mean(batch_loss) + threshold_scale * torch.std(batch_loss, unbiased=False)
    keep = []
    discard = []
    for i, image_id in enumerate(syn_image_ids):
        item = (image_id, batch_loss[i].item())
        if batch_loss[i] <= threshold:
            keep.append(item)
        else:
            discard.append(item)

    with open(os.path.join(save_path, filtered_name), "w") as file:
        for image_id, score in keep:
            file.write(f"{image_id},{score}\n")

    with open(os.path.join(save_path, discarded_name), "w") as file:
        for image_id, score in discard:
            file.write(f"{image_id},{score}\n")

    print(f"SCF threshold: {threshold.item()}")
    print(f"SCF kept {len(keep)} samples and discarded {len(discard)} samples")


def run_ranking(args):
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from tqdm import tqdm

    transform = build_transform(T, args.patch_h, args.patch_w)
    dinov2_model = torch.hub.load(args.model_repo, args.model_name, source="github").cuda()

    ori_image_ids = load_image_ids(args.ori_txt, args.sort_image_ids)
    syn_image_ids = load_image_ids(args.syn_txt, args.sort_image_ids)
    if len(ori_image_ids) != len(syn_image_ids):
        raise ValueError(
            "SCF requires aligned original and synthetic id lists with equal length; "
            f"got {len(ori_image_ids)} original ids and {len(syn_image_ids)} synthetic ids."
        )

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
        print("Sorted losses and indices have been written.")
    elif args.output_mode == "syn_scores":
        write_syn_scores(batch_loss, syn_image_ids, args.save_path, args.scores_name, args.echo_scores)
        print("Synthetic image scores have been written.")
    else:
        write_filtered_ids(
            batch_loss,
            syn_image_ids,
            args.save_path,
            args.filtered_name,
            args.discarded_name,
            args.threshold_scale,
            torch,
        )


def main(defaults=None):
    parser = build_parser(defaults)
    args = parser.parse_args()
    run_ranking(args)


if __name__ == "__main__":
    main()
