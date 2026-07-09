"""Semantic Diffusion Augmentation (SDA) image generation.

This is the paper-aligned public entrypoint for the SDA generation stage. It
combines:

- DDE: prompt combinations over location, time, and weather domains.
- LTP: long-tail preference allocation from semantic masks.
- structural controls: Canny edges, semantic masks, and depth maps.

Heavy ML imports and model initialization are delayed until generation starts,
so ``--help`` and argument inspection work in lightweight environments.
"""

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

from ImageNet_color import Color


REPO_ROOT = Path(__file__).resolve().parents[2]

def repo_path(*parts):
    return str(REPO_ROOT.joinpath(*parts))


pipe = None
depth_estimator = None
image_processor = None
image_segmentor = None
cv2 = None
np = None
torch = None
Image = None


@dataclass(frozen=True)
class PromptCondition:
    location: str
    time: str
    weather: str

    @property
    def suffix(self):
        return f"located in {self.location};{self.time};{self.weather}"

    @property
    def tag(self):
        return "_".join([self.location, self.time, self.weather]).replace(" ", "-").lower()


def parse_csv_arg(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def default_conditions(profile):
    if profile == "paper_high":
        locations = ["France", "USA", "China", "Italy"]
        times = ["noon", "afternoon"]
        weathers = ["sunny", "cloudy"]
    elif profile == "paper_limited":
        return [
            PromptCondition("France", "noon", "sunny"),
            PromptCondition("USA", "noon", "sunny"),
            PromptCondition("China", "noon", "sunny"),
            PromptCondition("France", "afternoon", "sunny"),
            PromptCondition("China", "afternoon", "sunny"),
        ]
    elif profile == "norway":
        locations = ["Norway"]
        times = ["noon", "afternoon"]
        weathers = ["sunny", "cloudy"]
    else:
        raise ValueError(f"Unknown prompt profile: {profile}")

    return [
        PromptCondition(location, time, weather)
        for location in locations
        for time in times
        for weather in weathers
    ]


def build_conditions(args):
    if args.locations or args.times or args.weathers:
        locations = parse_csv_arg(args.locations) if args.locations else ["France", "USA", "China", "Italy"]
        times = parse_csv_arg(args.times) if args.times else ["noon", "afternoon"]
        weathers = parse_csv_arg(args.weathers) if args.weathers else ["sunny", "cloudy"]
        return [
            PromptCondition(location, time, weather)
            for location in locations
            for time in times
            for weather in weathers
        ]
    return default_conditions(args.prompt_profile)


def load_models(args):
    global pipe, depth_estimator, image_processor, image_segmentor
    global cv2, np, torch, Image

    if pipe is not None:
        return

    import cv2 as cv2_module
    import numpy as np_module
    import torch as torch_module
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    from PIL import Image as pil_image_module
    from transformers import AutoImageProcessor, UperNetForSemanticSegmentation, pipeline as transformers_pipeline

    cv2 = cv2_module
    np = np_module
    torch = torch_module
    Image = pil_image_module

    controlnets = [
        ControlNetModel.from_pretrained(args.canny_controlnet, torch_dtype=torch.float16, use_safetensors=True),
        ControlNetModel.from_pretrained(args.seg_controlnet, torch_dtype=torch.float16),
        ControlNetModel.from_pretrained(args.depth_controlnet, torch_dtype=torch.float16),
    ]

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.stable_diffusion_model,
        controlnet=controlnets,
        torch_dtype=torch.float16,
    ).to(args.device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as exc:
        print(f"xFormers attention is unavailable; continuing without it: {exc}")
    pipe.enable_model_cpu_offload()

    depth_estimator = transformers_pipeline("depth-estimation")
    image_processor = AutoImageProcessor.from_pretrained(args.segmentation_model)
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained(args.segmentation_model).to(args.device)


def read_split_ids(path):
    image_ids = []
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            image_ids.append(line.replace(",", " ").split()[0])
    return image_ids


def write_generation_records(args, records):
    if not records:
        return

    record_path = Path(args.generation_record)
    syn_list_path = Path(args.syn_list)
    source_list_path = Path(args.source_list)
    for path in (record_path, syn_list_path, source_list_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    with record_path.open("a", encoding="utf-8") as file:
        for item in records:
            file.write(
                "{syn_id},{source_id},{condition},{seed}\n".format(
                    syn_id=item["syn_id"],
                    source_id=item["source_id"],
                    condition=item["condition"],
                    seed=item["seed"],
                )
            )

    with syn_list_path.open("a", encoding="utf-8") as file:
        for item in records:
            file.write(item["syn_id"] + "\n")

    with source_list_path.open("a", encoding="utf-8") as file:
        for item in records:
            file.write(item["source_id"] + "\n")

def read_candidate_scores(path):
    values = []
    if not path:
        return values
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            image_id = parts[0]
            score = float(parts[1]) if len(parts) > 1 else 0.0
            values.append((image_id, score))
    return values


def resolve_image_ids(args):
    split_ids = read_split_ids(args.split_file)
    split_set = set(split_ids)
    candidate_scores = read_candidate_scores(args.candidate_file)
    if not candidate_scores:
        return split_ids, {image_id: 0.0 for image_id in split_ids}

    ids = []
    scores = {}
    for image_id, score in candidate_scores:
        if image_id in split_set:
            ids.append(image_id)
            scores[image_id] = score
    return ids, scores


def parse_class_ids(value):
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_mask_array(mask_path):
    import numpy as np_module

    try:
        import cv2 as cv2_module

        mask = cv2_module.imread(mask_path, cv2_module.IMREAD_UNCHANGED)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            return mask
    except Exception:
        pass

    from PIL import Image as pil_image_module

    mask = np_module.array(pil_image_module.open(mask_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def infer_minority_class_ids(image_ids, args):
    explicit = parse_class_ids(args.minority_class_ids)
    if explicit is not None:
        return explicit

    import numpy as np_module

    totals = np_module.zeros(args.num_classes, dtype=np_module.float64)
    for image_id in image_ids:
        mask_path = os.path.join(args.mask_dir, f"{image_id}.png")
        if not os.path.exists(mask_path):
            continue
        mask = load_mask_array(mask_path)
        counts = np_module.bincount(mask.reshape(-1), minlength=args.num_classes)[: args.num_classes]
        totals += counts

    dominant = totals[args.dominant_class_id] if args.dominant_class_id < len(totals) else totals.max()
    if dominant <= 0:
        return [class_id for class_id in range(1, args.num_classes) if class_id != args.dominant_class_id]

    threshold = dominant * args.minority_ratio
    return [
        class_id
        for class_id in range(1, args.num_classes)
        if class_id != args.dominant_class_id and 0 < totals[class_id] < threshold
    ]


def compute_ltp_counts(image_ids, args):
    import numpy as np_module

    minority_ids = infer_minority_class_ids(image_ids, args)
    if not minority_ids:
        return {image_id: args.fixed_count for image_id in image_ids}

    gamma = []
    for image_id in image_ids:
        mask_path = os.path.join(args.mask_dir, f"{image_id}.png")
        if not os.path.exists(mask_path):
            gamma.append(0.0)
            continue
        mask = load_mask_array(mask_path)
        total_pixels = float(mask.size)
        proportions = [(mask == class_id).sum() / total_pixels for class_id in minority_ids]
        gamma.append(float(np_module.mean(proportions)))

    gamma = np_module.array(gamma, dtype=np_module.float64)
    if gamma.sum() <= 0:
        weights = np_module.ones(len(image_ids), dtype=np_module.float64) / max(len(image_ids), 1)
    else:
        p = gamma / gamma.sum()
        logits = p / args.temperature
        logits = logits - logits.max()
        weights = np_module.exp(logits)
        weights = weights / weights.sum()

    target_total = args.target_total if args.target_total > 0 else len(image_ids)
    raw = weights * target_total
    counts = np_module.floor(raw).astype(int)
    counts = np_module.clip(counts, 0, args.max_per_image)

    remaining = int(target_total - counts.sum())
    if remaining > 0:
        order = np_module.argsort(-(raw - np_module.floor(raw)))
        for idx in order:
            if remaining <= 0:
                break
            if counts[idx] < args.max_per_image:
                counts[idx] += 1
                remaining -= 1

    return {image_id: int(count) for image_id, count in zip(image_ids, counts)}


def compute_legacy_score_counts(image_ids, scores, args):
    counts = {}
    for image_id in image_ids:
        score = scores.get(image_id, 0.0)
        gen = math.ceil(math.sqrt(float(score) + 1e-2) / args.loss_scale)
        counts[image_id] = min(args.max_per_image, gen + args.min_extra if gen < 2 else gen)
    return counts


def compute_generation_counts(image_ids, scores, args):
    if args.allocation_mode == "fixed":
        return {image_id: args.fixed_count for image_id in image_ids}
    if args.allocation_mode == "legacy_score":
        return compute_legacy_score_counts(image_ids, scores, args)
    return compute_ltp_counts(image_ids, args)


def build_prompt(condition, args):
    return args.positive_prompt + condition.suffix


def segmentation(img, use_predicted_seg=True, gt_img=None, device="cuda"):
    if use_predicted_seg:
        pixel_values = image_processor(img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            outputs = image_segmentor(pixel_values)
        seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[img.size[::-1]])[0].cpu().numpy()

        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(Color.palette):
            color_seg[seg == label, :] = color
        return Image.fromarray(color_seg)
    return gt_img


def process_image(image):
    can_image = cv2.Canny(np.array(image), 100, 200)
    can_image = np.dstack([can_image] * 3)
    canny_image = Image.fromarray(can_image)

    depth_image = depth_estimator(image)["depth"]
    depth_array = np.array(depth_image)
    depth_image = Image.fromarray(np.dstack([depth_array] * 3))
    return canny_image, depth_image


def generate_image(image, gt_image, prompt, args, generator):
    canny_image, depth_image = process_image(image)
    seg_image = segmentation(
        image,
        use_predicted_seg=args.semantic_source == "predicted",
        gt_img=gt_image,
        device=args.device,
    )

    return pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        image=[canny_image, seg_image, depth_image],
        generator=generator,
        num_images_per_prompt=1,
        num_inference_steps=args.num_inference_steps,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
    ).images[0]


def build_parser(defaults=None):
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", default=defaults.get("image_dir", repo_path("FacadeWHU_origin", "JPEGImages")))
    parser.add_argument("--mask-dir", default=defaults.get("mask_dir", repo_path("FacadeWHU_origin", "SegmentationClass")))
    parser.add_argument("--split-file", default=defaults.get("split_file", repo_path("FacadeWHU_origin", "txt", "trainval.txt")))
    parser.add_argument("--candidate-file", default=defaults.get("candidate_file", ""))
    parser.add_argument("--output-dir", default=defaults.get("output_dir", repo_path("SDA_output", "syn_image")))
    parser.add_argument("--seed-record", default=defaults.get("seed_record", repo_path("SDA_output", "txt", "Seed_record.txt")))
    parser.add_argument("--generation-record", default=defaults.get("generation_record", repo_path("SDA_output", "txt", "synthetic_pairs.csv")))
    parser.add_argument("--syn-list", default=defaults.get("syn_list", repo_path("SDA_output", "txt", "syn_trainval.txt")))
    parser.add_argument("--source-list", default=defaults.get("source_list", repo_path("SDA_output", "txt", "source_trainval_for_syn.txt")))
    parser.add_argument("--allocation-mode", default=defaults.get("allocation_mode", "ltp"), choices=("ltp", "legacy_score", "fixed"))
    parser.add_argument("--target-total", default=defaults.get("target_total", 0), type=int)
    parser.add_argument("--fixed-count", default=defaults.get("fixed_count", 1), type=int)
    parser.add_argument("--max-per-image", default=defaults.get("max_per_image", 5), type=int)
    parser.add_argument("--max-source-images", default=defaults.get("max_source_images", 0), type=int)
    parser.add_argument("--minority-class-ids", default=defaults.get("minority_class_ids", ""))
    parser.add_argument("--dominant-class-id", default=defaults.get("dominant_class_id", 3), type=int)
    parser.add_argument("--num-classes", default=defaults.get("num_classes", 7), type=int)
    parser.add_argument("--minority-ratio", default=defaults.get("minority_ratio", 0.1), type=float)
    parser.add_argument("--temperature", default=defaults.get("temperature", 1 / 500), type=float)
    parser.add_argument("--min-extra", default=defaults.get("min_extra", 2), type=int)
    parser.add_argument("--loss-scale", default=defaults.get("loss_scale", 20.0), type=float)
    parser.add_argument("--prompt-profile", default=defaults.get("prompt_profile", "paper_high"), choices=("paper_high", "paper_limited", "norway"))
    parser.add_argument("--locations", default=defaults.get("locations", ""))
    parser.add_argument("--times", default=defaults.get("times", ""))
    parser.add_argument("--weathers", default=defaults.get("weathers", ""))
    parser.add_argument("--positive-prompt", default=defaults.get("positive_prompt", "high quality, extremely detailed, 8K, a panoramic street view photograph, physically logical; building, windows, doors, shops, balcony with railing, roof; the background mainly contains sky or sidewalk; the foreground may include the front of a car; colorful ;"))
    parser.add_argument("--negative-prompt", default=defaults.get("negative_prompt", "painting style, black and white style, overlapping structures, building or wall as background"))
    parser.add_argument("--semantic-source", default=defaults.get("semantic_source", "predicted"), choices=("predicted", "ground_truth"))
    parser.add_argument("--seed-min", default=defaults.get("seed_min", 1), type=int)
    parser.add_argument("--seed-max", default=defaults.get("seed_max", 900), type=int)
    parser.add_argument("--dry-run", action="store_true", default=defaults.get("dry_run", False))
    parser.add_argument("--device", default=defaults.get("device", "cuda"))
    parser.add_argument("--num-inference-steps", default=defaults.get("num_inference_steps", 20), type=int)
    parser.add_argument("--controlnet-conditioning-scale", nargs=3, default=defaults.get("controlnet_conditioning_scale", [0.9, 0.85, 1.0]), type=float)
    parser.add_argument("--canny-controlnet", default=defaults.get("canny_controlnet", "lllyasviel/sd-controlnet-canny"))
    parser.add_argument("--seg-controlnet", default=defaults.get("seg_controlnet", "lllyasviel/sd-controlnet-seg"))
    parser.add_argument("--depth-controlnet", default=defaults.get("depth_controlnet", "lllyasviel/sd-controlnet-depth"))
    parser.add_argument("--stable-diffusion-model", default=defaults.get("stable_diffusion_model", "runwayml/stable-diffusion-v1-5"))
    parser.add_argument("--segmentation-model", default=defaults.get("segmentation_model", "openmmlab/upernet-convnext-small"))
    return parser


def main(argv=None, defaults=None):
    args = build_parser(defaults).parse_args(argv)
    image_ids, scores = resolve_image_ids(args)
    if args.max_source_images > 0:
        image_ids = image_ids[: args.max_source_images]

    counts = compute_generation_counts(image_ids, scores, args)
    conditions = build_conditions(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    seed_parent = Path(args.seed_record).parent
    if str(seed_parent) not in ("", "."):
        seed_parent.mkdir(parents=True, exist_ok=True)

    planned_total = sum(counts.values())
    print(f"SDA generation plan: {len(image_ids)} source images, {planned_total} synthetic images, {len(conditions)} DDE prompts")
    if args.dry_run:
        for image_id in image_ids[:10]:
            print(f"{image_id}: {counts.get(image_id, 0)}")
        return
    if planned_total <= 0:
        return

    load_models(args)

    for image_id in image_ids:
        gen_num = counts.get(image_id, 0)
        if gen_num <= 0:
            continue

        image_path = os.path.join(args.image_dir, f"{image_id}.jpg")
        mask_path = os.path.join(args.mask_dir, f"{image_id}.png")
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Missing file: {image_path} or {mask_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        gt_image = Image.open(mask_path)

        for n in range(gen_num):
            seed = random.randint(args.seed_min, args.seed_max)
            condition = random.choice(conditions)
            prompt = build_prompt(condition, args)
            torch.manual_seed(seed)
            generator = torch.Generator(device=args.device).manual_seed(seed)

            with open(args.seed_record, "a") as file:
                file.write(f"{image_id},{seed},{condition.tag}\n")

            try:
                result = generate_image(image, gt_image, prompt, args, generator)
                output_name = f"syn_{image_id}_{condition.tag}_{n}.jpg"
                syn_id = Path(output_name).stem
                result.save(os.path.join(args.output_dir, output_name))
                write_generation_records(
                    args,
                    [
                        {
                            "syn_id": syn_id,
                            "source_id": image_id,
                            "condition": condition.tag,
                            "seed": seed,
                        }
                    ],
                )
                print(f"Generated: {output_name}")
            except Exception as exc:
                print(f"Generation failed: {image_id}_{n} - {exc}")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


