"""Structured ControlNet generation for the Norway SDA setting.

Defaults preserve the accepted-paper experiment paths and generation settings.
Heavy ML imports and model initialization are delayed until the script actually
runs, so --help works in lightweight environments.
"""

import argparse
import math
import os
import random

from ImageNet_color import Color


pipe = None
depth_estimator = None
image_processor = None
image_segmentor = None
cv2 = None
np = None
torch = None
Image = None


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
        ControlNetModel.from_pretrained(
            args.canny_controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ),
        ControlNetModel.from_pretrained(
            args.seg_controlnet,
            torch_dtype=torch.float16,
        ),
        ControlNetModel.from_pretrained(
            args.depth_controlnet,
            torch_dtype=torch.float16,
        ),
    ]

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.stable_diffusion_model,
        controlnet=controlnets,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as exc:
        print(f"xFormers attention is unavailable; continuing without it: {exc}")
    pipe.enable_model_cpu_offload()

    depth_estimator = transformers_pipeline("depth-estimation")
    image_processor = AutoImageProcessor.from_pretrained(args.segmentation_model)
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained(args.segmentation_model).to("cuda")


def search_string_in_file(target_string, filename):
    with open(filename, "r") as file:
        return any(target_string in line.split(",")[0] for line in file)


def parse_file_to_array(filename, has_loss=True):
    with open(filename, "r") as file:
        if has_loss:
            return [line.strip().split() for line in file]
        return [line.strip() for line in file]


def segmentation(img, seg=True, gt_img=None):
    if seg:
        pixel_values = image_processor(img, return_tensors="pt").pixel_values.to("cuda")
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


def generate_image(image, gt_image, prompt, negative_prompt, generator):
    canny_image, depth_image = process_image(image)
    seg_image = segmentation(image, seg=True, gt_img=gt_image)

    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=[canny_image, seg_image, depth_image],
        generator=generator,
        num_images_per_prompt=1,
        num_inference_steps=20,
        controlnet_conditioning_scale=[0.9, 0.85, 1],
    ).images[0]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder-path", default="FacadeWHU_origin/JPEGImages")
    parser.add_argument("--gt-path", default="FacadeWHU_origin/SegmentationClass")
    parser.add_argument("--save-path", default="Norway/output")
    parser.add_argument("--filename", default="FacadeWHU_origin/txt/trainval.txt")
    parser.add_argument("--small-txt", default="Norway/small/output_sort_1.txt")
    parser.add_argument("--seed-record", default="Seed_record.txt")
    parser.add_argument("--max-count", default=2, type=int)
    parser.add_argument("--min-extra", default=2, type=int)
    parser.add_argument("--loss-scale", default=20.0, type=float)
    parser.add_argument("--seed-min", default=1, type=int)
    parser.add_argument("--seed-max", default=900, type=int)
    parser.add_argument("--canny-controlnet", default="lllyasviel/sd-controlnet-canny")
    parser.add_argument("--seg-controlnet", default="lllyasviel/sd-controlnet-seg")
    parser.add_argument("--depth-controlnet", default="lllyasviel/sd-controlnet-depth")
    parser.add_argument("--stable-diffusion-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--segmentation-model", default="openmmlab/upernet-convnext-small")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    data_array = parse_file_to_array(args.small_txt, has_loss=True)
    os.makedirs(args.save_path, exist_ok=True)

    load_models(args)

    count = 0
    prompt_base = (
        "high quality, extremely detailed, 8K, a panoramic street view photograph, "
        "physically logical; building, windows, doors, shops, balcony with railing, roof; "
        "the background mainly contains sky or sidewalk; the foreground may include the front of a car; colorful ;"
    )
    negative_prompt = "painting style, black and white style, overlapping structures, building or wall as background"
    prompts = [
        prompt_base + "located in Norway;noon;sunny",
        prompt_base + "located in Norway;afternoon;sunny",
        prompt_base + "located in Norway;noon;cloudy",
        prompt_base + "located in Norway;afternoon;cloudy",
    ]

    for item in data_array:
        if count >= args.max_count:
            break

        if len(item) == 1:
            index_value = item[0]
            loss_value = 0.0
        else:
            index_value, loss_value = item[0], float(item[1])

        if not search_string_in_file(index_value, args.filename):
            continue

        count += 1

        gen = math.sqrt(float(loss_value) + 1e-2) / args.loss_scale
        gen = math.ceil(gen)
        gen_num = gen + args.min_extra if gen < 2 else gen

        img_path = os.path.join(args.folder_path, f"{index_value}.jpg")
        gt_img_path = os.path.join(args.gt_path, f"{index_value}.png")

        if not os.path.exists(img_path) or not os.path.exists(gt_img_path):
            print(f"Missing file: {img_path} or {gt_img_path}")
            continue

        image = Image.open(img_path)
        gt_image = Image.open(gt_img_path)

        for n in range(gen_num):
            seed = random.randint(args.seed_min, args.seed_max)
            torch.manual_seed(seed)

            with open(args.seed_record, "a") as file:
                file.write(f"{seed}\n")

            prompt = random.choice(prompts)
            generator = torch.Generator(device="cuda").manual_seed(seed)

            try:
                result = generate_image(image, gt_image, prompt, negative_prompt, generator)
                result.save(os.path.join(args.save_path, f"syn_{index_value}_cloudy_{n}.jpg"))
                print(f"Generated: {index_value}_{n}")
            except Exception as exc:
                print(f"Generation failed: {index_value}_{n} - {exc}")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
