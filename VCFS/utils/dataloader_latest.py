import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class DeeplabDataset(Dataset):
    def __init__(
        self,
        annotation_lines,
        input_shape,
        num_classes,
        train,
        dataset_path,
        augmentation_profile="legacy_resize_hsv",
    ):
        super().__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path
        self.augmentation_profile = augmentation_profile

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        image_id = self.annotation_lines[index].split()[0]
        image = Image.open(os.path.join(self.dataset_path, "JPEGImages", image_id + ".jpg"))
        mask = Image.open(os.path.join(self.dataset_path, "SegmentationClass", image_id + ".png"))

        image, mask = self.get_random_data(image, mask, self.input_shape, random=self.train)
        image = np.transpose(preprocess_input(np.array(image, np.float64)), [2, 0, 1])

        mask = np.array(mask)
        mask[mask >= self.num_classes] = self.num_classes
        one_hot_mask = np.eye(self.num_classes + 1)[mask.reshape([-1])]
        one_hot_mask = one_hot_mask.reshape((*self.input_shape, self.num_classes + 1))
        return image, mask, one_hot_mask

    @staticmethod
    def rand(a=0, b=1):
        return np.random.rand() * (b - a) + a

    def photometric_distort(self, image):
        if self.rand() < 0.5:
            delta = np.random.uniform(-32, 32)
            image = np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)

        if self.rand() < 0.2:
            alpha = np.random.uniform(0.5, 1.5)
            image = cv2.addWeighted(image, alpha, np.zeros_like(image), 0, 0)
            image = np.clip(image, 0, 255).astype(np.uint8)

        hue_delta, sat_delta, val_delta = 0.1, 0.7, 0.3
        random_factors = np.random.uniform(-1, 1, 3) * [hue_delta, sat_delta, val_delta] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
        dtype = image.dtype
        values = np.arange(0, 256, dtype=random_factors.dtype)

        lut_hue = ((values * random_factors[0]) % 180).astype(dtype)
        lut_sat = np.clip(values * random_factors[1], 0, 255).astype(dtype)
        lut_val = np.clip(values * random_factors[2], 0, 255).astype(dtype)

        image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    def paper_color_jitter(self, image):
        image = image.astype(np.float32)
        image += np.random.uniform(-32, 32)
        image *= np.random.uniform(0.5, 1.5)
        return np.clip(image, 0, 255).astype(np.uint8)

    def paper_facadewhu_data(self, image, mask, input_shape, random):
        height, width = input_shape
        image_width, image_height = image.size

        if not random:
            return (
                image.resize((width, height), Image.BICUBIC),
                mask.resize((width, height), Image.NEAREST),
            )

        if image_width < width or image_height < height:
            scale = max(width / image_width, height / image_height)
            resized = (int(round(image_width * scale)), int(round(image_height * scale)))
            image = image.resize(resized, Image.BICUBIC)
            mask = mask.resize(resized, Image.NEAREST)
            image_width, image_height = resized

        left = int(self.rand(0, image_width - width + 1))
        top = int(self.rand(0, image_height - height + 1))
        box = (left, top, left + width, top + height)
        image = image.crop(box)
        mask = mask.crop(box)

        if self.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return Image.fromarray(self.paper_color_jitter(np.asarray(image))), mask

    def get_random_data(self, image, mask, input_shape, random=True):
        image = cvtColor(image)
        mask = Image.fromarray(np.array(mask))
        if self.augmentation_profile == "paper_facadewhu":
            return self.paper_facadewhu_data(image, mask, input_shape, random)

        height, width = input_shape

        image = image.resize((width, height), Image.BICUBIC)
        mask = mask.resize((width, height), Image.NEAREST)

        if not random:
            return image, mask

        if self.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        image = self.photometric_distort(np.array(image))
        return image, mask


def deeplab_dataset_collate(batch):
    images = []
    masks = []
    one_hot_masks = []
    for image, mask, one_hot_mask in batch:
        images.append(image)
        masks.append(mask)
        one_hot_masks.append(one_hot_mask)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    masks = torch.from_numpy(np.array(masks)).long()
    one_hot_masks = torch.from_numpy(np.array(one_hot_masks)).type(torch.FloatTensor)
    return images, masks, one_hot_masks
