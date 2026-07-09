import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input
import time

class DeeplabDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(DeeplabDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape  # 期望的输入尺寸 (H, W)
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path
        self.apply_distort_flags = []

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        for path in self.dataset_path:
            filename = name
            self.apply_distort_flags.append('syn' not in filename)

        # 读取图像和标签
        jpg = Image.open(os.path.join(self.dataset_path, "JPEGImages", name + ".jpg"))
        png = Image.open(os.path.join(self.dataset_path, "SegmentationClass", name + ".png"))

        # 数据增强
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.train)

        # 预处理和one-hot编码
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((*self.input_shape, self.num_classes + 1))

        # seg_labels = np.eye(self.num_classes)[png.reshape([-1])]
        # seg_labels = seg_labels.reshape((*self.input_shape, self.num_classes))


        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a
    
    def photometric_distort(self, image):
        # 随机亮度
        if self.rand() < 0.5:
            delta = np.random.uniform(-32, 32)
            image = np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)

        # 随机对比度
        if self.rand() < 0.2:
            alpha = np.random.uniform(0.5, 1.5)
            image = cv2.addWeighted(image, alpha, np.zeros_like(image), 0, 0)
            image = np.clip(image, 0, 255)

        # HSV空间变换
        # if self.rand() < 0.5:
        #     image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        #     h, s, v = cv2.split(image)
        #     # 饱和度
        #     s = np.clip(s * self.rand(0.5, 1.5), 0, 255).astype(np.uint8)
        #     # 色相
        #     h = np.clip(h + self.rand(-18, 18), 0, 179).astype(np.uint8)
        #     image = cv2.merge((h, s, v))
        #     image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        # # 通道交换
        # if self.rand() < 0.5:
        #     image = image[..., np.random.permutation(3)]

        hue=0.1
        sat=0.7
        val=0.3

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
        dtype = image.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        return image

    def get_random_data(self, image, label, input_shape, random=True):
        """
        图像尺寸变换方式：只使用 resize，不使用任何裁剪（crop）
        - 训练模式 (random=True)：resize + 随机翻转 + 光度扭曲
        - 验证/测试模式 (random=False)：只做 resize
        """
        # 1. 统一转为 PIL 格式并转成 RGB
        image = cvtColor(image)                    # 你的工具函数，保证是 RGB
        label = Image.fromarray(np.array(label))   # 标签转为 PIL Image

        # 2. 【核心】直接 resize 到目标尺寸（这是你想要的尺寸变化方式）
        h, w = input_shape                         # input_shape 是 (H, W)
        image = image.resize((w, h), Image.BICUBIC)      # 图像用双三次插值
        label = label.resize((w, h), Image.NEAREST)      # 标签用最近邻插值（防止标签值被破坏）

        # 3. 只有训练模式才做额外数据增强
        if random:
            # 3.1 随机水平翻转（50%概率）
            if self.rand() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)

            # 3.2 光度扭曲（颜色增强）
            if self.apply_distort_flags:           # 你原来的判断逻辑
                image = self.photometric_distort(np.array(image))

        # 4. 返回处理后的图像和标签
        return image, label
    
    #####gai

    # def get_random_data(self, image, label, input_shape, random=True):

    #     image = cvtColor(image)
    #     # fn_img=image.filename
    #     # fn_png=label.filename
        
    #     label = Image.fromarray(np.array(label))
    #     h, w = input_shape
    #     iw, ih = image.size

    #     if not random:
    #         # 测试模式增强（随机裁剪版本）
    #         # Step 1: 保持长宽比缩放
    #         ###设定resize长宽
    #         # resize_w=1024
    #         # resize_h=512
    #         # ###
    #         # #scale = min(resize_w/w, resize_h/h)
    #         # scale = min(resize_w/iw, resize_h/ih)
    #         # new_w, new_h = int(iw*scale), int(ih*scale)
    #         # #image = image.resize((new_w, new_h), Image.BICUBIC)
    #         # image = image.resize((new_w, new_h), Image.BICUBIC)
    #         # label = label.resize((new_w, new_h), Image.NEAREST)

    #         # Step 2: 随机裁剪到 input_shape（与训练模式一致）
    #         ###ori
    #         # dx = int(self.rand(0, iw - w))
    #         # dy = int(self.rand(0, ih - h))
    #         # image = image.crop((dx, dy, dx+w, dy+h))
    #         # label = label.crop((dx, dy, dx+w, dy+h))

    #         # 改为中心裁剪
    #         # dx = (iw - w) // 2
    #         # dy = (ih - h) // 2
    #         # image = image.crop((dx, dy, dx+w, dy+h))
    #         # label = label.crop((dx, dy, dx+w, dy+h))

    #         #resize_0717
    #         image = image.resize((w, h), Image.BICUBIC)
    #         label = label.resize((w, h), Image.NEAREST)


    #         return image, label

    #     # -------------------------- 训练模式增强 --------------------------
    #     ###resize_0717
    #     image = image.resize((w, h), Image.BICUBIC)
    #     label = label.resize((w, h), Image.NEAREST)

    #     ####7-13
    #     #h, w = input_shape
    #     # dx = int(self.rand(0, iw - w))
    #     # dy = int(self.rand(0, ih - h))
    #     # image = image.crop((dx, dy, dx+w, dy+h))
    #     # label = label.crop((dx, dy, dx+w, dy+h))

    #     # 2. 随机裁剪到 input_shape 尺寸 #512*512 [没有resize部分]下面的代码有误
    #     # tw, th = input_shape
    #     # dx = int(self.rand(0, iw - tw))
    #     # dy = int(self.rand(0, ih - th))
    #     # image = image.crop((dx, dy, dx+tw, dy+th))
    #     # label = label.crop((dx, dy, dx+tw, dy+th))

    #     # 3. 随机翻转
    #     if self.rand() < 0.5:
    #         image = image.transpose(Image.FLIP_LEFT_RIGHT)
    #         label = label.transpose(Image.FLIP_LEFT_RIGHT)

    #     # 4. 光度扭曲
    #     if self.apply_distort_flags:
    #         image = self.photometric_distort(np.array(image))

    #     #创建debug目录（如果不存在）
    #     # debug_dir_jpg = "debug0327/img"
    #     # debug_dir_png = "debug0327/png"
    #     # os.makedirs(debug_dir_jpg, exist_ok=True)
    #     # os.makedirs(debug_dir_png, exist_ok=True)

    #     # # 生成唯一文件名（避免覆盖）
    #     # #unique_id = str(time.time()).replace(".", "")  # 或用uuid：import uuid; unique_id = uuid.uuid4().hex

    #     # # 分割路径，取最后一部分
    #     # filename1= fn_img.split('/')[-1]  # 'syn_0827_1.jpg'
    #     # # 分割扩展名，取前半部分
    #     # name_without_ext1 = filename1.split('.')[0]  
    #     # # 分割路径，取最后一部分
    #     # filename2 = fn_png.split('/')[-1]  # 'syn_0827_1.jpg'
    #     # # 分割扩展名，取前半部分
    #     # name_without_ext2 = filename2.split('.')[0] 

    #     # # 保存处理后的图像（需转换颜色空间）
    #     # cv2.imwrite(
    #     #     os.path.join(debug_dir_jpg, f"debug_img_{name_without_ext1}.jpg"),
    #     #     cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # OpenCV默认BGR格式
    #     # )

    #     # label_np = np.array(label)  # 将 PIL.Image 转换为 NumPy 数组
    #     # label_processed = label_np.astype(np.uint8) * 255
    #     # # 保存处理后的标签（假设label是单通道灰度图）
    #     # cv2.imwrite(
    #     #     os.path.join(debug_dir_png, f"debug_label_{name_without_ext2}.png"),
    #     #     label_processed.astype(np.uint8) * 255  # 标签值通常为0/1，乘以255提高可视性
    #     #       # 将 PIL.Image 转换为 NumPy 数组
    #     #     #label_processed = label_np.astype(np.uint8) * 255
    #     # )

    #     return Image.fromarray(image), label

    

def deeplab_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels