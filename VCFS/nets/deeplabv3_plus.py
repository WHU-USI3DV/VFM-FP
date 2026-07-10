"""Active VFM-FP DeepLabv3+ network definition.

Historical commented experiment variants were removed from the public release to
keep the implementation readable. The active DINO-fusion model below is kept
unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2

# Cross-attention block used by the active VFM-FP fusion model.
from nets.my_attention import Attention_cross

import torchvision.transforms as T
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.GaussianBlur(9, sigma=(0.1, 2.0)),
    T.Resize((36 * 14, 36 * 14)),
    T.CenterCrop((36 * 14, 36 * 14)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
 
from utils.fea_upscale import UpScale

# Lazy-load DINOv2 so importing this module stays lightweight.
_DINOV2_VITB14 = None

def _get_dinov2_vitb14():
    global _DINOV2_VITB14
    if _DINOV2_VITB14 is None:
        _DINOV2_VITB14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
    return _DINOV2_VITB14

def CONV(x, in_channels, out_channels):
    conv2_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True).cuda()
    y = conv2_layer(x)
    return y

# -------------------------------------------------------------------#
#   MobileNetV2 主干网络 (交叉注意力 + 前置 Alpha 尺度对齐版)
# -------------------------------------------------------------------#
class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        model           = mobilenetv2(pretrained)
        self.features   = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   =[2, 4, 7, 14]

        # ----------------控制开关----------------
        self.Fusion_CrossAttention = True 
        
        self.dino_layers =[2, 5, 8, 11]
        # self.dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()

        # ----------------核心创新点：独立的可学习尺度对齐参数----------------
        # 初始化为 0 (ones)
        self.alpha1 = nn.Parameter(torch.zeros(1))
        self.alpha2 = nn.Parameter(torch.zeros(1))
        self.alpha3 = nn.Parameter(torch.zeros(1))
        self.alpha4 = nn.Parameter(torch.zeros(1))
        
        # ----------------初始化四个融合层的交叉注意力模块----------------
        self.A1 = Attention_cross(24)
        self.A2 = Attention_cross(64)
        self.A3 = Attention_cross(96)
        self.A4 = Attention_cross(160)

        # ----------------配置空洞卷积----------------
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        _, _, H, W = x.shape
        
# Lazy-load DINOv2 so importing this module stays lightweight.
        new_H = H // 14 * 14
        new_W = W // 14 * 14
        
        token_hw = (new_H // 14, new_W // 14)
        with torch.no_grad():
            upsampled_tensor = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
# Lazy-load DINOv2 so importing this module stays lightweight.
            dino = _get_dinov2_vitb14().get_intermediate_layers(upsampled_tensor.cuda(), self.dino_layers)

        # ========================================================
        #   交叉注意力融合 (Cross Attention Fusion) + 前置尺度对齐
        # ========================================================
        if self.Fusion_CrossAttention:
            
            # -------- 第 1 融合层 --------
            x_2 = self.features[:3](x)
            D0 = UpScale(dino[0], x_2, token_hw=token_hw)
            
            # 1. 【前置 Alpha 对齐】：初始为1，保留完整特征分布
            D0_aligned = D0 * self.alpha1 
            # 2. 【交叉注意力融合】：因为 D0_aligned 含有正常信息，注意力矩阵一切正常
            x_2 = self.A1(x_2, D0_aligned)  
            
            low_level_features = self.features[3:4](x_2) # 提取给 Decoder 使用的浅层特征
            
            # -------- 第 2 融合层 --------
            x_7 = self.features[4:8](low_level_features)
            D1 = UpScale(dino[1], x_7, token_hw=token_hw)
            
            D1_aligned = D1 * self.alpha2
            x_7 = self.A2(x_7, D1_aligned)  
            
            # -------- 第 3 融合层 --------
            x_11 = self.features[8:12](x_7)
            D2 = UpScale(dino[2], x_11, token_hw=token_hw)
            
            D2_aligned = D2 * self.alpha3
            x_11 = self.A3(x_11, D2_aligned)  
            
            # -------- 第 4 融合层 --------
            x_14 = self.features[12:15](x_11)
            D3 = UpScale(dino[3], x_14, token_hw=token_hw)
            
            D3_aligned = D3 * self.alpha4
            x_14 = self.A4(x_14, D3_aligned)  
            
            # 将最终的高层特征输出给 ASPP 模块
            x = self.features[15:](x_14)

        else:
            # 安全备份逻辑 
            low_level_features = self.features[:4](x)
            x = self.features[4:](low_level_features)

        return low_level_features, x 


#-----------------------------------------#
#   ASPP特征提取模块
#-----------------------------------------#
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),  
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),  
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),  
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),      
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result

#-----------------------------------------#
#   DeepLabV3+ 主模型
#-----------------------------------------#
class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        
        if backbone == "xception":
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
            # MobileNetV2 内部已硬编码为 前置Alpha(初始化为1) 的交叉注意力融合
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )       

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)
        
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
        return x





