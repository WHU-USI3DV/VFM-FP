"""VFM-FP DeepLabv3+ network definition."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2

# Cross-attention block used by the active VFM-FP fusion model.
from nets.my_attention import Attention_cross
from utils.fea_upscale import DINOFeatureProjection


VCFS_ARCHITECTURE_VERSION = "registered_dino_projection_v1"
DINOV2_REPO = "facebookresearch/dinov2:b194f00db6136677fc8a4cc2ef2168f7699dfba2"
DINOV2_MODEL = "dinov2_vits14"
DINOV2_CHANNELS = 384
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)

# DINOv2 is a frozen external extractor and is intentionally omitted from
# VCFS checkpoints. The pinned model is loaded once per process.
_DINOV2_VITS14 = None


def _get_dinov2_vits14(device):
    global _DINOV2_VITS14
    if _DINOV2_VITS14 is None:
        _DINOV2_VITS14 = torch.hub.load(
            DINOV2_REPO,
            DINOV2_MODEL,
            skip_validation=True,
        )
        _DINOV2_VITS14.requires_grad_(False)
    _DINOV2_VITS14.to(device)
    _DINOV2_VITS14.eval()
    return _DINOV2_VITS14


def _normalize_for_dinov2(image):
    mean = image.new_tensor(DINOV2_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(DINOV2_STD).view(1, 3, 1, 1)
    return (image - mean) / std

class MobileNetV2(nn.Module):
    """MobileNetV2 backbone with multi-scale DINOv2 cross-attention fusion."""

    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        model           = mobilenetv2(pretrained)
        self.features   = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   =[2, 4, 7, 14]

        self.Fusion_CrossAttention = True 
        
        self.dino_layers =[2, 5, 8, 11]

        # These layers must be persistent: creating Conv/BN inside forward
        # made the old DINO projection random on every batch.
        self.dino_projections = nn.ModuleList(
            DINOFeatureProjection(DINOV2_CHANNELS, channels)
            for channels in (24, 64, 96, 160)
        )

        # Per-stage learnable gates. Zero initialization makes the fused branch
        # start from an identity-like residual path before learning VFM cues.
        self.alpha1 = nn.Parameter(torch.zeros(1))
        self.alpha2 = nn.Parameter(torch.zeros(1))
        self.alpha3 = nn.Parameter(torch.zeros(1))
        self.alpha4 = nn.Parameter(torch.zeros(1))
        
        self.A1 = Attention_cross(24)
        self.A2 = Attention_cross(64)
        self.A3 = Attention_cross(96)
        self.A4 = Attention_cross(160)

        # Configure dilated convolutions to match the DeepLab output stride.
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
        
        new_H = H // 14 * 14
        new_W = W // 14 * 14
        
        token_hw = (new_H // 14, new_W // 14)
        with torch.no_grad():
            upsampled_tensor = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
            dino_input = _normalize_for_dinov2(upsampled_tensor)
            dino = _get_dinov2_vits14(x.device).get_intermediate_layers(
                dino_input,
                self.dino_layers,
            )

        if self.Fusion_CrossAttention:
            x_2 = self.features[:3](x)
            D0 = self.dino_projections[0](dino[0], x_2, token_hw=token_hw)
            D0_aligned = D0 * self.alpha1 
            x_2 = self.A1(x_2, D0_aligned)  
            
            low_level_features = self.features[3:4](x_2)
            
            x_7 = self.features[4:8](low_level_features)
            D1 = self.dino_projections[1](dino[1], x_7, token_hw=token_hw)
            D1_aligned = D1 * self.alpha2
            x_7 = self.A2(x_7, D1_aligned)  
            
            x_11 = self.features[8:12](x_7)
            D2 = self.dino_projections[2](dino[2], x_11, token_hw=token_hw)
            D2_aligned = D2 * self.alpha3
            x_11 = self.A3(x_11, D2_aligned)  
            
            x_14 = self.features[12:15](x_11)
            D3 = self.dino_projections[3](dino[3], x_14, token_hw=token_hw)
            D3_aligned = D3 * self.alpha4
            x_14 = self.A4(x_14, D3_aligned)  
            
            x = self.features[15:](x_14)

        else:
            low_level_features = self.features[:4](x)
            x = self.features[4:](low_level_features)

        return low_level_features, x 


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling used by DeepLabv3+."""

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

class DeepLab(nn.Module):
    """DeepLabv3+ segmentation head wrapped around the selected backbone."""

    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        
        if backbone == "xception":
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
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



