import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
##my_loader_try
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torchvision.transforms as T


###
###DINOV2
from nets.attention import Attention, MultiheadAttention, Danet_PositAttention, Danet_ChannAttention

transform = T.Compose([
    T.GaussianBlur(9, sigma=(0.1, 2.0)),
    T.Resize((36 * 14, 36 * 14)),
    T.CenterCrop((36 * 14, 36 * 14)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

from utils.fea_upscale import UpScale
dinov2_vitb14=torch.hub.load('facebookresearch/dinov2','dinov2_vits14').cuda()


def CONV(x,in_channels,out_channels):
    conv2_layer=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True).cuda()
    y=conv2_layer(x)
    return y


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        model           = mobilenetv2(pretrained)
        self.features   = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]


        ##DINO
        self.check_QCatten=True
        self.alpha = nn.Parameter(torch.zeros(1)) 
        self.Fusion1=True

        self.dino_layers=[2,5,8,10]

        if self.check_QCatten:
             self.pam7 = Danet_PositAttention(64)
             self.pam11 = Danet_PositAttention(96)
             self.pam14 = Danet_PositAttention(160)
             self.pamx = Danet_PositAttention(320)

             self.cam7 = Danet_ChannAttention()
             self.cam11 = Danet_ChannAttention()
             self.cam14 = Danet_ChannAttention()
             self.camx = Danet_ChannAttention()

        else:
             self.attn_x=Attention(320)


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

        ###DINO
        _,_,H,W=x.shape

        new_H=H//14*14
        new_W=W//14*14

        with torch.no_grad():
            upsampled_tensor = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
            #print(upsampled_tensor)
            dino=dinov2_vitb14.get_intermediate_layers(upsampled_tensor.cuda(),self.dino_layers)

        if self.check_QCatten:
             if self.Fusion1:
                x_2=self.features[:3](x)
                D0=UpScale(dino[0],x_2)
                #D0=UpScale(features,x_2)
                fea0=D0*self.alpha
                x_2=fea0+x_2
                
                low_level_features=self.features[3:4](x_2)
                
                x_7=self.features[4:8](low_level_features)
                D1=UpScale(dino[1],x_7)
                #D1=UpScale(features,x_5)
                fea1=D1*self.alpha
                x_7=fea1+x_7
                
                x_7f=self.pam7(x_7)
                x_7s=self.cam7(x_7)
                x_7=x_7f+x_7s
                #x_5/=2
                x_11=self.features[8:12](x_7)
                D2=UpScale(dino[2],x_11)
                #D2=UpScale(features,x_10)
                fea2=D2*self.alpha
                x_11=fea2+x_11
                
                x_11f=self.pam11(x_11)
                x_11s=self.cam11(x_11)
                x_11=x_11f+x_11s
                #x_8/=2
                x_14=self.features[12:15](x_11)
                D3=UpScale(dino[3],x_14)
                #D3=UpScale(features,x_14)
                fea3=D3*self.alpha
                x_14=fea3+x_14
                x_14f=self.pam14(x_14)
                x_14s=self.cam14(x_14)
                x_14=x_14f+x_14s
                
                #x_11/=2
                x1=self.pamx(self.features[15:](x_14))
                x2=self.camx(self.features[15:](x_14))

                x=x1+x2

             else:
                low_level_features = self.features[:4](x)
                x = self.features[4:](low_level_features)
                if self.check_atten:
                    x=self.attn_x(x)

        # low_level_features = self.features[:4](x)
        # x = self.features[4:](low_level_features)
        return low_level_features, x 


#-----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
#-----------------------------------------#
class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
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
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)

		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)

	def forward(self, x):
		[b, c, row, col] = x.size()
        #-----------------------------------------#
        #   一共五个分支
        #-----------------------------------------#
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
        #-----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #-----------------------------------------#
		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
        #-----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        #-----------------------------------------#
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result

class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone=="xception":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            #----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone=="mobilenet":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            #----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        #-----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        #-----------------------------------------#
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
        #----------------------------------#
        #   浅层特征边
        #----------------------------------#
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
        #-----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        #-----------------------------------------#
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)
        
        #-----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        #-----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

