# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from nets.xception import xception
# from nets.mobilenetv2 import mobilenetv2
# from nets.attention import Attention, MultiheadAttention, Danet_PositAttention, Danet_ChannAttention

# from nets.my_attention import Attention_cross

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# import torch
 
# import torchvision.transforms as T
 
# import matplotlib.pyplot as plt
 
# from PIL import Image
 
# from sklearn.decomposition import PCA
# import cv2

# import numpy as np
# transform = T.Compose([
#     T.GaussianBlur(9, sigma=(0.1, 2.0)),
#     T.Resize((36 * 14, 36 * 14)),
#     T.CenterCrop((36 * 14, 36 * 14)),
#     T.ToTensor(),
#     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
# ])
 

# #from torchvision.models._utils import IntermediateLayerGetter
# #from nets.resnet import ResNet

# from utils.fea_upscale import UpScale
# dinov2_vitb14=torch.hub.load('facebookresearch/dinov2','dinov2_vits14').cuda()


# def CONV(x,in_channels,out_channels):
#     conv2_layer=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True).cuda()
#     y=conv2_layer(x)
#     return y

# class MobileNetV2(nn.Module):
#     def __init__(self, downsample_factor=8, pretrained=True, 
#                  attention=False, MCattention=False, Mattention=False, QCattention=False, 
#                  fusion=False):
#         super(MobileNetV2, self).__init__()
#         from functools import partial
        
#         model           = mobilenetv2(pretrained)
#         self.features   = model.features[:-1]

#         self.total_idx  = len(self.features)
#         self.down_idx   = [2, 4, 7, 14]


#         self.check_atten=attention
#         self.check_MCatten=False#多层注意力机制
#         self.check_Matten=False#多头注意力机制
#         self.check_QCatten=True#空间+通道注意力机制##True

#         self.Fusion1=True#True
#         self.Fusion2=False
#         self.Fusion3=False
#         self.dino_layers=[2,5,8,11]
#         self.alpha = nn.Parameter(torch.zeros(1)) 
#         self.A1=Attention_cross(24)
#         self.A2=Attention_cross(64)#64,在dino_danet_fusion3更改
#         self.A3=Attention_cross(96)#96
#         self.A4=Attention_cross(160)
        

#         if self.check_Matten:
#              #self.attn_2=MultiheadAttention(24)
#              self.attn_7=MultiheadAttention(64)
#              self.attn_11=MultiheadAttention(96)
#              self.attn_14=MultiheadAttention(160)
#              self.attn_x=MultiheadAttention(320)
        
#         elif self.check_MCatten:
#              self.attn_2=Attention(24)
#              self.attn_7=Attention(64)
#              self.attn_11=Attention(96)
#              self.attn_14=Attention(160)
#              self.attn_x=Attention(320)

#         elif self.check_QCatten:
#              self.pam7 = Danet_PositAttention(64)
#              self.pam11 = Danet_PositAttention(96)
#              self.pam14 = Danet_PositAttention(160)
#              self.pamx = Danet_PositAttention(320)

#              self.cam7 = Danet_ChannAttention()
#              self.cam11 = Danet_ChannAttention()
#              self.cam14 = Danet_ChannAttention()
#              self.camx = Danet_ChannAttention()

#         else:
#              self.attn_x=Attention(320)

#         if downsample_factor == 8:
#             for i in range(self.down_idx[-2], self.down_idx[-1]):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=2)
#                 )
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=4)
#                 )
#         elif downsample_factor == 16:
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=2)#/media/tan/04774106-a452-4b88-807f-629347ba23d5/tan/TAN/deeplabv3-plus-pytorch-main/nets
#                 )
        
#     def _nostride_dilate(self, m, dilate):
#         classname = m.__class__.__name__
#         if classname.find('Conv') != -1:
#             if m.stride == (2, 2):
#                 m.stride = (1, 1)
#                 if m.kernel_size == (3, 3):
#                     m.dilation = (dilate//2, dilate//2)
#                     m.padding = (dilate//2, dilate//2)
#             else:
#                 if m.kernel_size == (3, 3):
#                     m.dilation = (dilate, dilate)
#                     m.padding = (dilate, dilate)

#     def forward(self, x):#在这里才实际完成了网络操作
        
#         _,_,H,W=x.shape
        
#         new_H=H//14*14
#         new_W=W//14*14
        
#         nh=36
#         nw=72##73*36
        
#         with torch.no_grad():
#             upsampled_tensor = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
#             #print(upsampled_tensor)
#             dino=dinov2_vitb14.get_intermediate_layers(upsampled_tensor.cuda(),self.dino_layers)
        
#             # features_dict = dinov2_vitb14.forward_features(upsampled_tensor.cuda())
#             # features = features_dict['x_norm_patchtokens']
        
#         """         di=self.features[0,:,:]
#             di2=di.transpose(0,1)
#             di3 = di2.reshape(-1, nh , nw).cpu()
#             #features = features.reshape(4 * nh * nw, 384).cpu()
#             embeddings=di3
#             as_rgb=True
#             # pca = PCA(n_components=3)
#             # pca.fit(features)
#             # pca_features = pca.transform(features)
#             pca = PCA(n_components=3)
#             embed_dim = embeddings.shape[0]
#             shape = embeddings.shape[1:]
#             embed_flat = embeddings.reshape(embed_dim, -1).T
#             embed_flat = pca.fit_transform(embed_flat).T
#             embed_flat = embed_flat.reshape((3,) + shape)
#             if as_rgb:
#                 embed_flat = 255 * (embed_flat - embed_flat.min()) / np.ptp(embed_flat)
#                 embed_flat = np.transpose(embed_flat, (1,2,0))
#                 embed_flat = embed_flat.astype('uint8')
#             cv2.imwrite('feature5.png', embed_flat) """
        
#         """ with torch.no_grad():
#             features_dict = dinov2_vitb14.get_intermediate_layers(upsampled_tensor,self.dino_layers)
#             features_dict = dinov2_vitb14.forward_features(upsampled_tensor)
#             features = features_dict['x_norm_patchtokens'] """

#         if self.check_MCatten:
#             if self.Fusion1:
#                 x_2=self.features[:3](x)
#                 D0=UpScale(dino[0],x_2)
#                 fea0=D0*self.alpha
#                 x_2=fea0+x_2
#                 R_2=self.attn_2(x_2)
                
#                 low_level_features=self.features[3:4](R_2)
                
#                 x_7=self.features[4:8](low_level_features)
#                 D1=UpScale(dino[1],x_7)
#                 fea1=D1*self.alpha
#                 x_7=fea1+x_7
#                 R_7=self.attn_7(x_7)
                
#                 #x_5/=2
#                 x_11=self.features[8:12](R_7)
#                 D2=UpScale(dino[2],x_11)
#                 fea2=D2*self.alpha
#                 x_11=fea2+x_11
#                 R_11=self.attn_11(x_11)
#                 #x_8/=2
                
#                 x_14=self.features[12:15](R_11)
#                 D3=UpScale(dino[3],x_14)
#                 fea3=D3*self.alpha
#                 x_14=fea3+x_14
#                 R_14=self.attn_14(x_14)
#                 #x_11/=2
                
#                 x=self.attn_x(self.features[15:](R_14))
                
#             elif self.Fusion3:
#                 x_2=self.features[:3](x)
#                 D0=UpScale(dino[0],x_2)
#                 R_2=self.A1(x_2,D0)
#                 #r2=x_2+R_2*self.alpha
#                 ##x_2=fea0+x_2
                
#                 low_level_features=self.features[3:4](R_2)
                
#                 x_7=self.features[4:8](low_level_features)
#                 D1=UpScale(dino[1],x_7)
#                 R_7=self.attn_7(self.A2(x_7,D1))
                
#                 #x_5=fea1+x_5
#                 #x_5/=2
#                 x_11=self.features[8:12](R_7)
#                 D2=UpScale(dino[2],x_11)
#                 R_11=self.attn_11(self.A3(x_11,D2))
                
#                 #r10=x_10+R_10*self.alpha
#                 #x_8/=2
#                 x_14=self.features[12:15](R_11)
#                 D3=UpScale(dino[3],x_14)
#                 R_14=self.attn_14(self.A4(x_14,D3))
                
#                 x=self.attn_x(self.features[15:](R_14))
                
#             else:
#             #x=self.attn_x(x)
#                 low_level_features=self.features[:4](x)
#                 x_7=self.attn_7(self.features[4:8](low_level_features))#分别在第7，11，14层后添加注意力机制
#                 x_11=self.attn_11(self.features[8:12](x_7))
#                 x_14=self.attn_14(self.features[12:15](x_11))
#                 x=self.attn_x(self.features[15:](x_14))
            
        
#         elif self.check_Matten:
#             if self.Fusion1:
#                 x_2=self.features[:3](x)
#                 D0=UpScale(dino[0],x_2)
#                 #D0=UpScale(features,x_2)
#                 fea0=D0*self.alpha
#                 #fea0=D0*2
#                 x_2=fea0+x_2
#                 #R_2=self.attn_2(x_2)
                
#                 low_level_features=self.features[3:4](x_2)
                
#                 x_7=self.features[4:8](low_level_features)
#                 D1=UpScale(dino[1],x_7)
#                 #D1=UpScale(features,x_7)
#                 fea1=D1*self.alpha
#                 #fea1=D1*2
#                 x_7=fea1+x_7
#                 R_7=self.attn_7(x_7)
                
#                 #x_5/=2
#                 x_11=self.features[8:12](R_7)
#                 D2=UpScale(dino[2],x_11)
#                 #D2=UpScale(features,x_11)
#                 fea2=D2*self.alpha
#                 #fea2=D2*2
#                 x_11=fea2+x_11
#                 R_11=self.attn_11(x_11)
#                 #x_8/=2
                
#                 x_14=self.features[12:15](R_11)
#                 D3=UpScale(dino[3],x_14)
#                 #D3=UpScale(features,x_14)
#                 fea3=D3*self.alpha
#                 #fea3=D3*2
#                 x_14=fea3+x_14
#                 R_14=self.attn_14(x_14)
#                 #x_11/=2
                
#                 x=self.attn_x(self.features[15:](R_14))
                
#             elif self.Fusion3:
#                 x_2=self.features[:3](x)
#                 D0=UpScale(dino[0],x_2)
#                 #D0=UpScale(features,x_2)
#                 R_2=self.A1(x_2,D0)
#                 #r2=x_2+R_2*self.alpha
#                 ##x_2=fea0+x_2
                
#                 low_level_features=self.features[3:4](R_2)
                
#                 x_7=self.features[4:8](low_level_features)
#                 D1=UpScale(dino[1],x_7)
#                 #D1=UpScale(features,x_7)
#                 R_7=self.attn_7(self.A2(x_7,D1))
                
#                 #x_5=fea1+x_5
#                 #x_5/=2
#                 x_11=self.features[8:12](R_7)
#                 D2=UpScale(dino[2],x_11)
#                 #D2=UpScale(features,x_11)
#                 R_11=self.attn_11(self.A3(x_11,D2))
                
#                 #r10=x_10+R_10*self.alpha
#                 #x_8/=2
#                 x_14=self.features[12:15](R_11)
#                 D3=UpScale(dino[3],x_14)
#                 #D3=UpScale(features,x_14)
#                 R_14=self.attn_14(self.A4(x_14,D3))
                
#                 x=self.attn_x(self.features[15:](R_14))
                
#             else:
#                 low_level_features=self.features[:4](x)
#                 x_7=self.attn_7(self.features[4:8](low_level_features))#分别在第7，11，14层后添加注意力机制
#                 x_11=self.attn_11(self.features[8:12](x_7))
#                 x_14=self.attn_14(self.features[12:15](x_11))
#                 x=self.attn_x(self.features[15:](x_14))

#         elif self.check_QCatten:
            
#             if self.Fusion1:
#                 x_2=self.features[:3](x)
#                 D0=UpScale(dino[0],x_2)
#                 #D0=UpScale(features,x_2)
#                 fea0=D0*self.alpha
#                 x_2=fea0+x_2
                
#                 low_level_features=self.features[3:4](x_2)
                
#                 x_7=self.features[4:8](low_level_features)
#                 D1=UpScale(dino[1],x_7)
#                 #D1=UpScale(features,x_5)
#                 fea1=D1*self.alpha
#                 x_7=fea1+x_7
                
#                 x_7f=self.pam7(x_7)
#                 x_7s=self.cam7(x_7)
#                 x_7=x_7f+x_7s
#                 #x_5/=2
#                 x_11=self.features[8:12](x_7)
#                 D2=UpScale(dino[2],x_11)
#                 #D2=UpScale(features,x_10)
#                 fea2=D2*self.alpha
#                 x_11=fea2+x_11
                
#                 x_11f=self.pam11(x_11)
#                 x_11s=self.cam11(x_11)
#                 x_11=x_11f+x_11s
#                 #x_8/=2
#                 x_14=self.features[12:15](x_11)
#                 D3=UpScale(dino[3],x_14)
#                 #D3=UpScale(features,x_14)
#                 fea3=D3*self.alpha
#                 x_14=fea3+x_14
#                 x_14f=self.pam14(x_14)
#                 x_14s=self.cam14(x_14)
#                 x_14=x_14f+x_14s
                
#                 #x_11/=2
#                 x1=self.pamx(self.features[15:](x_14))
#                 x2=self.camx(self.features[15:](x_14))

#                 x=x1+x2
                
#             elif self.Fusion2:
#                 x_2=self.features[:3](x)
#                 D0=UpScale(dino[0],x_2)
#                 c2 = torch.cat((x_2, D0), dim=1)
#                 r2 = CONV(c2,in_channels=c2.shape[1], 
#                        out_channels=x_2.shape[1])
#                 r2=x_2+r2*self.alpha
#                 ##x_2=fea0+x_2
                
#                 low_level_features=self.features[3:4](r2)
                
#                 x_7=self.features[4:8](low_level_features)
#                 D1=UpScale(dino[1],x_7)
#                 c5=torch.cat((x_7,D1),dim=1)
#                 r5 = CONV(c5,in_channels=c5.shape[1], 
#                        out_channels=x_7.shape[1])
#                 r7=x_7+r5*self.alpha
#                 #x_5=fea1+x_5
#                 #x_5/=2
#                 x_11=self.features[8:12](r7)
#                 D2=UpScale(dino[2],x_11)
#                 c11 = torch.cat((x_11, D2), dim=1)
#                 r11 = CONV(c11,in_channels=c11.shape[1], 
#                        out_channels=x_11.shape[1])
#                 r11=x_11+r11*self.alpha
#                 #x_8/=2
#                 x_14=self.features[12:15](r11)
#                 D3=UpScale(dino[3],x_14)
#                 c14 = torch.cat((x_14, D3), dim=1)
#                 r14 = CONV(c14,in_channels=c14.shape[1], 
#                        out_channels=x_14.shape[1])
#                 r14=x_14+r14*self.alpha
#                 #x_11/=2
#                 x1=self.pamx(self.features[15:](r14))
#                 x2=self.camx(self.features[15:](r14))
#                 x=x1+x2

#                 # x_2=self.features[:3](x)
#                 # D0=UpScale(dino[0],x_2)
#                 # c2 = torch.cat((x_2, D0), dim=1)
#                 # r2 = CONV(c2,in_channels=c2.shape[1], 
#                 #        out_channels=x_2.shape[1])
#                 # r2=x_2+r2
#                 # ##x_2=fea0+x_2
                
#                 # low_level_features=self.features[3:4](r2)
                
#                 # x_7=self.features[4:8](low_level_features)
#                 # D1=UpScale(dino[1],x_7)
#                 # c5=torch.cat((x_7,D1),dim=1)
#                 # r5 = CONV(c5,in_channels=c5.shape[1], 
#                 #        out_channels=x_7.shape[1])
#                 # r7=x_7+r5
#                 # #x_5=fea1+x_5
#                 # #x_5/=2
#                 # x_11=self.features[8:12](r7)
#                 # D2=UpScale(dino[2],x_11)
#                 # c11 = torch.cat((x_11, D2), dim=1)
#                 # r11 = CONV(c11,in_channels=c11.shape[1], 
#                 #        out_channels=x_11.shape[1])
#                 # r11=x_11+r11
#                 # #x_8/=2
#                 # x_14=self.features[12:15](r11)
#                 # D3=UpScale(dino[3],x_14)
#                 # c14 = torch.cat((x_14, D3), dim=1)
#                 # r14 = CONV(c14,in_channels=c14.shape[1], 
#                 #        out_channels=x_14.shape[1])
#                 # r14=x_14+r14
#                 # #x_11/=2
#                 # x1=self.pamx(self.features[15:](r14))
#                 # x2=self.camx(self.features[15:](r14))
#                 # x=x1+x2
                
#             elif self.Fusion3:
#                 x_2=self.features[:3](x)
#                 D0=UpScale(dino[0],x_2)
#                 R_2=self.A1(x_2,D0)
#                 #r2=x_2+R_2*self.alpha
#                 ##x_2=fea0+x_2
                
#                 low_level_features=self.features[3:4](R_2)
                
#                 x_7=self.features[4:8](low_level_features)
#                 D1=UpScale(dino[1],x_7)
#                 R_7=self.A2(x_7,D1)
                
#                 #x_7f=self.pam7(R_7)
#                 #x_7s=self.cam7(R_7)
#                 #R_7=x_7f+x_7s
#                 #r5=x_5+R_5*self.alpha
#                 #x_5=fea1+x_5
#                 #x_5/=2
#                 x_11=self.features[8:12](R_7)
#                 D2=UpScale(dino[2],x_11)
#                 R_11=self.A3(x_11,D2)
                
#                 # x_11f=self.pam11(R_11)
#                 # x_11s=self.cam11(R_11)
#                 # R_11=x_11f+x_11s
#                 #r10=x_10+R_10*self.alpha
#                 #x_8/=2
#                 x_14=self.features[12:15](R_11)
#                 D3=UpScale(dino[3],x_14)
#                 R_14=self.A4(x_14,D3)
                
#                 # x_14f=self.pam14(R_14)
#                 # x_14s=self.cam14(R_14)
#                 # R_14=x_14f+x_14s
#                 #r14=x_14+R_14*self.alpha
#                 #x_11/=2
#                 x1=self.pamx(self.features[15:](R_14))
#                 x2=self.camx(self.features[15:](R_14))
#                 x=x1+x2
#                 #x=self.features[15:](R_14)
                
#             else:
#                 low_level_features=self.features[:4](x)
#                 x1=self.pamx(self.features[4:](low_level_features))
#                 x2=self.camx(self.features[4:](low_level_features))
#                 x=x1+x2
                

#             """ x_7f=self.pam7(self.features[4:8](low_level_features))
#             x_7s=self.cam7(self.features[4:8](low_level_features))
#             x_7=x_7f+x_7s

#             x_11f=self.pam11(self.features[8:12](x_7))
#             x_11s=self.cam11(self.features[8:12](x_7))
#             x_11=x_11f+x_11s

#             x_14f=self.pam14(self.features[12:15](x_11))
#             x_14s=self.cam14(self.features[12:15](x_11))
#             x_14=x_14f+x_14s

#             xf=self.pamx(self.features[15:](x_14))
#             xs=self.camx(self.features[15:](x_14)) """

                


#         else:
#             if self.Fusion1:
#                 # # print('##########Fusion1##########')
#                 # x_2=self.features[:3](x)
#                 # D0=UpScale(dino[0],x_2)
#                 # #D0=UpScale(features,x_2)
#                 # fea0=D0
#                 # x_2=fea0+x_2
                
#                 # low_level_features=self.features[3:4](x_2)
                
#                 # x_7=self.features[4:8](low_level_features)
#                 # D1=UpScale(dino[1],x_7)
#                 # #D1=UpScale(features,x_7)
#                 # fea1=D1
#                 # x_7=fea1+x_7
#                 # #x_5/=2
#                 # x_11=self.features[8:12](x_7)
#                 # D2=UpScale(dino[2],x_11)
#                 # #D2=UpScale(features,x_11)
#                 # fea2=D2
#                 # x_11=fea2+x_11
#                 # #x_8/=2
#                 # x_14=self.features[12:15](x_11)
#                 # D3=UpScale(dino[3],x_14)
#                 # #D3=UpScale(features,x_14)
#                 # fea3=D3
#                 # x_14=fea3+x_14
#                 # #low_level_features = self.features[:4](x)
#                 # x = self.features[15:](x_14)

#                 x_2=self.features[:3](x)
#                 D0=UpScale(dino[0],x_2)
#                 #D0=UpScale(features,x_2)
#                 fea0=D0*self.alpha
#                 x_2=fea0+x_2
                
#                 low_level_features=self.features[3:4](x_2)
                
#                 x_7=self.features[4:8](low_level_features)
#                 D1=UpScale(dino[1],x_7)
#                 #D1=UpScale(features,x_5)
#                 fea1=D1*self.alpha
#                 x_7=fea1+x_7
                
                
#                 #x_5/=2
#                 x_11=self.features[8:12](x_7)
#                 D2=UpScale(dino[2],x_11)
#                 #D2=UpScale(features,x_10)
#                 fea2=D2*self.alpha
#                 x_11=fea2+x_11
                
                
#                 #x_8/=2
#                 x_14=self.features[12:15](x_11)
#                 D3=UpScale(dino[3],x_14)
#                 #D3=UpScale(features,x_14)
#                 fea3=D3*self.alpha
#                 x_14=fea3+x_14
#                 #x_11/=2
                
#                 x=self.features[15:](x_14)

#             elif self.Fusion2:
#                 x_2=self.features[:3](x)
#                 D0=UpScale(dino[0],x_2)
#                 c2 = torch.cat((x_2, D0), dim=1)
#                 r2 = CONV(c2,in_channels=c2.shape[1], 
#                        out_channels=x_2.shape[1])
#                 r2=x_2+r2*self.alpha
#                 ##x_2=fea0+x_2
                
#                 low_level_features=self.features[3:4](r2)
                
#                 x_7=self.features[4:8](low_level_features)
#                 D1=UpScale(dino[1],x_7)
#                 c5=torch.cat((x_7,D1),dim=1)
#                 r5 = CONV(c5,in_channels=c5.shape[1], 
#                        out_channels=x_7.shape[1])
#                 r7=x_7+r5*self.alpha
#                 #x_5=fea1+x_5
#                 #x_5/=2
#                 x_11=self.features[8:12](r7)
#                 D2=UpScale(dino[2],x_11)
#                 c11 = torch.cat((x_11, D2), dim=1)
#                 r11 = CONV(c11,in_channels=c11.shape[1], 
#                        out_channels=x_11.shape[1])
#                 r11=x_11+r11*self.alpha
#                 #x_8/=2
#                 x_14=self.features[12:15](r11)
#                 D3=UpScale(dino[3],x_14)
#                 c14 = torch.cat((x_14, D3), dim=1)
#                 r14 = CONV(c14,in_channels=c14.shape[1], 
#                        out_channels=x_14.shape[1])
#                 r14=x_14+r14*self.alpha
#                 #x_11/=2
#                 # x1=self.pamx(self.features[15:](r14))
#                 # x2=self.camx(self.features[15:](r14))
#                 x=self.features[15:](r14)


#                 # print("####################FUSION2#####################")
#                 # x_2=self.features[:3](x)
#                 # D0=UpScale(dino[0],x_2)
#                 # c2 = torch.cat((x_2, D0), dim=1)
#                 # r2 = CONV(c2,in_channels=c2.shape[1], 
#                 #        out_channels=x_2.shape[1])
#                 # r2=x_2+r2
#                 # ##x_2=fea0+x_2
                
#                 # low_level_features=self.features[3:4](r2)
                
#                 # x_7=self.features[4:8](low_level_features)
#                 # D1=UpScale(dino[1],x_7)
#                 # c5=torch.cat((x_7,D1),dim=1)
#                 # r5 = CONV(c5,in_channels=c5.shape[1], 
#                 #        out_channels=x_7.shape[1])
#                 # r7=x_7+r5
#                 # #x_5=fea1+x_5
#                 # #x_5/=2
#                 # x_11=self.features[8:12](r7)
#                 # D2=UpScale(dino[2],x_11)
#                 # c11 = torch.cat((x_11, D2), dim=1)
#                 # r11 = CONV(c11,in_channels=c11.shape[1], 
#                 #        out_channels=x_11.shape[1])
#                 # r11=x_11+r11
#                 # #x_8/=2
#                 # x_14=self.features[12:15](r11)
#                 # D3=UpScale(dino[3],x_14)
#                 # c14 = torch.cat((x_14, D3), dim=1)
#                 # r14 = CONV(c14,in_channels=c14.shape[1], 
#                 #        out_channels=x_14.shape[1])
#                 # r14=x_14+r14
#                 # #x_11/=2
#                 # x=self.features[15:](r14)
                
#             elif self.Fusion3:
#                 x_2=self.features[:3](x)
#                 D0=UpScale(dino[0],x_2)
#                 #D0=UpScale(features,x_2)
#                 R_2=self.A1(x_2,D0)
#                 #r2=x_2+R_2*self.alpha
#                 ##x_2=fea0+x_2
                
#                 low_level_features=self.features[3:4](R_2)
                
#                 x_7=self.features[4:8](low_level_features)
#                 D1=UpScale(dino[1],x_7)
#                 #D1=UpScale(features,x_7)
#                 R_7=self.A2(x_7,D1)
#                 #r5=x_5+R_5*self.alpha
#                 #x_5=fea1+x_5
#                 #x_5/=2
#                 x_11=self.features[8:12](R_7)
#                 D2=UpScale(dino[2],x_11)
#                 #D2=UpScale(features,x_11)
#                 R_11=self.A3(x_11,D2)
#                 #r10=x_10+R_10*self.alpha
#                 #x_8/=2
#                 x_14=self.features[12:15](R_11)
#                 D3=UpScale(dino[3],x_14)
#                 #D3=UpScale(features,x_14)
#                 R_14=self.A4(x_14,D3)
#                 #low_level_features = self.features[:4](x)
#                 x = self.features[15:](R_14)
                
#             else:
#                 low_level_features = self.features[:4](x)
#                 x = self.features[4:](low_level_features)
#                 if self.check_atten:
#                     x=self.attn_x(x)


#         return low_level_features, x 


# #-----------------------------------------#
# #   ASPP特征提取模块
# #   利用不同膨胀率的膨胀卷积进行特征提取
# #-----------------------------------------#
# class ASPP(nn.Module):
# 	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
# 		super(ASPP, self).__init__()
# 		self.branch1 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),
# 		)
# 		self.branch2 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch3 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch4 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
# 		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
# 		self.branch5_relu = nn.ReLU(inplace=True)

# 		self.conv_cat = nn.Sequential(
# 				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),		
# 		)

# 	def forward(self, x):
# 		[b, c, row, col] = x.size()
#         #-----------------------------------------#
#         #   一共五个分支
#         #-----------------------------------------#
# 		conv1x1 = self.branch1(x)
# 		conv3x3_1 = self.branch2(x)
# 		conv3x3_2 = self.branch3(x)
# 		conv3x3_3 = self.branch4(x)
#         #-----------------------------------------#
#         #   第五个分支，全局平均池化+卷积
#         #-----------------------------------------#
# 		global_feature = torch.mean(x,2,True)
# 		global_feature = torch.mean(global_feature,3,True)
# 		global_feature = self.branch5_conv(global_feature)
# 		global_feature = self.branch5_bn(global_feature)
# 		global_feature = self.branch5_relu(global_feature)
# 		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
#         #-----------------------------------------#
#         #   将五个分支的内容堆叠起来
#         #   然后1x1卷积整合特征。
#         #-----------------------------------------#
# 		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
# 		result = self.conv_cat(feature_cat)
# 		return result

# class DeepLab(nn.Module):
#     def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16,attention=False,M_attention=False,QC_attention=True):
#         self.att=attention
#         self.Matt=M_attention
#         self.PCatt=QC_attention
#         self.fusion=False
#         super(DeepLab, self).__init__()
#         if backbone=="xception":
#             #----------------------------------#
#             #   获得两个特征层
#             #   浅层特征    [128,128,256]
#             #   主干部分    [30,30,2048]
#             #----------------------------------#
#             self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
#             in_channels = 2048
#             low_level_channels = 256
#         elif backbone=="mobilenet":
#             #----------------------------------#
#             #   获得两个特征层
#             #   浅层特征    [128,128,24]
#             #   主干部分    [30,30,320]
#             #----------------------------------#
#             self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained,attention=self.att,Mattention=self.Matt,QCattention=self.PCatt,fusion=self.fusion)
#             in_channels = 320
#             low_level_channels = 24

#         else:
#             raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

#         ##更改backbone为Resnet50
#         """ elif backbone=="resnet":
#              self.backbone= ResNet(downsample_factor=downsample_factor, pretrained=pretrained)
#              in_channels = 320
#              low_level_channels = 24 """
        

        

#         #-----------------------------------------#
#         #   ASPP特征提取模块
#         #   利用不同膨胀率的膨胀卷积进行特征提取
#         #-----------------------------------------#
#         self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
#         #----------------------------------#
#         #   浅层特征边
#         #----------------------------------#
#         self.shortcut_conv = nn.Sequential(
#             nn.Conv2d(low_level_channels, 48, 1),
#             nn.BatchNorm2d(48),
#             nn.ReLU(inplace=True)
#         )		

#         self.cat_conv = nn.Sequential(
#             nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),

#             nn.Conv2d(256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),

#             nn.Dropout(0.1),
#         )
#         self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

#     def forward(self, x):
#         H, W = x.size(2), x.size(3)
#         #-----------------------------------------#
#         #   获得两个特征层
#         #   low_level_features: 浅层特征-进行卷积处理
#         #   x : 主干部分-利用ASPP结构进行加强特征提取
#         #-----------------------------------------#
#         low_level_features, x = self.backbone(x)#得到浅层特征low和深层特征x
#         #-----------------------------------------#
#         x = self.aspp(x)
#         low_level_features = self.shortcut_conv(low_level_features)
        
#         #-----------------------------------------#
#         #   将加强特征边上采样
#         #   与浅层特征堆叠后利用卷积进行特征提取
#         #-----------------------------------------#
#         x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
#         x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
#         x = self.cls_conv(x)
#         x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
#         return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from nets.xception import xception
# from nets.mobilenetv2 import mobilenetv2
# from nets.attention import Attention, MultiheadAttention, Danet_PositAttention, Danet_ChannAttention

# from nets.my_attention import Attention_cross

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# import torch
 
# import torchvision.transforms as T
 
# import matplotlib.pyplot as plt
 
# from PIL import Image
 
# from sklearn.decomposition import PCA
# import cv2

# import numpy as np
# transform = T.Compose([
#     T.GaussianBlur(9, sigma=(0.1, 2.0)),
#     T.Resize((36 * 14, 36 * 14)),
#     T.CenterCrop((36 * 14, 36 * 14)),
#     T.ToTensor(),
#     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
# ])
 
# from utils.fea_upscale import UpScale
# dinov2_vitb14=torch.hub.load('facebookresearch/dinov2','dinov2_vits14').cuda()


# def CONV(x,in_channels,out_channels):
#     conv2_layer=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True).cuda()
#     y=conv2_layer(x)
#     return y

# # ---------------------------------------------------------#
# # 门控融合机制 (Gated Fusion)
# # ---------------------------------------------------------#
# class GatedFusion(nn.Module):
#     def __init__(self, channels):
#         super(GatedFusion, self).__init__()
#         # 1x1卷积将拼接后的特征压缩回原通道数，然后接Sigmoid生成0~1的门控权重
#         self.gate = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(channels),
#             nn.Sigmoid()
#         )

#     def forward(self, x_cnn, x_dino):
#         # 1. 拼接
#         cat_feat = torch.cat([x_cnn, x_dino], dim=1)
#         # 2. 生成门控特征图 (Gate)
#         g = self.gate(cat_feat)
#         # 3. 控制 Dino 特征并加到原 CNN 特征上
#         out = x_cnn + g * x_dino
#         return out


# class MobileNetV2(nn.Module):
#     def __init__(self, downsample_factor=8, pretrained=True, 
#                  attention=False, MCattention=False, Mattention=False, QCattention=False, 
#                  fusion=False):
#         super(MobileNetV2, self).__init__()
#         from functools import partial
        
#         model           = mobilenetv2(pretrained)
#         self.features   = model.features[:-1]

#         self.total_idx  = len(self.features)
#         self.down_idx   = [2, 4, 7, 14]

#         # ----------------彻底关闭注意力机制----------------
#         self.check_atten=attention
#         self.check_MCatten=False  # 关闭多层注意力机制
#         self.check_Matten=False   # 关闭多头注意力机制
#         self.check_QCatten=False  # 【修改为False】关闭空间+通道注意力机制

#         # ----------------开启门控融合策略----------------
#         self.Fusion_Gate=True # 【开启】使用门控融合
#         self.Fusion1=False    # 关闭原有的相加
#         self.Fusion2=False
#         self.Fusion3=False
        
#         self.dino_layers=[2,5,8,11]
#         self.alpha = nn.Parameter(torch.zeros(1)) 
#         self.A1=Attention_cross(24)
#         self.A2=Attention_cross(64)
#         self.A3=Attention_cross(96)
#         self.A4=Attention_cross(160)
        
#         # ----------------初始化各个融合层的门控模块----------------
#         self.gate1 = GatedFusion(24)
#         self.gate2 = GatedFusion(64)
#         self.gate3 = GatedFusion(96)
#         self.gate4 = GatedFusion(160)

#         # 这里的初始化保留，以防代码其他地方调用，但前向传播中不会用到
#         if self.check_Matten:
#              self.attn_7=MultiheadAttention(64)
#              self.attn_11=MultiheadAttention(96)
#              self.attn_14=MultiheadAttention(160)
#              self.attn_x=MultiheadAttention(320)
        
#         elif self.check_MCatten:
#              self.attn_2=Attention(24)
#              self.attn_7=Attention(64)
#              self.attn_11=Attention(96)
#              self.attn_14=Attention(160)
#              self.attn_x=Attention(320)

#         elif self.check_QCatten:
#              self.pam7 = Danet_PositAttention(64)
#              self.pam11 = Danet_PositAttention(96)
#              self.pam14 = Danet_PositAttention(160)
#              self.pamx = Danet_PositAttention(320)

#              self.cam7 = Danet_ChannAttention()
#              self.cam11 = Danet_ChannAttention()
#              self.cam14 = Danet_ChannAttention()
#              self.camx = Danet_ChannAttention()

#         else:
#              self.attn_x=Attention(320)

#         if downsample_factor == 8:
#             for i in range(self.down_idx[-2], self.down_idx[-1]):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=2)
#                 )
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=4)
#                 )
#         elif downsample_factor == 16:
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=2)
#                 )
        
#     def _nostride_dilate(self, m, dilate):
#         classname = m.__class__.__name__
#         if classname.find('Conv') != -1:
#             if m.stride == (2, 2):
#                 m.stride = (1, 1)
#                 if m.kernel_size == (3, 3):
#                     m.dilation = (dilate//2, dilate//2)
#                     m.padding = (dilate//2, dilate//2)
#             else:
#                 if m.kernel_size == (3, 3):
#                     m.dilation = (dilate, dilate)
#                     m.padding = (dilate, dilate)

#     def forward(self, x):
        
#         _,_,H,W=x.shape
        
#         new_H=H//14*14
#         new_W=W//14*14
        
#         with torch.no_grad():
#             upsampled_tensor = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
#             dino=dinov2_vitb14.get_intermediate_layers(upsampled_tensor.cuda(),self.dino_layers)

#         # 因为所有注意力都关了，代码会自动进入这个 else 分支
#         # 此分支仅执行网络提取和 Gated Fusion，没有任何其他注意力机制
#         if self.Fusion_Gate:
#             # --------第1融合层--------
#             x_2 = self.features[:3](x)
#             D0 = UpScale(dino[0], x_2, token_hw=token_hw)
#             x_2 = self.gate1(x_2, D0)  # 仅门控融合
            
#             low_level_features = self.features[3:4](x_2) # 提取浅层特征用于Decoder
            
#             # --------第2融合层--------
#             x_7 = self.features[4:8](low_level_features)
#             D1 = UpScale(dino[1], x_7, token_hw=token_hw)
#             x_7 = self.gate2(x_7, D1)  # 仅门控融合
            
#             # --------第3融合层--------
#             x_11 = self.features[8:12](x_7)
#             D2 = UpScale(dino[2], x_11, token_hw=token_hw)
#             x_11 = self.gate3(x_11, D2)  # 仅门控融合
            
#             # --------第4融合层--------
#             x_14 = self.features[12:15](x_11)
#             D3 = UpScale(dino[3], x_14, token_hw=token_hw)
#             x_14 = self.gate4(x_14, D3)  # 仅门控融合
            
#             # 最终输出高级特征给 ASPP
#             x = self.features[15:](x_14)

#         # 下面的代码都是其他实验的备份代码（目前不会被执行到）
#         elif self.Fusion1:
#             x_2=self.features[:3](x)
#             D0=UpScale(dino[0],x_2)
#             fea0=D0*self.alpha
#             x_2=fea0+x_2
#             low_level_features=self.features[3:4](x_2)
#             x_7=self.features[4:8](low_level_features)
#             D1=UpScale(dino[1],x_7)
#             fea1=D1*self.alpha
#             x_7=fea1+x_7
#             x_11=self.features[8:12](x_7)
#             D2=UpScale(dino[2],x_11)
#             fea2=D2*self.alpha
#             x_11=fea2+x_11
#             x_14=self.features[12:15](x_11)
#             D3=UpScale(dino[3],x_14)
#             fea3=D3*self.alpha
#             x_14=fea3+x_14
#             x=self.features[15:](x_14)

#         elif self.Fusion2:
#             x_2=self.features[:3](x)
#             D0=UpScale(dino[0],x_2)
#             c2 = torch.cat((x_2, D0), dim=1)
#             r2 = CONV(c2,in_channels=c2.shape[1], out_channels=x_2.shape[1])
#             r2=x_2+r2*self.alpha
#             low_level_features=self.features[3:4](r2)
#             x_7=self.features[4:8](low_level_features)
#             D1=UpScale(dino[1],x_7)
#             c5=torch.cat((x_7,D1),dim=1)
#             r5 = CONV(c5,in_channels=c5.shape[1], out_channels=x_7.shape[1])
#             r7=x_7+r5*self.alpha
#             x_11=self.features[8:12](r7)
#             D2=UpScale(dino[2],x_11)
#             c11 = torch.cat((x_11, D2), dim=1)
#             r11 = CONV(c11,in_channels=c11.shape[1], out_channels=x_11.shape[1])
#             r11=x_11+r11*self.alpha
#             x_14=self.features[12:15](r11)
#             D3=UpScale(dino[3],x_14)
#             c14 = torch.cat((x_14, D3), dim=1)
#             r14 = CONV(c14,in_channels=c14.shape[1], out_channels=x_14.shape[1])
#             r14=x_14+r14*self.alpha
#             x=self.features[15:](r14)
            
#         elif self.Fusion3:
#             x_2=self.features[:3](x)
#             D0=UpScale(dino[0],x_2)
#             R_2=self.A1(x_2,D0)
#             low_level_features=self.features[3:4](R_2)
#             x_7=self.features[4:8](low_level_features)
#             D1=UpScale(dino[1],x_7)
#             R_7=self.A2(x_7,D1)
#             x_11=self.features[8:12](R_7)
#             D2=UpScale(dino[2],x_11)
#             R_11=self.A3(x_11,D2)
#             x_14=self.features[12:15](R_11)
#             D3=UpScale(dino[3],x_14)
#             R_14=self.A4(x_14,D3)
#             x = self.features[15:](R_14)
            
#         else:
#             low_level_features = self.features[:4](x)
#             x = self.features[4:](low_level_features)
#             if self.check_atten:
#                 x=self.attn_x(x)

#         return low_level_features, x 


# #-----------------------------------------#
# #   ASPP特征提取模块
# #   利用不同膨胀率的膨胀卷积进行特征提取
# #-----------------------------------------#
# class ASPP(nn.Module):
# 	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
# 		super(ASPP, self).__init__()
# 		self.branch1 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),
# 		)
# 		self.branch2 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch3 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch4 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
# 		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
# 		self.branch5_relu = nn.ReLU(inplace=True)

# 		self.conv_cat = nn.Sequential(
# 				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),		
# 		)

# 	def forward(self, x):
# 		[b, c, row, col] = x.size()
# 		conv1x1 = self.branch1(x)
# 		conv3x3_1 = self.branch2(x)
# 		conv3x3_2 = self.branch3(x)
# 		conv3x3_3 = self.branch4(x)
# 		global_feature = torch.mean(x,2,True)
# 		global_feature = torch.mean(global_feature,3,True)
# 		global_feature = self.branch5_conv(global_feature)
# 		global_feature = self.branch5_bn(global_feature)
# 		global_feature = self.branch5_relu(global_feature)
# 		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
# 		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
# 		result = self.conv_cat(feature_cat)
# 		return result

# class DeepLab(nn.Module):
#     def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16,attention=False,M_attention=False,QC_attention=True):
#         self.att=attention
#         self.Matt=M_attention
#         self.PCatt=QC_attention
#         self.fusion=False
#         super(DeepLab, self).__init__()
#         if backbone=="xception":
#             self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
#             in_channels = 2048
#             low_level_channels = 256
#         elif backbone=="mobilenet":
#             # MobileNetV2 内已经被我们硬编码改为门控融合无注意力了
#             self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained,attention=self.att,Mattention=self.Matt,QCattention=self.PCatt,fusion=self.fusion)
#             in_channels = 320
#             low_level_channels = 24

#         else:
#             raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

#         #-----------------------------------------#
#         #   ASPP特征提取模块
#         #-----------------------------------------#
#         self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
#         #----------------------------------#
#         #   浅层特征边
#         #----------------------------------#
#         self.shortcut_conv = nn.Sequential(
#             nn.Conv2d(low_level_channels, 48, 1),
#             nn.BatchNorm2d(48),
#             nn.ReLU(inplace=True)
#         )		

#         self.cat_conv = nn.Sequential(
#             nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),

#             nn.Conv2d(256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),

#             nn.Dropout(0.1),
#         )
#         self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

#     def forward(self, x):
#         H, W = x.size(2), x.size(3)
#         low_level_features, x = self.backbone(x)
#         x = self.aspp(x)
#         low_level_features = self.shortcut_conv(low_level_features)
        
#         x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
#         x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
#         x = self.cls_conv(x)
#         x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
#         return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from nets.xception import xception
# from nets.mobilenetv2 import mobilenetv2
# from nets.attention import Attention, MultiheadAttention, Danet_PositAttention, Danet_ChannAttention

# # 导入你原有的交叉注意力模块
# from nets.my_attention import Attention_cross

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# import torchvision.transforms as T
# import matplotlib.pyplot as plt
# from PIL import Image
# from sklearn.decomposition import PCA
# import cv2
# import numpy as np

# transform = T.Compose([
#     T.GaussianBlur(9, sigma=(0.1, 2.0)),
#     T.Resize((36 * 14, 36 * 14)),
#     T.CenterCrop((36 * 14, 36 * 14)),
#     T.ToTensor(),
#     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
# ])
 
# from utils.fea_upscale import UpScale
# dinov2_vitb14=torch.hub.load('facebookresearch/dinov2','dinov2_vits14').cuda()


# def CONV(x,in_channels,out_channels):
#     conv2_layer=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True).cuda()
#     y=conv2_layer(x)
#     return y


# class MobileNetV2(nn.Module):
#     def __init__(self, downsample_factor=8, pretrained=True, 
#                  attention=False, MCattention=False, Mattention=False, QCattention=False, 
#                  fusion=False):
#         super(MobileNetV2, self).__init__()
#         from functools import partial
        
#         model           = mobilenetv2(pretrained)
#         self.features   = model.features[:-1]

#         self.total_idx  = len(self.features)
#         self.down_idx   =[2, 4, 7, 14]

#         # ----------------彻底关闭其他附加注意力机制----------------
#         self.check_atten=False
#         self.check_MCatten=False  # 关闭多层注意力
#         self.check_Matten=False   # 关闭多头注意力
#         self.check_QCatten=False  # 关闭空间+通道注意力

#         # ----------------开启交叉注意力融合策略----------------
#         self.Fusion_CrossAttention=True # 【新增】开启交叉注意力融合
#         self.Fusion1=False    
#         self.Fusion2=False
#         self.Fusion3=False
        
#         self.dino_layers=[2,5,8,11]
#         self.alpha = nn.Parameter(torch.zeros(1)) 
        
#         # ----------------初始化四个融合层的交叉注意力模块----------------
#         # 你的原代码已经写好了这部分，直接保留即可
#         self.A1=Attention_cross(24)
#         self.A2=Attention_cross(64)
#         self.A3=Attention_cross(96)
#         self.A4=Attention_cross(160)

#         if downsample_factor == 8:
#             for i in range(self.down_idx[-2], self.down_idx[-1]):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=2)
#                 )
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=4)
#                 )
#         elif downsample_factor == 16:
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=2)
#                 )
        
#     def _nostride_dilate(self, m, dilate):
#         classname = m.__class__.__name__
#         if classname.find('Conv') != -1:
#             if m.stride == (2, 2):
#                 m.stride = (1, 1)
#                 if m.kernel_size == (3, 3):
#                     m.dilation = (dilate//2, dilate//2)
#                     m.padding = (dilate//2, dilate//2)
#             else:
#                 if m.kernel_size == (3, 3):
#                     m.dilation = (dilate, dilate)
#                     m.padding = (dilate, dilate)

#     def forward(self, x):
        
#         _,_,H,W=x.shape
        
#         new_H=H//14*14
#         new_W=W//14*14
        
#         with torch.no_grad():
#             upsampled_tensor = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
#             dino=dinov2_vitb14.get_intermediate_layers(upsampled_tensor.cuda(),self.dino_layers)


#         # ========================================================
#         # 核心逻辑：仅使用交叉注意力融合 (Cross Attention Fusion)
#         # ========================================================
#         if self.Fusion_CrossAttention:
            
#             # --------第1融合层--------
#             x_2 = self.features[:3](x)
#             D0 = UpScale(dino[0], x_2, token_hw=token_hw)
#             # CNN特征和Dino特征通过交叉注意力融合
#             x_2 = self.A1(x_2, D0)  
            
#             low_level_features = self.features[3:4](x_2) # 提取浅层特征用于Decoder
            
#             # --------第2融合层--------
#             x_7 = self.features[4:8](low_level_features)
#             D1 = UpScale(dino[1], x_7, token_hw=token_hw)
#             # 交叉注意力融合
#             x_7 = self.A2(x_7, D1)  
            
#             # --------第3融合层--------
#             x_11 = self.features[8:12](x_7)
#             D2 = UpScale(dino[2], x_11, token_hw=token_hw)
#             # 交叉注意力融合
#             x_11 = self.A3(x_11, D2)  
            
#             # --------第4融合层--------
#             x_14 = self.features[12:15](x_11)
#             D3 = UpScale(dino[3], x_14, token_hw=token_hw)
#             # 交叉注意力融合
#             x_14 = self.A4(x_14, D3)  
            
#             # 最终输出高级特征给 ASPP
#             x = self.features[15:](x_14)

#         # ----------------其他实验备份（不会执行到）----------------
#         else:
#             low_level_features = self.features[:4](x)
#             x = self.features[4:](low_level_features)

#         return low_level_features, x 


# #-----------------------------------------#
# #   ASPP特征提取模块
# #   利用不同膨胀率的膨胀卷积进行特征提取
# #-----------------------------------------#
# class ASPP(nn.Module):
# 	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
# 		super(ASPP, self).__init__()
# 		self.branch1 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),
# 		)
# 		self.branch2 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch3 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch4 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
# 		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
# 		self.branch5_relu = nn.ReLU(inplace=True)

# 		self.conv_cat = nn.Sequential(
# 				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),		
# 		)

# 	def forward(self, x):
# 		[b, c, row, col] = x.size()
# 		conv1x1 = self.branch1(x)
# 		conv3x3_1 = self.branch2(x)
# 		conv3x3_2 = self.branch3(x)
# 		conv3x3_3 = self.branch4(x)
# 		global_feature = torch.mean(x,2,True)
# 		global_feature = torch.mean(global_feature,3,True)
# 		global_feature = self.branch5_conv(global_feature)
# 		global_feature = self.branch5_bn(global_feature)
# 		global_feature = self.branch5_relu(global_feature)
# 		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
# 		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
# 		result = self.conv_cat(feature_cat)
# 		return result

# class DeepLab(nn.Module):
#     def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16,attention=False,M_attention=False,QC_attention=True):
#         self.att=attention
#         self.Matt=M_attention
#         self.PCatt=QC_attention
#         self.fusion=False
#         super(DeepLab, self).__init__()
#         if backbone=="xception":
#             self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
#             in_channels = 2048
#             low_level_channels = 256
#         elif backbone=="mobilenet":
#             # MobileNetV2 内部已硬编码为交叉注意力融合，并关闭了其他注意力
#             self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained,attention=self.att,Mattention=self.Matt,QCattention=self.PCatt,fusion=self.fusion)
#             in_channels = 320
#             low_level_channels = 24

#         else:
#             raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

#         #-----------------------------------------#
#         #   ASPP特征提取模块
#         #-----------------------------------------#
#         self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
#         #----------------------------------#
#         #   浅层特征边
#         #----------------------------------#
#         self.shortcut_conv = nn.Sequential(
#             nn.Conv2d(low_level_channels, 48, 1),
#             nn.BatchNorm2d(48),
#             nn.ReLU(inplace=True)
#         )		

#         self.cat_conv = nn.Sequential(
#             nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),

#             nn.Conv2d(256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),

#             nn.Dropout(0.1),
#         )
#         self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

#     def forward(self, x):
#         H, W = x.size(2), x.size(3)
#         low_level_features, x = self.backbone(x)
#         x = self.aspp(x)
#         low_level_features = self.shortcut_conv(low_level_features)
        
#         x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
#         x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
#         x = self.cls_conv(x)
#         x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
#         return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from nets.xception import xception
# from nets.mobilenetv2 import mobilenetv2

# # -------------------------------------------------------------------#
# #   MobileNetV2 主干网络 (DeepLabV3+ 专用版)
# #   去除了所有的 DINO 融合和注意力机制，保留了原始的空洞卷积逻辑
# # -------------------------------------------------------------------#
# class MobileNetV2(nn.Module):
#     def __init__(self, downsample_factor=8, pretrained=True):
#         super(MobileNetV2, self).__init__()
#         from functools import partial
        
#         # 加载 MobileNetV2，并去掉最后的全连接和池化层
#         model           = mobilenetv2(pretrained)
#         self.features   = model.features[:-1] 

#         self.total_idx  = len(self.features)
#         self.down_idx   = [2, 4, 7, 14]

#         # ---------------------------------------------------------#
#         #   根据 downsample_factor 设置不同层的空洞卷积 (Dilated Conv)
#         #   这是 DeepLab 系列的核心操作，用来在不下采样的情况下扩大感受野
#         # ---------------------------------------------------------#
#         if downsample_factor == 8:
#             for i in range(self.down_idx[-2], self.down_idx[-1]):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=2)
#                 )
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=4)
#                 )
#         elif downsample_factor == 16:
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=2)
#                 )
        
#     def _nostride_dilate(self, m, dilate):
#         classname = m.__class__.__name__
#         if classname.find('Conv') != -1:
#             if m.stride == (2, 2):
#                 m.stride = (1, 1)
#                 if m.kernel_size == (3, 3):
#                     m.dilation = (dilate//2, dilate//2)
#                     m.padding = (dilate//2, dilate//2)
#             else:
#                 if m.kernel_size == (3, 3):
#                     m.dilation = (dilate, dilate)
#                     m.padding = (dilate, dilate)

#     def forward(self, x):
#         # ---------------------------------------------------------#
#         #   原版 DeepLabV3+ 的特征提取流程：
#         #   1. 获取浅层特征 (low_level_features)，一般取前4层，通道数为24
#         #   2. 获取深层特征 (x)，通过后续所有层，通道数为320
#         # ---------------------------------------------------------#
#         low_level_features = self.features[:4](x)
#         x = self.features[4:](low_level_features)

#         return low_level_features, x 


# #-----------------------------------------#
# #   ASPP 特征提取模块
# #   利用不同膨胀率的膨胀卷积进行多尺度特征提取
# #-----------------------------------------#
# class ASPP(nn.Module):
# 	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
# 		super(ASPP, self).__init__()
# 		self.branch1 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),
# 		)
# 		self.branch2 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch3 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch4 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
# 		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
# 		self.branch5_relu = nn.ReLU(inplace=True)

# 		self.conv_cat = nn.Sequential(
# 				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),		
# 		)

# 	def forward(self, x):
# 		[b, c, row, col] = x.size()
#         # 一共五个分支
# 		conv1x1 = self.branch1(x)
# 		conv3x3_1 = self.branch2(x)
# 		conv3x3_2 = self.branch3(x)
# 		conv3x3_3 = self.branch4(x)
        
#         # 第五个分支：全局平均池化 + 卷积
# 		global_feature = torch.mean(x, 2, True)
# 		global_feature = torch.mean(global_feature, 3, True)
# 		global_feature = self.branch5_conv(global_feature)
# 		global_feature = self.branch5_bn(global_feature)
# 		global_feature = self.branch5_relu(global_feature)
# 		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
#         # 将五个分支的内容拼接起来，然后用1x1卷积整合特征
# 		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
# 		result = self.conv_cat(feature_cat)
# 		return result


# #-----------------------------------------#
# #   DeepLabV3+ 完整模型
# #-----------------------------------------#
# class DeepLab(nn.Module):
#     def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
#         super(DeepLab, self).__init__()
        
#         if backbone == "xception":
#             # 浅层特征通道数 256，主干输出通道数 2048
#             self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
#             in_channels = 2048
#             low_level_channels = 256
#         elif backbone == "mobilenet":
#             # 浅层特征通道数 24，主干输出通道数 320
#             self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
#             in_channels = 320
#             low_level_channels = 24
#         else:
#             raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

#         #-----------------------------------------#
#         #   ASPP 特征提取模块
#         #-----------------------------------------#
#         self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
#         #----------------------------------#
#         #   浅层特征边 (Decoder)
#         #----------------------------------#
#         self.shortcut_conv = nn.Sequential(
#             nn.Conv2d(low_level_channels, 48, 1),
#             nn.BatchNorm2d(48),
#             nn.ReLU(inplace=True)
#         )		

#         # 将 ASPP 结果与浅层特征拼接后的卷积块
#         self.cat_conv = nn.Sequential(
#             nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),

#             nn.Conv2d(256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),

#             nn.Dropout(0.1),
#         )
#         self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

#     def forward(self, x):
#         H, W = x.size(2), x.size(3)
        
#         # 1. 主干网络提取特征
#         low_level_features, x = self.backbone(x)
        
#         # 2. 高层特征经过 ASPP 模块
#         x = self.aspp(x)
        
#         # 3. 浅层特征通过 1x1 卷积降维
#         low_level_features = self.shortcut_conv(low_level_features)
        
#         # 4. 高层特征上采样，并与浅层特征拼接
#         x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
#         x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        
#         # 5. 最后通过分类卷积层，并恢复到原图大小
#         x = self.cls_conv(x)
#         x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
#         return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from nets.xception import xception
# from nets.mobilenetv2 import mobilenetv2
# from nets.attention import Attention, MultiheadAttention, Danet_PositAttention, Danet_ChannAttention

# # 导入你原有的交叉注意力模块
# from nets.my_attention import Attention_cross

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# import torchvision.transforms as T
# import matplotlib.pyplot as plt
# from PIL import Image
# from sklearn.decomposition import PCA
# import cv2
# import numpy as np

# transform = T.Compose([
#     T.GaussianBlur(9, sigma=(0.1, 2.0)),
#     T.Resize((36 * 14, 36 * 14)),
#     T.CenterCrop((36 * 14, 36 * 14)),
#     T.ToTensor(),
#     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
# ])
 
# from utils.fea_upscale import UpScale
# dinov2_vitb14=torch.hub.load('facebookresearch/dinov2','dinov2_vits14').cuda()


# def CONV(x,in_channels,out_channels):
#     conv2_layer=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True).cuda()
#     y=conv2_layer(x)
#     return y


# class MobileNetV2(nn.Module):
#     def __init__(self, downsample_factor=8, pretrained=True, 
#                  attention=False, MCattention=False, Mattention=False, QCattention=False, 
#                  fusion=False):
#         super(MobileNetV2, self).__init__()
#         from functools import partial
        
#         model           = mobilenetv2(pretrained)
#         self.features   = model.features[:-1]

#         self.total_idx  = len(self.features)
#         self.down_idx   =[2, 4, 7, 14]

#         # ----------------彻底关闭其他附加注意力机制----------------
#         self.check_atten=False
#         self.check_MCatten=False  # 关闭多层注意力
#         self.check_Matten=False   # 关闭多头注意力
#         self.check_QCatten=False  # 关闭空间+通道注意力

#         # ----------------开启交叉注意力融合策略----------------
#         self.Fusion_CrossAttention=True # 【新增】开启交叉注意力融合
#         self.Fusion1=False    
#         self.Fusion2=False
#         self.Fusion3=False
        
#         self.dino_layers=[2,5,8,11]
#         self.alpha = nn.Parameter(torch.zeros(1)) 
        
#         # ----------------初始化四个融合层的交叉注意力模块----------------
#         # 你的原代码已经写好了这部分，直接保留即可
#         self.A1=Attention_cross(24)
#         self.A2=Attention_cross(64)
#         self.A3=Attention_cross(96)
#         self.A4=Attention_cross(160)

#         if downsample_factor == 8:
#             for i in range(self.down_idx[-2], self.down_idx[-1]):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=2)
#                 )
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=4)
#                 )
#         elif downsample_factor == 16:
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=2)
#                 )
        
#     def _nostride_dilate(self, m, dilate):
#         classname = m.__class__.__name__
#         if classname.find('Conv') != -1:
#             if m.stride == (2, 2):
#                 m.stride = (1, 1)
#                 if m.kernel_size == (3, 3):
#                     m.dilation = (dilate//2, dilate//2)
#                     m.padding = (dilate//2, dilate//2)
#             else:
#                 if m.kernel_size == (3, 3):
#                     m.dilation = (dilate, dilate)
#                     m.padding = (dilate, dilate)

#     def forward(self, x):
        
#         _,_,H,W=x.shape
        
#         new_H=H//14*14
#         new_W=W//14*14
        
#         with torch.no_grad():
#             upsampled_tensor = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
#             dino=dinov2_vitb14.get_intermediate_layers(upsampled_tensor.cuda(),self.dino_layers)


#         # ========================================================
#         # 核心逻辑：仅使用交叉注意力融合 (Cross Attention Fusion)
#         # ========================================================
#         if self.Fusion_CrossAttention:
            
#             # --------第1融合层--------
#             x_2 = self.features[:3](x)
#             D0 = UpScale(dino[0], x_2, token_hw=token_hw)
#             # CNN特征和Dino特征通过交叉注意力融合
#             x_2 = self.A1(x_2, D0)  
            
#             low_level_features = self.features[3:4](x_2) # 提取浅层特征用于Decoder
            
#             # --------第2融合层--------
#             x_7 = self.features[4:8](low_level_features)
#             D1 = UpScale(dino[1], x_7, token_hw=token_hw)
#             # 交叉注意力融合
#             x_7 = self.A2(x_7, D1)  
            
#             # --------第3融合层--------
#             x_11 = self.features[8:12](x_7)
#             D2 = UpScale(dino[2], x_11, token_hw=token_hw)
#             # 交叉注意力融合
#             x_11 = self.A3(x_11, D2)  
            
#             # --------第4融合层--------
#             x_14 = self.features[12:15](x_11)
#             D3 = UpScale(dino[3], x_14, token_hw=token_hw)
#             # 交叉注意力融合
#             x_14 = self.A4(x_14, D3)  
            
#             # 最终输出高级特征给 ASPP
#             x = self.features[15:](x_14)

#         # ----------------其他实验备份（不会执行到）----------------
#         else:
#             low_level_features = self.features[:4](x)
#             x = self.features[4:](low_level_features)

#         return low_level_features, x 


# #-----------------------------------------#
# #   ASPP特征提取模块
# #   利用不同膨胀率的膨胀卷积进行特征提取
# #-----------------------------------------#
# class ASPP(nn.Module):
# 	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
# 		super(ASPP, self).__init__()
# 		self.branch1 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),
# 		)
# 		self.branch2 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch3 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch4 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
# 		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
# 		self.branch5_relu = nn.ReLU(inplace=True)

# 		self.conv_cat = nn.Sequential(
# 				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),		
# 		)

# 	def forward(self, x):
# 		[b, c, row, col] = x.size()
# 		conv1x1 = self.branch1(x)
# 		conv3x3_1 = self.branch2(x)
# 		conv3x3_2 = self.branch3(x)
# 		conv3x3_3 = self.branch4(x)
# 		global_feature = torch.mean(x,2,True)
# 		global_feature = torch.mean(global_feature,3,True)
# 		global_feature = self.branch5_conv(global_feature)
# 		global_feature = self.branch5_bn(global_feature)
# 		global_feature = self.branch5_relu(global_feature)
# 		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
# 		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
# 		result = self.conv_cat(feature_cat)
# 		return result

# class DeepLab(nn.Module):
#     def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16,attention=False,M_attention=False,QC_attention=True):
#         self.att=attention
#         self.Matt=M_attention
#         self.PCatt=QC_attention
#         self.fusion=False
#         super(DeepLab, self).__init__()
#         if backbone=="xception":
#             self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
#             in_channels = 2048
#             low_level_channels = 256
#         elif backbone=="mobilenet":
#             # MobileNetV2 内部已硬编码为交叉注意力融合，并关闭了其他注意力
#             self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained,attention=self.att,Mattention=self.Matt,QCattention=self.PCatt,fusion=self.fusion)
#             in_channels = 320
#             low_level_channels = 24

#         else:
#             raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

#         #-----------------------------------------#
#         #   ASPP特征提取模块
#         #-----------------------------------------#
#         self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
#         #----------------------------------#
#         #   浅层特征边
#         #----------------------------------#
#         self.shortcut_conv = nn.Sequential(
#             nn.Conv2d(low_level_channels, 48, 1),
#             nn.BatchNorm2d(48),
#             nn.ReLU(inplace=True)
#         )		

#         self.cat_conv = nn.Sequential(
#             nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),

#             nn.Conv2d(256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),

#             nn.Dropout(0.1),
#         )
#         self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

#     def forward(self, x):
#         H, W = x.size(2), x.size(3)
#         low_level_features, x = self.backbone(x)
#         x = self.aspp(x)
#         low_level_features = self.shortcut_conv(low_level_features)
        
#         x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
#         x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
#         x = self.cls_conv(x)
#         x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
#         return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from nets.xception import xception
# from nets.mobilenetv2 import mobilenetv2

# # 导入你原有的交叉注意力模块
# from nets.my_attention import Attention_cross

# import torchvision.transforms as T
# import cv2
# import numpy as np

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transform = T.Compose([
#     T.GaussianBlur(9, sigma=(0.1, 2.0)),
#     T.Resize((36 * 14, 36 * 14)),
#     T.CenterCrop((36 * 14, 36 * 14)),
#     T.ToTensor(),
#     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
# ])
 
# from utils.fea_upscale import UpScale

# # 加载 DINOv2 模型
# _DINOV2_VITB14 = None
#
# def _get_dinov2_vitb14():
#     global _DINOV2_VITB14
#     if _DINOV2_VITB14 is None:
#         _DINOV2_VITB14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
#     return _DINOV2_VITB14

# def CONV(x, in_channels, out_channels):
#     conv2_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True).cuda()
#     y = conv2_layer(x)
#     return y

# # -------------------------------------------------------------------#
# #   MobileNetV2 主干网络 (交叉注意力 + 尺度对齐版)
# # -------------------------------------------------------------------#
# class MobileNetV2(nn.Module):
#     def __init__(self, downsample_factor=8, pretrained=True):
#         super(MobileNetV2, self).__init__()
#         from functools import partial
        
#         model           = mobilenetv2(pretrained)
#         self.features   = model.features[:-1]

#         self.total_idx  = len(self.features)
#         self.down_idx   =[2, 4, 7, 14]

#         # ----------------控制开关----------------
#         self.Fusion_CrossAttention = True 
        
#         self.dino_layers = [2, 5, 8, 11]
        
#         # ----------------核心创新点：独立的可学习尺度对齐参数----------------
#         # 每一层初始化为 0，保证初期网络退化为纯 CNN（零初始化残差机制）
#         # 随着训练，网络会自动学习 CNN 与 DINO 特征的数值比例
#         self.alpha1 = nn.Parameter(torch.zeros(1))
#         self.alpha2 = nn.Parameter(torch.zeros(1))
#         self.alpha3 = nn.Parameter(torch.zeros(1))
#         self.alpha4 = nn.Parameter(torch.zeros(1))
        
#         # ----------------初始化四个融合层的交叉注意力模块----------------
#         self.A1 = Attention_cross(24)
#         self.A2 = Attention_cross(64)
#         self.A3 = Attention_cross(96)
#         self.A4 = Attention_cross(160)

#         # ----------------配置空洞卷积----------------
#         if downsample_factor == 8:
#             for i in range(self.down_idx[-2], self.down_idx[-1]):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=2)
#                 )
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=4)
#                 )
#         elif downsample_factor == 16:
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=2)
#                 )
        
#     def _nostride_dilate(self, m, dilate):
#         classname = m.__class__.__name__
#         if classname.find('Conv') != -1:
#             if m.stride == (2, 2):
#                 m.stride = (1, 1)
#                 if m.kernel_size == (3, 3):
#                     m.dilation = (dilate//2, dilate//2)
#                     m.padding = (dilate//2, dilate//2)
#             else:
#                 if m.kernel_size == (3, 3):
#                     m.dilation = (dilate, dilate)
#                     m.padding = (dilate, dilate)

#     def forward(self, x):
#         _, _, H, W = x.shape
        
#         # 根据 DINOv2 的 Patch Size (14) 调整尺寸
#         new_H = H // 14 * 14
#         new_W = W // 14 * 14
        
#         with torch.no_grad():
#             upsampled_tensor = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
#             # 获取 DINOv2 对应层的特征
#             dino = _get_dinov2_vitb14().get_intermediate_layers(upsampled_tensor.cuda(), self.dino_layers)

#         # ========================================================
#         #   交叉注意力融合 (Cross Attention Fusion) + 尺度对齐
#         # ========================================================
#         if self.Fusion_CrossAttention:
            
#             # -------- 第 1 融合层 --------
#             x_2 = self.features[:3](x)
#             D0 = UpScale(dino[0], x_2, token_hw=token_hw)
            
#             # 1. 尺度对齐：利用 alpha1 调节 DINO 特征的数值分布
#             D0_aligned = D0 * self.alpha1 
#             # 2. 交叉注意力融合：CNN 特征做 Query，对齐后的 DINO 特征做 Key 和 Value
#             # (如果 A1 内部自带了残差 x + attn_out，因为 D0_aligned 为 0，输出则完美等于 x_2)
#             x_2 = self.A1(x_2, D0_aligned)  
            
#             low_level_features = self.features[3:4](x_2) # 提取给 Decoder 使用的浅层特征
            
#             # -------- 第 2 融合层 --------
#             x_7 = self.features[4:8](low_level_features)
#             D1 = UpScale(dino[1], x_7, token_hw=token_hw)
            
#             D1_aligned = D1 * self.alpha2
#             x_7 = self.A2(x_7, D1_aligned)  
            
#             # -------- 第 3 融合层 --------
#             x_11 = self.features[8:12](x_7)
#             D2 = UpScale(dino[2], x_11, token_hw=token_hw)
            
#             D2_aligned = D2 * self.alpha3
#             x_11 = self.A3(x_11, D2_aligned)  
            
#             # -------- 第 4 融合层 --------
#             x_14 = self.features[12:15](x_11)
#             D3 = UpScale(dino[3], x_14, token_hw=token_hw)
            
#             D3_aligned = D3 * self.alpha4
#             x_14 = self.A4(x_14, D3_aligned)  
            
#             # 将最终的高层特征输出给 ASPP 模块
#             x = self.features[15:](x_14)

#         else:
#             # 安全备份逻辑 (默认不会走到这里)
#             low_level_features = self.features[:4](x)
#             x = self.features[4:](low_level_features)

#         return low_level_features, x 


# #-----------------------------------------#
# #   ASPP特征提取模块
# #   利用不同膨胀率的膨胀卷积进行特征提取
# #-----------------------------------------#
# class ASPP(nn.Module):
# 	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
# 		super(ASPP, self).__init__()
# 		self.branch1 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),
# 		)
# 		self.branch2 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch3 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch4 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
# 		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
# 		self.branch5_relu = nn.ReLU(inplace=True)

# 		self.conv_cat = nn.Sequential(
# 				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),		
# 		)

# 	def forward(self, x):
# 		[b, c, row, col] = x.size()
# 		conv1x1 = self.branch1(x)
# 		conv3x3_1 = self.branch2(x)
# 		conv3x3_2 = self.branch3(x)
# 		conv3x3_3 = self.branch4(x)
		
# 		global_feature = torch.mean(x, 2, True)
# 		global_feature = torch.mean(global_feature, 3, True)
# 		global_feature = self.branch5_conv(global_feature)
# 		global_feature = self.branch5_bn(global_feature)
# 		global_feature = self.branch5_relu(global_feature)
# 		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
# 		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
# 		result = self.conv_cat(feature_cat)
# 		return result

# #-----------------------------------------#
# #   DeepLabV3+ 主模型
# #-----------------------------------------#
# class DeepLab(nn.Module):
#     def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
#         super(DeepLab, self).__init__()
        
#         if backbone == "xception":
#             self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
#             in_channels = 2048
#             low_level_channels = 256
#         elif backbone == "mobilenet":
#             # MobileNetV2 内部已清理并硬编码为交叉注意力融合
#             self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
#             in_channels = 320
#             low_level_channels = 24
#         else:
#             raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

#         #-----------------------------------------#
#         #   ASPP 特征提取模块
#         #-----------------------------------------#
#         self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
#         #----------------------------------#
#         #   浅层特征边 (Decoder)
#         #----------------------------------#
#         self.shortcut_conv = nn.Sequential(
#             nn.Conv2d(low_level_channels, 48, 1),
#             nn.BatchNorm2d(48),
#             nn.ReLU(inplace=True)
#         )		

#         self.cat_conv = nn.Sequential(
#             nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),

#             nn.Conv2d(256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),

#             nn.Dropout(0.1),
#         )
#         self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

#     def forward(self, x):
#         H, W = x.size(2), x.size(3)
        
#         low_level_features, x = self.backbone(x)
#         x = self.aspp(x)
#         low_level_features = self.shortcut_conv(low_level_features)
        
#         x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
#         x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
#         x = self.cls_conv(x)
#         x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
#         return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from nets.xception import xception
# from nets.mobilenetv2 import mobilenetv2

# import torchvision.transforms as T
# import cv2
# import numpy as np

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transform = T.Compose([
#     T.GaussianBlur(9, sigma=(0.1, 2.0)),
#     T.Resize((36 * 14, 36 * 14)),
#     T.CenterCrop((36 * 14, 36 * 14)),
#     T.ToTensor(),
#     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
# ])
 
# from utils.fea_upscale import UpScale

# # 加载 DINOv2 模型
# _DINOV2_VITB14 = None
#
# def _get_dinov2_vitb14():
#     global _DINOV2_VITB14
#     if _DINOV2_VITB14 is None:
#         _DINOV2_VITB14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
#     return _DINOV2_VITB14

# def CONV(x, in_channels, out_channels):
#     # (注：这个是原代码里的函数，但每次调用都会重新初始化权重，不可学习。我们在下面采用了正规的类定义。)
#     conv2_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True).cuda()
#     y = conv2_layer(x)
#     return y

# # ---------------------------------------------------------#
# # 拼接融合机制 (Concat Fusion)
# # ---------------------------------------------------------#
# class ConcatFusion(nn.Module):
#     def __init__(self, channels):
#         super(ConcatFusion, self).__init__()
#         # 使用 1x1 卷积将拼接后翻倍的通道数 (channels * 2) 压缩回原通道数 (channels)
#         self.conv = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x_cnn, x_dino):
#         # 1. 在通道维度进行拼接 (Concat)
#         cat_feat = torch.cat([x_cnn, x_dino], dim=1)
#         # 2. 通过 1x1 卷积进行降维和特征整合
#         out = self.conv(cat_feat)
#         return out


# # -------------------------------------------------------------------#
# #   MobileNetV2 主干网络 (Concat 拼接融合版)
# # -------------------------------------------------------------------#
# class MobileNetV2(nn.Module):
#     def __init__(self, downsample_factor=8, pretrained=True):
#         super(MobileNetV2, self).__init__()
#         from functools import partial
        
#         model           = mobilenetv2(pretrained)
#         self.features   = model.features[:-1]

#         self.total_idx  = len(self.features)
#         self.down_idx   =[2, 4, 7, 14]

#         # ----------------控制开关----------------
#         self.Fusion_Concat = True 
        
#         self.dino_layers =[2, 5, 8, 11]

#         # ----------------初始化四个阶段的 Concat 融合模块----------------
#         # 对应的通道数分别是 24, 64, 96, 160
#         self.concat1 = ConcatFusion(24)
#         self.concat2 = ConcatFusion(64)
#         self.concat3 = ConcatFusion(96)
#         self.concat4 = ConcatFusion(160)

#         # ----------------配置空洞卷积----------------
#         if downsample_factor == 8:
#             for i in range(self.down_idx[-2], self.down_idx[-1]):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=2)
#                 )
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=4)
#                 )
#         elif downsample_factor == 16:
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=2)
#                 )
        
#     def _nostride_dilate(self, m, dilate):
#         classname = m.__class__.__name__
#         if classname.find('Conv') != -1:
#             if m.stride == (2, 2):
#                 m.stride = (1, 1)
#                 if m.kernel_size == (3, 3):
#                     m.dilation = (dilate//2, dilate//2)
#                     m.padding = (dilate//2, dilate//2)
#             else:
#                 if m.kernel_size == (3, 3):
#                     m.dilation = (dilate, dilate)
#                     m.padding = (dilate, dilate)

#     def forward(self, x):
#         _, _, H, W = x.shape
        
#         # 根据 DINOv2 的 Patch Size (14) 调整尺寸
#         new_H = H // 14 * 14
#         new_W = W // 14 * 14
        
#         with torch.no_grad():
#             upsampled_tensor = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
#             # 获取 DINOv2 对应层的特征
#             dino = _get_dinov2_vitb14().get_intermediate_layers(upsampled_tensor.cuda(), self.dino_layers)

#         # ========================================================
#         #   Concat 拼接融合 (拼接 + 1x1卷积)
#         # ========================================================
#         if self.Fusion_Concat:
            
#             # -------- 第 1 融合层 --------
#             x_2 = self.features[:3](x)
#             D0 = UpScale(dino[0], x_2, token_hw=token_hw)
            
#             # 使用 Concat 模块融合
#             x_2 = self.concat1(x_2, D0)  
            
#             low_level_features = self.features[3:4](x_2) # 提取给 Decoder 使用的浅层特征
            
#             # -------- 第 2 融合层 --------
#             x_7 = self.features[4:8](low_level_features)
#             D1 = UpScale(dino[1], x_7, token_hw=token_hw)
            
#             x_7 = self.concat2(x_7, D1)  
            
#             # -------- 第 3 融合层 --------
#             x_11 = self.features[8:12](x_7)
#             D2 = UpScale(dino[2], x_11, token_hw=token_hw)
            
#             x_11 = self.concat3(x_11, D2)  
            
#             # -------- 第 4 融合层 --------
#             x_14 = self.features[12:15](x_11)
#             D3 = UpScale(dino[3], x_14, token_hw=token_hw)
            
#             x_14 = self.concat4(x_14, D3)  
            
#             # 将最终的高层特征输出给 ASPP 模块
#             x = self.features[15:](x_14)

#         else:
#             # 安全备份逻辑
#             low_level_features = self.features[:4](x)
#             x = self.features[4:](low_level_features)

#         return low_level_features, x 


# #-----------------------------------------#
# #   ASPP特征提取模块
# #   利用不同膨胀率的膨胀卷积进行特征提取
# #-----------------------------------------#
# class ASPP(nn.Module):
# 	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
# 		super(ASPP, self).__init__()
# 		self.branch1 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),
# 		)
# 		self.branch2 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch3 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch4 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
# 		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
# 		self.branch5_relu = nn.ReLU(inplace=True)

# 		self.conv_cat = nn.Sequential(
# 				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),		
# 		)

# 	def forward(self, x):
# 		[b, c, row, col] = x.size()
# 		conv1x1 = self.branch1(x)
# 		conv3x3_1 = self.branch2(x)
# 		conv3x3_2 = self.branch3(x)
# 		conv3x3_3 = self.branch4(x)
		
# 		global_feature = torch.mean(x, 2, True)
# 		global_feature = torch.mean(global_feature, 3, True)
# 		global_feature = self.branch5_conv(global_feature)
# 		global_feature = self.branch5_bn(global_feature)
# 		global_feature = self.branch5_relu(global_feature)
# 		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
# 		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
# 		result = self.conv_cat(feature_cat)
# 		return result

# #-----------------------------------------#
# #   DeepLabV3+ 主模型
# #-----------------------------------------#
# class DeepLab(nn.Module):
#     def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
#         super(DeepLab, self).__init__()
        
#         if backbone == "xception":
#             self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
#             in_channels = 2048
#             low_level_channels = 256
#         elif backbone == "mobilenet":
#             # MobileNetV2 内部已清理并硬编码为 Concat 拼接融合
#             self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
#             in_channels = 320
#             low_level_channels = 24
#         else:
#             raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

#         #-----------------------------------------#
#         #   ASPP 特征提取模块
#         #-----------------------------------------#
#         self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
#         #----------------------------------#
#         #   浅层特征边 (Decoder)
#         #----------------------------------#
#         self.shortcut_conv = nn.Sequential(
#             nn.Conv2d(low_level_channels, 48, 1),
#             nn.BatchNorm2d(48),
#             nn.ReLU(inplace=True)
#         )		

#         self.cat_conv = nn.Sequential(
#             nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),

#             nn.Conv2d(256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),

#             nn.Dropout(0.1),
#         )
#         self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

#     def forward(self, x):
#         H, W = x.size(2), x.size(3)
        
#         low_level_features, x = self.backbone(x)
#         x = self.aspp(x)
#         low_level_features = self.shortcut_conv(low_level_features)
        
#         x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
#         x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
#         x = self.cls_conv(x)
#         x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
#         return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from nets.xception import xception
# from nets.mobilenetv2 import mobilenetv2

# import torchvision.transforms as T
# import cv2
# import numpy as np

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transform = T.Compose([
#     T.GaussianBlur(9, sigma=(0.1, 2.0)),
#     T.Resize((36 * 14, 36 * 14)),
#     T.CenterCrop((36 * 14, 36 * 14)),
#     T.ToTensor(),
#     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
# ])
 
# from utils.fea_upscale import UpScale

# # 加载 DINOv2 模型
# _DINOV2_VITB14 = None
#
# def _get_dinov2_vitb14():
#     global _DINOV2_VITB14
#     if _DINOV2_VITB14 is None:
#         _DINOV2_VITB14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
#     return _DINOV2_VITB14

# def CONV(x, in_channels, out_channels):
#     conv2_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True).cuda()
#     y = conv2_layer(x)
#     return y

# # ---------------------------------------------------------#
# # 门控融合机制 (Gated Fusion) + 尺度对齐 (Zero-init Residual)
# # ---------------------------------------------------------#
# class GatedFusion(nn.Module):
#     def __init__(self, channels):
#         super(GatedFusion, self).__init__()
#         # 1x1卷积将拼接后的特征压缩回原通道数，然后接Sigmoid生成0~1的空间+通道门控权重图
#         self.gate = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(channels),
#             nn.Sigmoid()
#         )
#         # 核心创新点：独立的可学习尺度对齐参数，初始化为 0
#         self.alpha = nn.Parameter(torch.zeros(1))

#     def forward(self, x_cnn, x_dino):
#         # 1. 在通道维度拼接
#         cat_feat = torch.cat([x_cnn, x_dino], dim=1)
#         # 2. 生成门控权重图 (Gate)
#         g = self.gate(cat_feat)
#         # 3. 门控筛选 DINO 特征后，乘以 alpha 尺度对齐，并以残差形式加到 CNN 特征上
#         # 初始时 alpha 为 0，out 完美等于 x_cnn
#         out = x_cnn + self.alpha * (g * x_dino)
#         return out


# # -------------------------------------------------------------------#
# #   MobileNetV2 主干网络 (门控机制 + 尺度对齐版)
# # -------------------------------------------------------------------#
# class MobileNetV2(nn.Module):
#     def __init__(self, downsample_factor=8, pretrained=True):
#         super(MobileNetV2, self).__init__()
#         from functools import partial
        
#         model           = mobilenetv2(pretrained)
#         self.features   = model.features[:-1]

#         self.total_idx  = len(self.features)
#         self.down_idx   = [2, 4, 7, 14]

#         # ----------------控制开关----------------
#         self.Fusion_Gate = True 
        
#         self.dino_layers = [2, 5, 8, 11]

#         # ----------------初始化四个阶段的门控融合模块----------------
#         # 对应 MobileNetV2 在这四个特征层的通道数分别是 24, 64, 96, 160
#         # (模块内部自带了 alpha 参数，负责对齐与零初始化)
#         self.gate1 = GatedFusion(24)
#         self.gate2 = GatedFusion(64)
#         self.gate3 = GatedFusion(96)
#         self.gate4 = GatedFusion(160)

#         # ----------------配置空洞卷积----------------
#         if downsample_factor == 8:
#             for i in range(self.down_idx[-2], self.down_idx[-1]):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=2)
#                 )
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=4)
#                 )
#         elif downsample_factor == 16:
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=2)
#                 )
        
#     def _nostride_dilate(self, m, dilate):
#         classname = m.__class__.__name__
#         if classname.find('Conv') != -1:
#             if m.stride == (2, 2):
#                 m.stride = (1, 1)
#                 if m.kernel_size == (3, 3):
#                     m.dilation = (dilate//2, dilate//2)
#                     m.padding = (dilate//2, dilate//2)
#             else:
#                 if m.kernel_size == (3, 3):
#                     m.dilation = (dilate, dilate)
#                     m.padding = (dilate, dilate)

#     def forward(self, x):
#         _, _, H, W = x.shape
        
#         # 根据 DINOv2 的 Patch Size (14) 调整尺寸
#         new_H = H // 14 * 14
#         new_W = W // 14 * 14
        
#         with torch.no_grad():
#             upsampled_tensor = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
#             # 获取 DINOv2 对应层的特征
#             dino = _get_dinov2_vitb14().get_intermediate_layers(upsampled_tensor.cuda(), self.dino_layers)

#         # ========================================================
#         #   门控融合 (生成权重 + DINO筛选 + alpha残差对齐)
#         # ========================================================
#         if self.Fusion_Gate:
            
#             # -------- 第 1 融合层 --------
#             x_2 = self.features[:3](x)
#             D0 = UpScale(dino[0], x_2, token_hw=token_hw)
            
#             # 使用自带 alpha 对齐的 GatedFusion 模块进行融合
#             x_2 = self.gate1(x_2, D0)  
            
#             low_level_features = self.features[3:4](x_2) # 提取给 Decoder 使用的浅层特征
            
#             # -------- 第 2 融合层 --------
#             x_7 = self.features[4:8](low_level_features)
#             D1 = UpScale(dino[1], x_7, token_hw=token_hw)
            
#             x_7 = self.gate2(x_7, D1)  
            
#             # -------- 第 3 融合层 --------
#             x_11 = self.features[8:12](x_7)
#             D2 = UpScale(dino[2], x_11, token_hw=token_hw)
            
#             x_11 = self.gate3(x_11, D2)  
            
#             # -------- 第 4 融合层 --------
#             x_14 = self.features[12:15](x_11)
#             D3 = UpScale(dino[3], x_14, token_hw=token_hw)
            
#             x_14 = self.gate4(x_14, D3)  
            
#             # 将最终的高层特征输出给 ASPP 模块
#             x = self.features[15:](x_14)

#         else:
#             # 安全备份逻辑
#             low_level_features = self.features[:4](x)
#             x = self.features[4:](low_level_features)

#         return low_level_features, x 


# #-----------------------------------------#
# #   ASPP特征提取模块
# #   利用不同膨胀率的膨胀卷积进行特征提取
# #-----------------------------------------#
# class ASPP(nn.Module):
#     def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
#         super(ASPP, self).__init__()
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         self.branch2 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),  
#         )
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),  
#         )
#         self.branch4 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),  
#         )
#         self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
#         self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
#         self.branch5_relu = nn.ReLU(inplace=True)

#         self.conv_cat = nn.Sequential(
#             nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),      
#         )

#     def forward(self, x):
#         [b, c, row, col] = x.size()
#         conv1x1 = self.branch1(x)
#         conv3x3_1 = self.branch2(x)
#         conv3x3_2 = self.branch3(x)
#         conv3x3_3 = self.branch4(x)
        
#         global_feature = torch.mean(x, 2, True)
#         global_feature = torch.mean(global_feature, 3, True)
#         global_feature = self.branch5_conv(global_feature)
#         global_feature = self.branch5_bn(global_feature)
#         global_feature = self.branch5_relu(global_feature)
#         global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        
#         feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
#         result = self.conv_cat(feature_cat)
#         return result


# #-----------------------------------------#
# #   DeepLabV3+ 主模型
# #-----------------------------------------#
# class DeepLab(nn.Module):
#     def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
#         super(DeepLab, self).__init__()
        
#         if backbone == "xception":
#             self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
#             in_channels = 2048
#             low_level_channels = 256
#         elif backbone == "mobilenet":
#             # MobileNetV2 内部已硬编码为带有 Alpha 对齐的门控 (Gated) 融合
#             self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
#             in_channels = 320
#             low_level_channels = 24
#         else:
#             raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

#         #-----------------------------------------#
#         #   ASPP 特征提取模块
#         #-----------------------------------------#
#         self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
#         #----------------------------------#
#         #   浅层特征边 (Decoder)
#         #----------------------------------#
#         self.shortcut_conv = nn.Sequential(
#             nn.Conv2d(low_level_channels, 48, 1),
#             nn.BatchNorm2d(48),
#             nn.ReLU(inplace=True)
#         )       

#         self.cat_conv = nn.Sequential(
#             nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),

#             nn.Conv2d(256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),

#             nn.Dropout(0.1),
#         )
#         self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

#     def forward(self, x):
#         H, W = x.size(2), x.size(3)
        
#         low_level_features, x = self.backbone(x)
#         x = self.aspp(x)
#         low_level_features = self.shortcut_conv(low_level_features)
        
#         x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
#         x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
#         x = self.cls_conv(x)
#         x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
#         return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from nets.xception import xception
# from nets.mobilenetv2 import mobilenetv2

# # 导入自注意力模块 (DANet 双重注意力)
# from nets.attention import Danet_PositAttention, Danet_ChannAttention

# import torchvision.transforms as T
# import cv2
# import numpy as np

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transform = T.Compose([
#     T.GaussianBlur(9, sigma=(0.1, 2.0)),
#     T.Resize((36 * 14, 36 * 14)),
#     T.CenterCrop((36 * 14, 36 * 14)),
#     T.ToTensor(),
#     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
# ])
 
# from utils.fea_upscale import UpScale

# # 加载 DINOv2 模型
# _DINOV2_VITB14 = None
#
# def _get_dinov2_vitb14():
#     global _DINOV2_VITB14
#     if _DINOV2_VITB14 is None:
#         _DINOV2_VITB14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
#     return _DINOV2_VITB14

# # ---------------------------------------------------------#
# # 门控融合机制 (Gated Fusion) - 纯净版
# # 不再包含 alpha，纯粹只负责接收特征并计算门控
# # ---------------------------------------------------------#
# class GatedFusion(nn.Module):
#     def __init__(self, channels):
#         super(GatedFusion, self).__init__()
#         # 1x1卷积将拼接后的特征压缩回原通道数，接Sigmoid生成0~1的门控权重
#         self.gate = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(channels),
#             nn.Sigmoid()
#         )

#     def forward(self, x_cnn, x_dino_aligned):
#         # 1. 此时传进来的 x_dino_aligned 已经是经过 alpha 缩放的了
#         cat_feat = torch.cat([x_cnn, x_dino_aligned], dim=1)
#         # 2. 生成门控权重图 (Gate)
#         g = self.gate(cat_feat)
#         # 3. 门控筛选对齐后的 DINO 特征，并以残差形式加到 CNN 特征上
#         out = x_cnn + g * x_dino_aligned
#         return out


# # -------------------------------------------------------------------#
# #   MobileNetV2 主干网络 (先尺度对齐 + 门控融合 + DANet自注意力)
# # -------------------------------------------------------------------#
# class MobileNetV2(nn.Module):
#     def __init__(self, downsample_factor=8, pretrained=True):
#         super(MobileNetV2, self).__init__()
#         from functools import partial
        
#         model           = mobilenetv2(pretrained)
#         self.features   = model.features[:-1]

#         self.total_idx  = len(self.features)
#         self.down_idx   =[2, 4, 7, 14]

#         # ----------------控制开关----------------
#         self.Fusion_Gate = True 
        
#         self.dino_layers =[2, 5, 8, 11]

#         # ---------------- 核心创新：独立的可学习尺度对齐参数 ----------------
#         # 每一层初始化为 0，前置对齐，保证网络平滑退化为纯 CNN
#         self.alpha1 = nn.Parameter(torch.zeros(1))
#         self.alpha2 = nn.Parameter(torch.zeros(1))
#         self.alpha3 = nn.Parameter(torch.zeros(1))
#         self.alpha4 = nn.Parameter(torch.zeros(1))

#         # ---------------- 1. 初始化门控融合模块 ----------------
#         self.gate1 = GatedFusion(24)
#         self.gate2 = GatedFusion(64)
#         self.gate3 = GatedFusion(96)
#         self.gate4 = GatedFusion(160)

#         # ---------------- 2. 初始化自注意力模块 (DANet) ----------------
#         # 空间/位置注意力 (Positional Attention)
#         self.pam7  = Danet_PositAttention(64)
#         self.pam11 = Danet_PositAttention(96)
#         self.pam14 = Danet_PositAttention(160)
#         self.pamx  = Danet_PositAttention(320)

#         # 通道注意力 (Channel Attention)
#         self.cam7  = Danet_ChannAttention()
#         self.cam11 = Danet_ChannAttention()
#         self.cam14 = Danet_ChannAttention()
#         self.camx  = Danet_ChannAttention()

#         # ----------------配置空洞卷积----------------
#         if downsample_factor == 8:
#             for i in range(self.down_idx[-2], self.down_idx[-1]):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=2)
#                 )
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=4)
#                 )
#         elif downsample_factor == 16:
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=2)
#                 )
        
#     def _nostride_dilate(self, m, dilate):
#         classname = m.__class__.__name__
#         if classname.find('Conv') != -1:
#             if m.stride == (2, 2):
#                 m.stride = (1, 1)
#                 if m.kernel_size == (3, 3):
#                     m.dilation = (dilate//2, dilate//2)
#                     m.padding = (dilate//2, dilate//2)
#             else:
#                 if m.kernel_size == (3, 3):
#                     m.dilation = (dilate, dilate)
#                     m.padding = (dilate, dilate)

#     def forward(self, x):
#         _, _, H, W = x.shape
        
#         # 根据 DINOv2 的 Patch Size (14) 调整尺寸
#         new_H = H // 14 * 14
#         new_W = W // 14 * 14
        
#         with torch.no_grad():
#             upsampled_tensor = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
#             dino = _get_dinov2_vitb14().get_intermediate_layers(upsampled_tensor.cuda(), self.dino_layers)

#         # ========================================================
#         #   前向传播：先Alpha对齐 -> 再门控融合 -> 最后双重自注意力
#         # ========================================================
#         if self.Fusion_Gate:
            
#             # -------- 第 1 融合层 --------
#             x_2 = self.features[:3](x)
#             D0 = UpScale(dino[0], x_2, token_hw=token_hw)
            
#             # 1. 先进行 Alpha 尺度对齐
#             D0_aligned = D0 * self.alpha1
#             # 2. 将对齐后的特征送入门控模块
#             x_2 = self.gate1(x_2, D0_aligned)  
            
#             low_level_features = self.features[3:4](x_2) # 提取给 Decoder 使用的浅层特征
            
#             # -------- 第 2 融合层 --------
#             x_7 = self.features[4:8](low_level_features)
#             D1 = UpScale(dino[1], x_7, token_hw=token_hw)
            
#             # 1. 先进行 Alpha 尺度对齐
#             D1_aligned = D1 * self.alpha2
#             # 2. 门控融合
#             x_7 = self.gate2(x_7, D1_aligned)  
#             # 3. 自注意力强化 (空间 + 通道)
#             x_7f = self.pam7(x_7)
#             x_7s = self.cam7(x_7)
#             x_7 = x_7f + x_7s
            
#             # -------- 第 3 融合层 --------
#             x_11 = self.features[8:12](x_7)
#             D2 = UpScale(dino[2], x_11, token_hw=token_hw)
            
#             D2_aligned = D2 * self.alpha3
#             x_11 = self.gate3(x_11, D2_aligned)  
            
#             x_11f = self.pam11(x_11)
#             x_11s = self.cam11(x_11)
#             x_11 = x_11f + x_11s
            
#             # -------- 第 4 融合层 --------
#             x_14 = self.features[12:15](x_11)
#             D3 = UpScale(dino[3], x_14, token_hw=token_hw)
            
#             D3_aligned = D3 * self.alpha4
#             x_14 = self.gate4(x_14, D3_aligned)  
            
#             x_14f = self.pam14(x_14)
#             x_14s = self.cam14(x_14)
#             x_14 = x_14f + x_14s
            
#             # -------- 最终层 --------
#             x = self.features[15:](x_14)
            
#             x1 = self.pamx(x)
#             x2 = self.camx(x)
#             x = x1 + x2

#         else:
#             # 安全备份逻辑
#             low_level_features = self.features[:4](x)
#             x = self.features[4:](low_level_features)

#         return low_level_features, x 


# #-----------------------------------------#
# #   ASPP特征提取模块
# #-----------------------------------------#
# class ASPP(nn.Module):
# 	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
# 		super(ASPP, self).__init__()
# 		self.branch1 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),
# 		)
# 		self.branch2 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch3 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch4 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),	
# 		)
# 		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
# 		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
# 		self.branch5_relu = nn.ReLU(inplace=True)

# 		self.conv_cat = nn.Sequential(
# 				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),		
# 		)

# 	def forward(self, x):
# 		[b, c, row, col] = x.size()
# 		conv1x1 = self.branch1(x)
# 		conv3x3_1 = self.branch2(x)
# 		conv3x3_2 = self.branch3(x)
# 		conv3x3_3 = self.branch4(x)
		
# 		global_feature = torch.mean(x, 2, True)
# 		global_feature = torch.mean(global_feature, 3, True)
# 		global_feature = self.branch5_conv(global_feature)
# 		global_feature = self.branch5_bn(global_feature)
# 		global_feature = self.branch5_relu(global_feature)
# 		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
# 		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
# 		result = self.conv_cat(feature_cat)
# 		return result

# #-----------------------------------------#
# #   DeepLabV3+ 主模型
# #-----------------------------------------#
# class DeepLab(nn.Module):
#     def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
#         super(DeepLab, self).__init__()
        
#         if backbone == "xception":
#             self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
#             in_channels = 2048
#             low_level_channels = 256
#         elif backbone == "mobilenet":
#             # MobileNetV2 目前集成了[Alpha前置对齐] -> [门控筛选] -> [DANet自注意力]
#             self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
#             in_channels = 320
#             low_level_channels = 24
#         else:
#             raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

#         self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
#         self.shortcut_conv = nn.Sequential(
#             nn.Conv2d(low_level_channels, 48, 1),
#             nn.BatchNorm2d(48),
#             nn.ReLU(inplace=True)
#         )		

#         self.cat_conv = nn.Sequential(
#             nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),

#             nn.Conv2d(256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),

#             nn.Dropout(0.1),
#         )
#         self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

#     def forward(self, x):
#         H, W = x.size(2), x.size(3)
        
#         low_level_features, x = self.backbone(x)
#         x = self.aspp(x)
#         low_level_features = self.shortcut_conv(low_level_features)
        
#         x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
#         x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
#         x = self.cls_conv(x)
#         x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
#         return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from nets.xception import xception
# from nets.mobilenetv2 import mobilenetv2

# # 导入你的交叉注意力模块
# from nets.my_attention import Attention_cross

# import torchvision.transforms as T
# import cv2
# import numpy as np

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transform = T.Compose([
#     T.GaussianBlur(9, sigma=(0.1, 2.0)),
#     T.Resize((36 * 14, 36 * 14)),
#     T.CenterCrop((36 * 14, 36 * 14)),
#     T.ToTensor(),
#     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
# ])
 
# from utils.fea_upscale import UpScale

# # 加载 DINOv2 模型
# _DINOV2_VITB14 = None
#
# def _get_dinov2_vitb14():
#     global _DINOV2_VITB14
#     if _DINOV2_VITB14 is None:
#         _DINOV2_VITB14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
#     return _DINOV2_VITB14

# def CONV(x, in_channels, out_channels):
#     conv2_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True).cuda()
#     y = conv2_layer(x)
#     return y

# # -------------------------------------------------------------------#
# #   MobileNetV2 主干网络 (交叉注意力 + 前置 Alpha 尺度对齐版)
# # -------------------------------------------------------------------#
# class MobileNetV2(nn.Module):
#     def __init__(self, downsample_factor=8, pretrained=True):
#         super(MobileNetV2, self).__init__()
#         from functools import partial
        
#         model           = mobilenetv2(pretrained)
#         self.features   = model.features[:-1]

#         self.total_idx  = len(self.features)
#         self.down_idx   =[2, 4, 7, 14]

#         # ----------------控制开关----------------
#         self.Fusion_CrossAttention = True 
        
#         self.dino_layers =[2, 5, 8, 11]
        
#         # ----------------核心创新点：独立的可学习尺度对齐参数----------------
#         # 初始化为 1 (ones)，完美避开前置 alpha 导致的 Softmax 失效问题！
#         # 初期完全信任 DINO 特征，后续让网络自己学习缩放比例
#         self.alpha1 = nn.Parameter(torch.ones(1))
#         self.alpha2 = nn.Parameter(torch.ones(1))
#         self.alpha3 = nn.Parameter(torch.ones(1))
#         self.alpha4 = nn.Parameter(torch.ones(1))
        
#         # ----------------初始化四个融合层的交叉注意力模块----------------
#         self.A1 = Attention_cross(24)
#         self.A2 = Attention_cross(64)
#         self.A3 = Attention_cross(96)
#         self.A4 = Attention_cross(160)

#         # ----------------配置空洞卷积----------------
#         if downsample_factor == 8:
#             for i in range(self.down_idx[-2], self.down_idx[-1]):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=2)
#                 )
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=4)
#                 )
#         elif downsample_factor == 16:
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(
#                     partial(self._nostride_dilate, dilate=2)
#                 )
        
#     def _nostride_dilate(self, m, dilate):
#         classname = m.__class__.__name__
#         if classname.find('Conv') != -1:
#             if m.stride == (2, 2):
#                 m.stride = (1, 1)
#                 if m.kernel_size == (3, 3):
#                     m.dilation = (dilate//2, dilate//2)
#                     m.padding = (dilate//2, dilate//2)
#             else:
#                 if m.kernel_size == (3, 3):
#                     m.dilation = (dilate, dilate)
#                     m.padding = (dilate, dilate)

#     def forward(self, x):
#         _, _, H, W = x.shape
        
#         # 根据 DINOv2 的 Patch Size (14) 调整尺寸
#         new_H = H // 14 * 14
#         new_W = W // 14 * 14
        
#         with torch.no_grad():
#             upsampled_tensor = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
#             # 获取 DINOv2 对应层的特征
#             dino = _get_dinov2_vitb14().get_intermediate_layers(upsampled_tensor.cuda(), self.dino_layers)

#         # ========================================================
#         #   交叉注意力融合 (Cross Attention Fusion) + 前置尺度对齐
#         # ========================================================
#         if self.Fusion_CrossAttention:
            
#             # -------- 第 1 融合层 --------
#             x_2 = self.features[:3](x)
#             D0 = UpScale(dino[0], x_2, token_hw=token_hw)
            
#             # 1. 【前置 Alpha 对齐】：初始为1，保留完整特征分布
#             D0_aligned = D0 * self.alpha1 
#             # 2. 【交叉注意力融合】：因为 D0_aligned 含有正常信息，注意力矩阵一切正常
#             x_2 = self.A1(x_2, D0_aligned)  
            
#             low_level_features = self.features[3:4](x_2) # 提取给 Decoder 使用的浅层特征
            
#             # -------- 第 2 融合层 --------
#             x_7 = self.features[4:8](low_level_features)
#             D1 = UpScale(dino[1], x_7, token_hw=token_hw)
            
#             D1_aligned = D1 * self.alpha2
#             x_7 = self.A2(x_7, D1_aligned)  
            
#             # -------- 第 3 融合层 --------
#             x_11 = self.features[8:12](x_7)
#             D2 = UpScale(dino[2], x_11, token_hw=token_hw)
            
#             D2_aligned = D2 * self.alpha3
#             x_11 = self.A3(x_11, D2_aligned)  
            
#             # -------- 第 4 融合层 --------
#             x_14 = self.features[12:15](x_11)
#             D3 = UpScale(dino[3], x_14, token_hw=token_hw)
            
#             D3_aligned = D3 * self.alpha4
#             x_14 = self.A4(x_14, D3_aligned)  
            
#             # 将最终的高层特征输出给 ASPP 模块
#             x = self.features[15:](x_14)

#         else:
#             # 安全备份逻辑 
#             low_level_features = self.features[:4](x)
#             x = self.features[4:](low_level_features)

#         return low_level_features, x 


# #-----------------------------------------#
# #   ASPP特征提取模块
# #-----------------------------------------#
# class ASPP(nn.Module):
#     def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
#         super(ASPP, self).__init__()
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         self.branch2 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),  
#         )
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),  
#         )
#         self.branch4 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),  
#         )
#         self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
#         self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
#         self.branch5_relu = nn.ReLU(inplace=True)

#         self.conv_cat = nn.Sequential(
#             nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),      
#         )

#     def forward(self, x):
#         [b, c, row, col] = x.size()
#         conv1x1 = self.branch1(x)
#         conv3x3_1 = self.branch2(x)
#         conv3x3_2 = self.branch3(x)
#         conv3x3_3 = self.branch4(x)
        
#         global_feature = torch.mean(x, 2, True)
#         global_feature = torch.mean(global_feature, 3, True)
#         global_feature = self.branch5_conv(global_feature)
#         global_feature = self.branch5_bn(global_feature)
#         global_feature = self.branch5_relu(global_feature)
#         global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        
#         feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
#         result = self.conv_cat(feature_cat)
#         return result

# #-----------------------------------------#
# #   DeepLabV3+ 主模型
# #-----------------------------------------#
# class DeepLab(nn.Module):
#     def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
#         super(DeepLab, self).__init__()
        
#         if backbone == "xception":
#             self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
#             in_channels = 2048
#             low_level_channels = 256
#         elif backbone == "mobilenet":
#             # MobileNetV2 内部已硬编码为 前置Alpha(初始化为1) 的交叉注意力融合
#             self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
#             in_channels = 320
#             low_level_channels = 24
#         else:
#             raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

#         self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
#         self.shortcut_conv = nn.Sequential(
#             nn.Conv2d(low_level_channels, 48, 1),
#             nn.BatchNorm2d(48),
#             nn.ReLU(inplace=True)
#         )       

#         self.cat_conv = nn.Sequential(
#             nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),

#             nn.Conv2d(256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),

#             nn.Dropout(0.1),
#         )
#         self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

#     def forward(self, x):
#         H, W = x.size(2), x.size(3)
        
#         low_level_features, x = self.backbone(x)
#         x = self.aspp(x)
#         low_level_features = self.shortcut_conv(low_level_features)
        
#         x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
#         x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
#         x = self.cls_conv(x)
#         x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2

# 导入你的交叉注意力模块
from nets.my_attention import Attention_cross

import torchvision.transforms as T
import cv2
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.GaussianBlur(9, sigma=(0.1, 2.0)),
    T.Resize((36 * 14, 36 * 14)),
    T.CenterCrop((36 * 14, 36 * 14)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
 
from utils.fea_upscale import UpScale

# 加载 DINOv2 模型
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
        
        # 根据 DINOv2 的 Patch Size (14) 调整尺寸
        new_H = H // 14 * 14
        new_W = W // 14 * 14
        
        token_hw = (new_H // 14, new_W // 14)
        with torch.no_grad():
            upsampled_tensor = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
            # 获取 DINOv2 对应层的特征
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





