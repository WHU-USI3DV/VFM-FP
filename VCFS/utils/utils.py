import random
import cv2
import numpy as np
import torch
from PIL import Image

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):

    def rand(a=0, b=1):
        return np.random.rand() * (b - a) + a
    
    ###0717_resize
    iw, ih  = image.size
    w, h    = size
    image = image.resize((w, h), Image.BICUBIC)

    return image, size[0], size[1]
    
    
    ###0717_gray
    # iw, ih = image.size
    # target_w, target_h = size  # 目标尺寸，如(512, 512)
    # new_w, new_h = 1024, 1024
    
    # # 计算原始图像在1024×1024中的位置
    # scale = min(new_w/iw, new_h/ih)
    # nw_1024 = int(iw * scale)  # 原始内容在1024图像中的宽度
    # nh_1024 = int(ih * scale)  # 原始内容在1024图像中的高度

    # # 创建1024×1024的灰底图像并居中粘贴原始图像
    # image_1024 = Image.new('RGB', (new_w, new_h), (128, 128, 128))
    # image_1024.paste(image.resize((nw_1024, nh_1024), Image.BICUBIC), 
    #                 ((new_w-nw_1024)//2, (new_h-nh_1024)//2))
    
    # # 第二步：将1024×1024图像缩放到目标尺寸(如512×512)
    # image_resized = image_1024.resize((target_w, target_h), Image.BICUBIC)
    
    # # 计算原始内容在最终图像中的尺寸
    # nw_final = int(nw_1024 * target_w / new_w)
    # nh_final = int(nh_1024 * target_h / new_h)
    
    # return image_resized, nw_final, nh_final

    # iw, ih  = image.size
    # w, h    = size


    # nh=int(ih*0.5)
    # nw=int(iw*0.5)

    # new_image = Image.new('RGB', size, (128,128,128))
    # new_image.paste(image, ((1024-iw)//2, (1024-ih)//2))
    # image   = image.resize((w,h), Image.BICUBIC)

    # return image, nw, nh


    # scale   = min(w/iw, h/ih)
    # nw      = int(iw*scale)
    # nh      = int(ih*scale)

    # image   = image.resize((nw,nh), Image.BICUBIC)
    # new_image = Image.new('RGB', size, (128,128,128))
    # new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    #####facadeWHU#####
    #ori
    # resize_scale_w=1008#1024
    # resize_scale_h=504

    # w, h    = size
 
    # scale   = min(resize_scale_w/iw, resize_scale_h/ih)
    # nw      = int(iw*scale)
    # nh      = int(ih*scale)

    # image   = image.resize((resize_scale_w,resize_scale_h), Image.BICUBIC)

    # resize_scale_w=512#1024
    # resize_scale_h=512

    
    # new_image = Image.new('RGB', size, (128,128,128))
    # new_image.paste(image, ((1024-iw)//2, (1024-ih)//2))

    # dx = int(rand(0, nw - w))
    # dy = int(rand(0, nh - h))
    # image = image.crop((dx, dy, dx+w, dy+h))

    
# ###0716_gray
#     dx = int(rand(0, iw - w))
#     dy = int(rand(0, ih - h))

    #####ECP#####
    # w=512
    # h=512
    # if iw >= w and ih >= h:
    #     # 随机或中心裁剪
    #     if random:
    #         dx = int(rand(0, iw - w))
    #         dy = int(rand(0, ih - h))
    #     else:
    #         dx = (iw - w) // 2
    #         dy = (ih - h) // 2
    #     image = image.crop((dx, dy, dx + w, dy + h))
    #     label = label.crop((dx, dy, dx + w, dy + h))
    # else:
    #     # 创建灰色背景
    #     new_image = Image.new('RGB', (w, h), (128, 128, 128))
    #     new_label = Image.new('L', (w, h), 0)
    #     # 计算粘贴位置
    #     # if random:
    #     #     dx = int(rand(0, w - iw))
    #     #     dy = int(rand(0, h - ih))
    #     # else:
    #     dx = (w - iw) // 2
    #     dy = (h - ih) // 2
    #     # 粘贴到背景
    #     new_image.paste(image, (dx, dy))
    #     new_label.paste(label, (dx, dy))
        # image, label = new_image, new_label

    # return image


    # return image, nw, nh
    

def resize_img(image,size,label):

    def rand(a=0, b=1):
        return np.random.rand() * (b - a) + a
    

    ###0717_resize
    iw, ih  = image.size
    w, h    = size
    image = image.resize((w, h), Image.BICUBIC)
    label = label.resize((w, h), Image.NEAREST)
    
    # target_size

    #####FadcadeWHU#####
    ###ori
    # tw=512#512
    # th=512

###0715_crop_random
    # w,h = size

    # iw, ih    = image.size

    # dx = int(rand(0, iw - w))
    # dy = int(rand(0, ih - h))
    # image = image.crop((dx, dy, dx+w, dy+h))
    # label = label.crop((dx, dy, dx+w, dy+h))

###0716_resize
    # w,h =size
    # iw, ih    = image.size
    # image   = image.resize((w,h), Image.BICUBIC)
    # label   = label.resize((w,h), Image.NEAREST)




    # tw, th = size  # 改为传入的尺寸(512×512)
    
    # # 确定性中心裁剪
    # dx = (image.width - tw) // 2
    # dy = (image.height - th) // 2
    # image = image.crop((dx, dy, dx+tw, dy+th))
    # label = label.crop((dx, dy, dx+tw, dy+th))
    
    # # 对齐训练时的标签处理
    # label_np = np.array(label)
    # label_np[label_np >= num_classes] = num_classes
    # label = Image.fromarray(label_np)



    ####ecp_0620

    # resize_scale_w=256#1024
    # resize_scale_h=512

    # w, h    = size
    # iw,ih = image.size
 
    # scale   = min(resize_scale_w/iw, resize_scale_h/ih)
    # nw      = int(iw*scale)
    # nh      = int(ih*scale)

    # image   = image.resize((resize_scale_w,resize_scale_h), Image.BICUBIC)

    # resize_scale_w=512#1024
    # resize_scale_h=512

    
    # new_image = Image.new('RGB', size, (128,128,128))
    # new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    # dx = int(rand(0, nw - w))
    # dy = int(rand(0, nh - h))
    # image = image.crop((dx, dy, dx+w, dy+h))

    #####ECP#####
    # w=512
    # h=512

    # random = False
    # w, h=size

    # iw,ih = image.size
    # if iw >= w and ih >= h:
    #     # 随机或中心裁剪
    #     if random:
    #         dx = int(rand(0, iw - w))
    #         dy = int(rand(0, ih - h))
    #     else:
    #         dx = (iw - w) // 2
    #         dy = (ih - h) // 2
    #     image = image.crop((dx, dy, dx + w, dy + h))
    #     label = label.crop((dx, dy, dx + w, dy + h))
    # else:
    #     # 创建灰色背景
    #     new_image = Image.new('RGB', (w, h), (128, 128, 128))
    #     new_label = Image.new('L', (w, h), 0)
    #     # 计算粘贴位置
    #     # if random:
    #     #     dx = int(rand(0, w - iw))
    #     #     dy = int(rand(0, h - ih))
    #     # else:
    #     dx = (w - iw) // 2
    #     dy = (h - ih) // 2
    #     # 粘贴到背景
    #     new_image.paste(image, (dx, dy))
    #     new_label.paste(label, (dx, dy))
    #     image, label = new_image, new_label


#######ECP_2########
    # resize_w=1024
    # resize_h=1024
    # ###
    # #scale = min(resize_w/w, resize_h/h)
    # scale = min(resize_w/iw, resize_h/ih)
    # new_w, new_h = int(iw*scale), int(ih*scale)
    # #image = image.resize((new_w, new_h), Image.BICUBIC)
    # image = image.resize((new_w, new_h), Image.BICUBIC)
    # label = label.resize((new_w, new_h), Image.NEAREST)

    # # if random:
    # dx = int(rand(0, iw - w))
    # dy = int(rand(0, ih - h))
    # # else:
    # #     dx = (iw - w) // 2
    # #     dy = (ih - h) // 2
    # image = image.crop((dx, dy, dx + w, dy + h))
    # label = label.crop((dx, dy, dx + w, dy + h))

    # # if iw >= 300 and ih >= 300:
    #             # 随机或中心裁剪
                

    # # else:
    # #     print('######WRONG######')
    

    return image,label

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'mobilenet' : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
        'xception'  : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth',
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)