import argparse
import os

from PIL import Image
from tqdm import tqdm

from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results
from utils.class_config import load_class_config

def _parse_shape(value):
    parts = [int(part.strip()) for part in value.replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("input shape must be like 512,512 or 512x512")
    return parts

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VCFS predictions with mIoU.")
    parser.add_argument("--class-config", default=None, help="Path to the dataset class JSON.")
    parser.add_argument("--dataset-path", default=None, help="VOC-style dataset directory.")
    parser.add_argument("--miou-out-path", default=None, help="Directory for prediction masks and mIoU results.")
    parser.add_argument("--miou-mode", type=int, choices=(0, 1, 2), default=0, help="0: predict and evaluate, 1: predict only, 2: evaluate only.")
    parser.add_argument("--num-classes", type=int, default=None, help="Override class count.")
    parser.add_argument("--model-path", default=None, help="Checkpoint path used for prediction.")
    parser.add_argument("--input-shape", type=_parse_shape, default=None, help="Input size, for example 512,512 or 512x512.")
    args = parser.parse_args()
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = args.miou_mode
    class_config = load_class_config(args.class_config or os.environ.get("VCFS_CLASS_CONFIG"))
    #------------------------------#
    #   分类个数+1、如2+1
    #------------------------------#
    num_classes_value = args.num_classes if args.num_classes is not None else os.environ.get("VCFS_NUM_CLASSES")
    num_classes     = int(num_classes_value or class_config["num_classes"])
    #--------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    #--------------------------------------------#
    #name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    name_classes    = class_config["classes"]
    #name_classes    = ["background","wall","window","door","balcony","roof","shop","sky","chimney"]
    # name_classes    = ["_background_","cat","dog"]
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = args.dataset_path or os.environ.get('VCFS_DATASET_PATH', 'wuhan_test')

    image_ids       = open(os.path.join(VOCdevkit_path, "txt/val.txt"),'r').read().splitlines() 
    gt_dir          = os.path.join(VOCdevkit_path, "SegmentationClass/")
    miou_out_path   = args.miou_out_path or os.environ.get("VCFS_MIOU_OUT_PATH", "facadewhu_vfm-fp_0512")
    pred_dir        = os.path.join(miou_out_path, 'detection-results')
    gt_crop_dir     = os.path.join(miou_out_path, 'Ground_True_Crop')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        if not os.path.exists(gt_crop_dir):
            os.makedirs(gt_crop_dir)
            
        print("Load model.")
        deeplab_kwargs = {"class_config": args.class_config}
        if args.num_classes is not None:
            deeplab_kwargs["num_classes"] = num_classes
        if args.model_path is not None:
            deeplab_kwargs["model_path"] = args.model_path
        if args.input_shape is not None:
            deeplab_kwargs["input_shape"] = args.input_shape
        deeplab = DeeplabV3(**deeplab_kwargs)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)

            gt_path = os.path.join(VOCdevkit_path, "SegmentationClass/"+image_id+".png")
            gt = Image.open(gt_path)

            # image,gt       = deeplab.get_miou_png(image,gt)
            image       = deeplab.get_miou_png(image)


            image.save(os.path.join(pred_dir, image_id + ".png"))
            
            # gt.save(os.path.join(gt_crop_dir, image_id + ".png"))
            gt.save(os.path.join(gt_dir, image_id + ".png"))

        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        ###lastest
        # hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_crop_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
