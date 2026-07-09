import os
import time
import torch
from thop import profile, clever_format
from nets.deeplabv3_plus import DeepLab

# 设置只使用一张显卡（测试速度和显存通常在单卡下进行）
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def evaluate_model_complexity():
    # --------------------------------------------------#
    #   1. 初始化参数 (请保持与你 train.py 中一致)
    # --------------------------------------------------#
    num_classes = 7             # 你的类别数
    backbone = "mobilenet"      # 主干网络
    downsample_factor = 8       # 下采样倍数
    input_shape = [512, 512]    # 输入分辨率
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"正在评估模型: DeepLabV3+ ({backbone})")
    print(f"输入分辨率: {input_shape[0]}x{input_shape[1]}, 类别数: {num_classes}")
    
    # --------------------------------------------------#
    #   2. 构建模型并切换到评估模式 (非常重要)
    # --------------------------------------------------#
    # 注意：测试复杂度不需要加载预训练权重 pretrained=False 即可
    model = DeepLab(num_classes=num_classes, backbone=backbone, 
                    downsample_factor=downsample_factor, pretrained=False)
    model = model.to(device)
    model.eval() # 切换到预测模式，关闭 Dropout 和 BatchNorm 的更新

    # 创建一个模拟输入张量 (Batch_Size=1, Channels=3, H=512, W=512)
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)

    # ==================================================#
    #   指标 1 & 2: Parameters (参数量) & FLOPs (计算量)
    # ==================================================#
    print("\n[1/3] 计算 FLOPs 和 参数量 (Params)...")
    # MACs (乘加操作数) 通常约等于 FLOPs 的一半，很多论文中将 MACs * 2 作为 FLOPs
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    flops = macs * 2 
    
    # 格式化输出为 G (Giga) 和 M (Million)
    flops_str, params_str = clever_format([flops, params], "%.2f")
    print(f" => Parameters: {params / 1e6:.2f} M ({params_str})")
    print(f" => FLOPs:      {flops / 1e9:.2f} G ({flops_str})")


    # ==================================================#
    #   指标 3: GPU Memory Usage (显存占用)
    # ==================================================#
    print("\n[2/3] 计算推理显存占用 (GPU Memory)...")
    torch.cuda.empty_cache() # 清空没用的显存
    torch.cuda.reset_peak_memory_stats() # 重置显存峰值统计
    
    with torch.no_grad():
        _ = model(dummy_input)
        
    max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2) # 转换为 MB
    print(f" => Max GPU Memory: {max_memory:.2f} MB")


    # ==================================================#
    #   指标 4: Inference Speed (推理速度: Time & FPS)
    # ==================================================#
    print("\n[3/3] 计算推理速度 (Inference Speed)...")
    # 速度测试前必须先预热 (Warm-up) 显卡，否则第一次跑会很慢
    warmup_iters = 50
    test_iters = 200
    
    print(f"    - 正在预热显卡 ({warmup_iters} iters)...")
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_input)
            
    print(f"    - 开始测试速度 ({test_iters} iters)...")
    torch.cuda.synchronize() # 同步GPU，确保之前的任务都跑完
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(test_iters):
            _ = model(dummy_input)
            
    torch.cuda.synchronize() # 再次同步，确保这 200 次全跑完再停止计时
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_ms = (total_time / test_iters) * 1000 # 换算成毫秒
    fps = 1.0 / (total_time / test_iters)
    
    print(f" => Average Time per image: {avg_time_ms:.2f} ms")
    print(f" => FPS (Frames Per Second): {fps:.2f} fps")
    
    print("\n=== 测试完成 ===")
    print("你可以将以上数据填入 LaTeX 表格中了！")

if __name__ == "__main__":
    evaluate_model_complexity()