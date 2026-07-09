# import torch

# from torch import Tensor
# from torch import nn
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def transpose_qkv(X,num_heads):
#     X=X.reshape(X.shape[0],X.shape[1],num_heads,-1)#补充，一般num_hidden指的是q,k,v的最后维度大小
#     X=X.permute(0,2,1,3)
#     return X.reshape(-1,X.shape[2],X.shape[3])#把前两维合在一起了

# def transpose_output(X,num_heads):
#     X=X.reshape(-1,num_heads,X.shape[1],X.shape[2])
#     X=X.permute(0,2,1,3)
#     return X.reshape(X.shape[0],X.shape[1],-1)

# class Attention_cross(nn.Module):
#     def __init__(
#         self,
#         in_channels: int
        
#     ) -> None:
#         super().__init__()

#         self.in_channels = in_channels
#         self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
#         self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
#         self.value = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)
#         self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
#         self.softmax = nn.Softmax(dim = -1)
        
        

#     def forward(self, input1: Tensor, input2:Tensor) -> Tensor:
        
#         batch_size, channels, height, width = input1.size()
        
#         #concat = torch.cat((input1, input2), dim=1)
#         conv2_layer=nn.Conv2d(input2.shape[1], channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True).cuda()
#         Con_input=conv2_layer(input2)
        

#         # input: B, C, H, W -> q: B, H * W, C // 8
#         q = self.query(input1).view(batch_size, -1, height * width).permute(0, 2, 1)#转置

#         #input: B, C, H, W -> k: B, C // 8, H * W
#         k = self.key(input1).view(batch_size, -1, height * width)

#         #input: B, C, H, W -> v: B, C, H * W
#         v = self.value(Con_input).view(batch_size, -1, height * width)

#         #q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
#         attn_matrix = torch.bmm(q, k)  #torch.bmm进行tensor矩阵乘法
#         attn_matrix = self.softmax(attn_matrix)#softmax
#         out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
#         out = out.view(*input1.shape)
 
#         return self.gamma * out + input1
    
import torch
from torch import Tensor
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Attention_cross(nn.Module):
    def __init__(self, in_channels: int, dino_channels: int = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        
        # 如果不知道 DINO 的通道数，默认它和 in_channels 已经对齐
        if dino_channels is None:
            dino_channels = in_channels
            
        # 1. 致命Bug修复：将通道对齐的 1x1 卷积移到 __init__ 中，使其可以被训练！
        # 这对应 SCI 论文公式中的 Conv1x1
        self.dino_proj = nn.Conv2d(dino_channels, in_channels, kernel_size=1, stride=1)

        # 2. 定义 Q, K, V 的线性映射层
        # 降维处理 (in_channels // 8) 减少计算量，这是标准做法
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        
        # 3. 零初始化残差参数 (与你在 MobileNetV2 里的 alpha 配合/或者替代)
        self.gamma = nn.Parameter(torch.zeros(1))  
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        # input1: CNN 特征
        # input2: DINO 特征 (可能已经被你的 alpha 对齐过了)
        batch_size, channels, height, width = input1.size()
        
        # 1. 对 DINO 特征进行通道对齐投影
        aligned_dino = self.dino_proj(input2)

        # 2. 生成 Query (来自 CNN)
        # q: B, C//8, H*W -> B, H*W, C//8 (通过 permute 转置准备做矩阵乘法)
        q = self.query(input1).view(batch_size, -1, height * width).permute(0, 2, 1)

        # 3. 生成 Key 和 Value (全部来自投影后的 DINO)
        # 这完美契合我们刚写的 SCI 公式: K = F_VFM * W_K, V = F_VFM * W_V
        k = self.key(aligned_dino).view(batch_size, -1, height * width)    # B, C//8, H*W
        v = self.value(aligned_dino).view(batch_size, -1, height * width)  # B, C, H*W

        # 4. 计算缩放点积注意力 (Cross-Attention Map)
        # q (B, H*W, C//8) 乘以 k (B, C//8, H*W) -> attn_matrix (B, H*W, H*W)
        attn_matrix = torch.bmm(q, k)
        
        # (可选但推荐的步骤)：缩放因子 sqrt(d_k) 防止梯度消失，让 softmax 更平滑
        d_k = q.size(-1)
        attn_matrix = attn_matrix / (d_k ** 0.5) 
        
        attn_matrix = self.softmax(attn_matrix)

        # 5. 注意力特征聚合
        # v (B, C, H*W) 乘以 attn_matrix 的转置 (B, H*W, H*W) -> out (B, C, H*W)
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  
        out = out.view(batch_size, channels, height, width)
 
        # 6. 零初始化残差连接
        return out + input1