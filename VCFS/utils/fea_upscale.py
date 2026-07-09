import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt

# 瀹氫箟涓€涓浆缃嵎绉眰锛岀敤浜庝笂閲囨牱
class MLP(nn.Module):
    def __init__(self, input, output):
        super(MLP, self).__init__()
        self.shape=input.shape
        self.embed_dim=768
        self.drop=0.0
        # 灞曞钩鍚庣殑杈撳叆鐗瑰緛鏁伴噺
        self.flattened_input_size = self.shape[0] * self.shape[1] * self.shape[2] * self.shape[3]
        # 娣诲姞绗竴涓殣钘忓眰
        self.fc1 = nn.Linear(self.flattened_input_size, self.embed_dim,bias=True)
        # 娣诲姞杈撳嚭灞傦紝杈撳嚭缁村害涓?4
        self.fc2 = nn.Linear(self.embed_dim, output,bias=True)
        self.drop=nn.Dropout(self.drop)
        
        self.bn1 = nn.BatchNorm2d(self.embed_dim, momentum=0.1)
        self.bn2 = nn.BatchNorm2d(output, momentum=0.1)
        
    
    def forward(self, x):
        # 灞曞钩杈撳叆寮犻噺
        x = x.view(x.size(0), -1)
        # 閫氳繃绗竴涓殣钘忓眰
        x = F.relu(self.bn1(self.fc1(x)))
        x=self.drop(x)
        # 閫氳繃杈撳嚭灞?        x = F.relu(self.bn2(self.fc2(x)))  # 鍙互閫夋嫨娣诲姞婵€娲诲嚱鏁帮紝濡傞渶瑕?        x=self.drop(x)
        
        return x


def _infer_token_hw(num_tokens, deep_hw, token_hw=None):
    if token_hw is not None:
        token_h, token_w = token_hw
        if token_h * token_w != num_tokens:
            raise ValueError(
                "DINO token grid {}x{} does not match {} tokens.".format(
                    token_h, token_w, num_tokens
                )
            )
        return int(token_h), int(token_w)

    square = int(round(math.sqrt(num_tokens)))
    if square * square == num_tokens:
        return square, square

    deep_h, deep_w = deep_hw
    approx_h = max(1, int(round(math.sqrt(num_tokens * deep_h / max(deep_w, 1)))))
    candidates = sorted(range(1, num_tokens + 1), key=lambda value: abs(value - approx_h))
    for token_h in candidates:
        if num_tokens % token_h == 0:
            return token_h, num_tokens // token_h

    raise ValueError("Could not infer a valid DINO token grid for {} tokens.".format(num_tokens))


def UpScale(origin_dino, deep, token_hw=None):
    B1, C1, H1, W1 = deep.shape
    B2, HW, C2 = origin_dino.shape

    if B1 != B2:
        raise ValueError("Batch size mismatch between DINO features and CNN features.")

    token_h, token_w = _infer_token_hw(HW, (H1, W1), token_hw=token_hw)
    dino = origin_dino.permute(0, 2, 1).reshape(B2, C2, token_h, token_w)
    dino = dino.to(device=deep.device, dtype=deep.dtype)

    upsampled_tensor = F.interpolate(dino, size=(H1, W1), mode='bilinear', align_corners=False)
    upsampled_tensor = upsampled_tensor.view(B1, C2, H1, W1)

    channel_T = nn.Conv2d(C2, C1, 1, 1, padding=0, dilation=1, bias=True).to(deep.device)
    global_bn = nn.BatchNorm2d(C1, momentum=0.1).to(deep.device)
    relu = nn.ReLU(inplace=True).to(deep.device)

    global_feature = channel_T(upsampled_tensor)
    global_feature = global_bn(global_feature)
    global_feature = relu(global_feature)

    return global_feature
