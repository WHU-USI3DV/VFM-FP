"""Cross-attention fusion block used by the VCFS backbone."""

import torch
from torch import Tensor
from torch import nn


class Attention_cross(nn.Module):
    """Fuse CNN features with projected DINOv2 features at one feature scale."""

    def __init__(self, in_channels: int, dino_channels: int = None) -> None:
        super().__init__()
        self.in_channels = in_channels

        if dino_channels is None:
            dino_channels = in_channels

        # Keep this projection registered so it is trained and checkpointed.
        self.dino_proj = nn.Conv2d(dino_channels, in_channels, kernel_size=1, stride=1)

        attention_channels = max(1, in_channels // 8)
        self.query = nn.Conv2d(in_channels, attention_channels, kernel_size=1, stride=1)
        self.key   = nn.Conv2d(in_channels, attention_channels, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.scale = attention_channels ** -0.5
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        """Return residual cross-attention output.

        Args:
            input1: CNN feature map, shaped ``B x C x H x W``.
            input2: DINOv2 feature map aligned to the same spatial scale.
        """
        batch_size, channels, height, width = input1.size()
        aligned_dino = self.dino_proj(input2)

        q = self.query(input1).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(aligned_dino).view(batch_size, -1, height * width)
        v = self.value(aligned_dino).view(batch_size, -1, height * width)
        
        attn_matrix = torch.bmm(q, k) * self.scale
        attn_matrix = self.softmax(attn_matrix)

        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  
        out = out.view(batch_size, channels, height, width)
 
        # Residual fusion follows Eq. (5) in the paper. Scale alignment is
        # controlled by the per-level alpha parameter in the backbone.
        return out + input1
