"""Trainable spatial and channel alignment for DINOv2 token features."""

import math

import torch.nn as nn
import torch.nn.functional as F


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


class DINOFeatureProjection(nn.Module):
    """Resize DINO tokens and project them to one CNN feature scale."""

    def __init__(self, dino_channels, output_channels, bn_momentum=0.1):
        super().__init__()
        self.dino_channels = int(dino_channels)
        self.output_channels = int(output_channels)
        self.channel_transform = nn.Conv2d(
            self.dino_channels,
            self.output_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.batch_norm = nn.BatchNorm2d(self.output_channels, momentum=bn_momentum)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, origin_dino, deep, token_hw=None):
        batch_size, cnn_channels, cnn_h, cnn_w = deep.shape
        dino_batch, num_tokens, dino_channels = origin_dino.shape

        if batch_size != dino_batch:
            raise ValueError("Batch size mismatch between DINO features and CNN features.")
        if cnn_channels != self.output_channels:
            raise ValueError(
                "CNN feature channels {} do not match projection output channels {}.".format(
                    cnn_channels, self.output_channels
                )
            )
        if dino_channels != self.dino_channels:
            raise ValueError(
                "DINO feature channels {} do not match projection input channels {}.".format(
                    dino_channels, self.dino_channels
                )
            )

        token_h, token_w = _infer_token_hw(
            num_tokens,
            (cnn_h, cnn_w),
            token_hw=token_hw,
        )
        dino = origin_dino.permute(0, 2, 1).reshape(
            dino_batch,
            dino_channels,
            token_h,
            token_w,
        )
        dino = dino.to(device=deep.device, dtype=deep.dtype)
        dino = F.interpolate(
            dino,
            size=(cnn_h, cnn_w),
            mode="bilinear",
            align_corners=False,
        )
        return self.activation(self.batch_norm(self.channel_transform(dino)))
