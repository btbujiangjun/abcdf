"""
CNN Building Blocks with Advanced Normalization Techniques

This module implements various CNN layers, including upsampling, downsampling, 
residual connections, and weight-standardized convolutions.

Author: Jiang Jun
Date: 2025-02-11
"""

import torch
import torch.nn as nn


class UpSampleLayer(nn.Module):
    """
    Upsampling Layer:
        1) Uses nearest-neighbor interpolation to scale feature maps by 2×.
        2) Applies a 3×3 convolution to adjust the number of channels.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int, optional): Number of output channels. Defaults to in_dim.
    """
    def __init__(self, in_dim:int, out_dim=None):
        super().__init__()
        self.sampler = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_dim, out_dim or in_dim, 3, padding=1)
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.sampler(x)

class DownSampleLayer(nn.Module):
    """
    Downsampling Layer:
        1) Reduces spatial resolution by 2× using pixel shuffling.
        2) Applies a 1×1 convolution to adjust the number of channels.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int, optional): Number of output channels. Defaults to in_dim.
    """
    def __init__(self, in_dim:int, out_dim=None):
        super().__init__()
        self.p1, self.p2 = 2, 2
        self.sampler = nn.Conv2d(in_dim * 4, out_dim or in_dim, 1)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        #Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2)
        b, c, hp, wp = x.shape
        h, w = hp // self.p1, wp // self.p2
        # 1)拆分维度[b, c, h, p1, w, p2]
        x = x.view(b, c, h, self.p1, w, self.p2)
        # 2)调整维度[b, c, p1, p2, h, w]
        x = x.permute(0, 1, 3, 5, 2, 4)
        # 3)合并维度[b, c * p1 * p2, h, w]
        x = x.contiguous().view(b, c * self.p1 * self.p2, h, w)
        return self.sampler(x)

class ResidualLayer(nn.Module):
    """
    Residual Connection Layer.

    This wraps a given function `fn` such that the input is added 
    to the output, implementing a residual connection.

    Args:
        fn (nn.Module): A function/layer to apply to the input.
    """
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x:torch.Tensor, *args, **kwargs)->torch.Tensor:
        return self.fn(x, *args, **kwargs) + x


class WeightStandardizedConv2d(nn.Conv2d):
    """
    Weight Standardized Convolution (https://arxiv.org/abs/1903.10520).

    Weight standardization normalizes the kernel weights before convolution, 
    which can improve training stability when combined with Group Normalization.

    Args:
        Same as `torch.nn.Conv2d`.
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight

        # Compute mean and variance across (out_channels, in_channels, kernel_h, kernel_w)
        mean = weight.mean(
            dim=tuple(range(1, weight.dim())), 
            keepdim=True
        )
        var = weight.var(
            dim=tuple(range(1, weight.dim())), 
            unbiased=False, 
            keepdim=True
        )
        
        # Normalize weights
        norm_weight = (weight - mean) * (var + eps).rsqrt()
        return nn.functional.conv2d(x, 
            norm_weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )



