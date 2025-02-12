"""
ResNet Block with Time Embedding Support

This module implements:
1. Block: A basic convolutional block with group normalization and SiLU activation.
2. ResnetBlock: A residual block with optional time-dependent embedding.

Author: Jiang Jun
Date: 2025-02-11
"""

import torch
import torch.nn as nn
from module.layer import WeightStandardizedConv2d

class Block(nn.Module):
    """
    Basic Convolutional Block with Weight Standardization.

    Args:
        in_dim (int): Input channel dimension.
        out_dim (int): Output channel dimension.
        groups (int): Number of groups for GroupNorm (default: 8).
    """
    def __init__(self, in_dim, out_dim, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(in_dim, out_dim, 3, padding=1)
        self.norm = nn.GroupNorm(groups, out_dim)
        self.activate = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            assert isinstance(scale_shift, tuple) and len(scale_shift) == 2, \
                "scale_shift must be a tuple (scale, shift)"
            scale, shift = scale_shift
            scale, shift = scale_shift
            # 从time emb得到scale/shift, 并对特性图进行缩放和偏移
            x = x * (scale + 1) + shift

        return self.activate(x)

class ResnetBlock(nn.Module):
    """
    Residual Block with Optional Time Embedding.

    Args:
        in_dim (int): Input channel dimension.
        out_dim (int): Output channel dimension.
        time_emb_dim (int, optional): Time embedding dimension (default: None).
        groups (int): Number of groups for GroupNorm (default: 8).
    """
    def __init__(self, 
            in_dim, 
            out_dim, 
            time_emb_dim=None, 
            groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_dim * 2)
        ) if time_emb_dim is not None else None

        self.block1 = Block(in_dim, out_dim, groups=groups)
        self.block2 = Block(out_dim, out_dim, groups=groups)
        self.res_conv = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, time_emb=None):
        """
        Forward pass of ResnetBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).
            time_emb (torch.Tensor, optional): Time embedding tensor of shape (B, time_emb_dim).

        Returns:
            torch.Tensor: Transformed tensor of shape (B, C_out, H, W).
        """
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.reshape(time_emb.size(0), time_emb.size(1), 1, 1)
            # 将time_emb拆分为(scale, shift)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        
        # Residual connection
        return torch.add(h, self.res_conv(x))



