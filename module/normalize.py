"""
Layer Normalization and Pre-Normalization Modules

This module implements:
1. LayerNorm: A simplified layer normalization technique applied along the channel dimension.
2. PreNorm: A wrapper that applies normalization before passing input to another function.

Author: Jiang Jun
Date: 2025-02-11
"""

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Channel-wise Layer Normalization.

    Normalizes input along the channel dimension while keeping spatial dimensions unchanged.

    Args:
        dim (int): Number of input channels.
    """
    def __init__(self, dim:int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x:torch.Tensor)->torch.Tensor:
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = x.var(dim=1, unbiased=False, keepdim=True)
        mean = x.mean(dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    """
    Pre-Normalization Wrapper.

    Applies LayerNorm before passing the input to a given function.

    Args:
        dim (int): Number of input channels.
        fn (nn.Module): Function or module to apply after normalization.
    """
    def __init__(self, dim, fn: nn.Module):
        super().__init__()
        self.norm_fn = nn.Sequential(LayerNorm(dim), fn)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.norm_fn(x)
        

