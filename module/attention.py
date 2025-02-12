#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Attention Modules

This module defines two types of attention mechanisms:
1. `Attention`: Standard multi-head self-attention (MHSA).
2. `LinearAttention`: Efficient self-attention with reduced computational cost.

Author: Jiang Jun  
Date: 2025-02-11  
"""

import torch
import torch.nn as nn
from module.normalize import LayerNorm

class Attention(nn.Module):
    """
    Standard Multi-Head Self-Attention (MHSA).

    This module applies self-attention across spatial features, enhancing 
    feature interactions across different regions of the input tensor.

    Attributes:
        scale (float): Scaling factor for query vectors.
        heads (int): Number of attention heads.
        to_qkv (nn.Conv2d): Convolution layer to compute Q, K, V matrices.
        to_out (nn.Conv2d): Output projection layer.
    """

    def __init__(self, dim, heads: int = 4, head_dim: int = 32):
        """
        Args:
            dim (int): Input channel dimension.
            heads (int, optional): Number of attention heads. Defaults to 4.
            head_dim (int, optional): Dimension per attention head. Defaults to 32.
        """
        super().__init__()
        self.scale = head_dim ** -0.5
        self.heads = heads
        hidden_dim = head_dim * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor (B, C, H, W).
        """
        b, c, h, w = x.shape
        
        qkv = self.to_qkv(x).chunk(3, dim=1)
        #q,k,v=map(lambda t:rearrange(t,'b (h c) x y -> b h c (x y)',h=self.heads),qkv)
        q, k, v = [t.view(t.shape[0], self.heads, -1, t.shape[2] * t.shape[3]) for t in qkv]
        
        q = q * self.scale
        #torch.einsum('b h d i, b h d j -> b h i j', q, k)
        sim = torch.matmul(q.permute(0, 1, 3, 2), k)
        attn = sim.softmax(dim=-1)
        
        #torch.einsum('b h i j, b h d j -> b h i d', attn, v)
        out = torch.matmul(attn, v.permute(0, 1, 3, 2))
        #rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        out = out.view(out.shape[0], out.shape[1], h, w, out.shape[3]).permute(0, 1, 4, 2, 3)
        out = out.contiguous().view(out.shape[0], out.shape[1] * out.shape[2], h, w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    """
    Linear Self-Attention.

    A more efficient alternative to traditional self-attention, reducing 
    computational complexity from O(N^2) to O(N).

    Attributes:
        scale (float): Scaling factor for query vectors.
        heads (int): Number of attention heads.
        to_qkv (nn.Conv2d): Convolution layer to compute Q, K, V matrices.
        to_out (nn.Sequential): Output projection layer with LayerNorm.
    """

    def __init__(self, dim: int, heads: int = 4, head_dim: int = 32): 
        """
        Args:
            dim (int): Input channel dimension.
            heads (int, optional): Number of attention heads. Defaults to 4.
            head_dim (int, optional): Dimension per attention head. Defaults to 32.
        """
        super().__init__()
        self.scale = head_dim ** -0.5
        self.heads = heads
        hidden_dim = head_dim * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass of the linear attention module.

        Args:
            x (torch.Tensor): Input tensor (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor (B, C, H, W).
        """
        b, c, h ,w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        
        #map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q, k, v = [t.view(t.shape[0], self.heads, -1, t.shape[2] * t.shape[3]) for t in qkv]
        
        q = q.softmax(dim=-2) * self.scale # scale q
        k = k.softmax(dim=-1)
        v = v / (h * w) #normalize v

        #torch.einsum('b h d n, b h e n -> b h d e', k, v)
        context = torch.matmul(k, v.permute(0, 1, 3, 2))

        #torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = torch.matmul(context.permute(0, 1, 3, 2), q)

        #rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out.view(b, out.shape[1] * out.shape[2], h, w))






