#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
U-Net Denoising Model

This module defines the `ABCDenoise` abstract base class and its implementation `Unet`.
It is a diffusion-based neural network used for denoising tasks in generative models.

Features:
- Downsampling and upsampling stages with residual connections
- Time-based positional embeddings
- Attention mechanisms for feature refinement

Author: Jiang Jun  
Date: 2025-02-11  
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from model.config import MODEL_CONFIG
from module.normalize import PreNorm
from module.resnet import ResnetBlock
from module.attention import Attention, LinearAttention
from module.layer import ResidualLayer, UpSampleLayer, DownSampleLayer
from module.embedding import SinPosEmbedding, RandomSinPosEmbedding


class ABCDenoise(ABC, nn.Module):
    """
    Abstract base class for denoising models.
    
    This class defines a standard interface for diffusion-based denoising architectures.
    """
    def __init__(self, cfg:MODEL_CONFIG):
        super().__init__()
        self.cfg = cfg
        self.dim = self.cfg["dim"]
        self.in_channels = self.cfg["in_channels"]
        self.out_channels = self.cfg["out_channels"] or self.cfg["in_channels"]

    @abstractmethod
    def forward(self, x:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        pass


class Unet(ABCDenoise):
    """
    U-Net Denoising Model

    Implements a U-Net structure with residual connections, attention layers,
    and a hierarchical downsampling and upsampling process.

    Attributes:
        init_conv (nn.Conv2d): Initial convolution layer.
        time_mlp (nn.Sequential): Time embedding MLP.
        downs (nn.ModuleList): Downsampling layers.
        ups (nn.ModuleList): Upsampling layers.
        final_res_block (ResnetBlock): Final residual block.
        final_conv (nn.Conv2d): Output convolution layer.
    """
    def __init__(self, cfg: MODEL_CONFIG):
        super().__init__(cfg["Unet"])
        self.base_cfg = cfg
        self.device = self.base_cfg["device"]
        self.init_conv = nn.Conv2d(self.in_channels, self.dim, 7, padding=3).to(self.device)

        # if dim_factor is (1, 2, 3, 4) and dim == 64
        # then: dim_seq = [64, 64*1, 64*2, 64*4, 64*8]
        #       in_out_dim = [(64, 64),(64, 128),(128, 256),(256,512)]
        dim_seq = [self.dim, *map(lambda m : self.dim * m, self.cfg["dim_factor"])]
        in_out_dim = list(zip(dim_seq[:-1], dim_seq[1:]))

        pos_emb_cfg = self.cfg["pos_emb"]
        if (pos_emb_cfg["type"] == "random"): #random or default
            sin_pos_emb = RandomSinPosEmbedding(
                pos_emb_cfg["dim"], 
                pos_emb_cfg["learnable"],
                device=self.device
            )
            fourier_dim = pos_emb_cfg["dim"] + 1
        else:
            sin_pos_emb = SinPosEmbedding(self.dim, device=self.device)
            fourier_dim = self.dim

        time_dim = 4 * self.dim

        # 时间嵌入先经过正弦嵌入，然后用两个全连接层并将维度转为time_dim
        self.time_mlp = nn.Sequential(
            sin_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        ).to(self.device)

        
        groups = self.cfg["resnet_block_groups"]
        self.downs = nn.ModuleList([]).to(self.device)
        self.ups = nn.ModuleList([]).to(self.device)
        
        """
        U-Net down stage:
          1)ResnetBlock(in_dim, in_dim)
          2)ResnetBlock(in_dim, in_dim)
          3)Residual(PreNorm(LinearAttention))
          4)不是最后层用Downsample；最后一层用3x3卷积保持分辨率
        """
        for i , (in_dim, out_dim) in enumerate(in_out_dim):
            self.downs.append(nn.ModuleList([
                ResnetBlock(in_dim, in_dim, time_emb_dim=time_dim, groups=groups),
                ResnetBlock(in_dim, in_dim, time_emb_dim=time_dim, groups=groups),
                ResidualLayer(PreNorm(in_dim, LinearAttention(in_dim))),
                DownSampleLayer(in_dim, out_dim) if i < len(in_out_dim) - 1 
                    else nn.Conv2d(in_dim, out_dim, 3, padding=1)
            ]))

        """
        U-Net bottom stage:ResnetBlock->self-attention->ResnetBlock
        """
        mid_dim = dim_seq[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, groups=groups).to(self.device)
        self.mid_attn = ResidualLayer(PreNorm(mid_dim, Attention(mid_dim))).to(self.device)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, groups=groups).to(self.device)

        """
        U-Net up stage:
          1) ResnetBlock(out_dim + in_dim, out_dim)
          2) ResnetBlock(out_dim + in_dim, out_dim)
          3) Residual(PreNorm(LinearAttention))
          4) 不是最后层UpSample；最后一层用3x3卷积保持分辨率
        """
        for i, (in_dim, out_dim) in enumerate(reversed(in_out_dim)):
            self.ups.append(nn.ModuleList([
                ResnetBlock(out_dim + in_dim, out_dim, time_emb_dim = time_dim, groups=groups),
                ResnetBlock(out_dim + in_dim, out_dim, time_emb_dim = time_dim, groups=groups),
                ResidualLayer(PreNorm(out_dim, LinearAttention(out_dim))),
                UpSampleLayer(out_dim, in_dim) if i < len(in_out_dim) - 1
                    else nn.Conv2d(out_dim, in_dim, 3, padding=1)
            ]))
        """
        Final layers
        """
        self.final_res_block = ResnetBlock(self.dim * 2, self.dim, time_emb_dim=time_dim).to(self.device)
        self.final_conv = nn.Conv2d(self.dim, self.out_channels or self.in_channels, 1).to(self.device)

    def forward(self, x:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the U-Net model.

        Args:
            x (torch.Tensor): Input image tensor.
            t (torch.Tensor): Time embedding tensor.

        Returns:
            torch.Tensor: Output denoised image tensor.
        """
        x = self.init_conv(x)
        r = x.clone() #res connection
        
        t = self.time_mlp(t)

        """
        U-Net downsample stage
        """
        h = []
        for b1, b2, attn, down_sample in self.downs:
            x = b1(x, t)
            h.append(x)

            x = b2(x, t)
            x = attn(x)
            h.append(x)

            x = down_sample(x)

        """
        U-Net bottom stage
        """
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        """
        U-Net upsample stage
        """
        for b1, b2, attn, up_sample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = b1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = b2(x, t)
            x = attn(x)

            x = up_sample(x)

        """
        Final processing
        """
        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)






