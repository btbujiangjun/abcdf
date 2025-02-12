
"""
Sinusoidal Positional Embedding

This module implements sinusoidal-based positional encodings for input sequences.
It provides both fixed and random sinusoidal embeddings, useful for transformer-based models.

Author: Jiang Jun
Date: 2025-02-11
"""

import math
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class ABCPosEmbedding(ABC, nn.Module):
    """
    Abstract base class for positional embeddings.

    Attributes:
        dim (int): Dimension of the embedding.
        device (str): Device where the embeddings are stored.
    """
    def __init__(self, dim:int, device="cpu"):
        super().__init__()
        self.dim = dim
        self.device = device
    
    @abstractmethod
    def forward(self, t:torch.Tensor)->torch.Tensor:
        pass

class SinPosEmbedding(ABCPosEmbedding):
    """
    Standard fixed sinusoidal positional embedding.

    Attributes:
        emb (torch.Tensor): Precomputed sinusoidal embedding values.
    """
    def __init__(self, dim:int, device='cpu'):
        """
        Initializes the fixed sinusoidal positional embedding.

        Args:
            dim (int): Dimension of the embedding.
            device (str, optional): Device for storing embeddings. Default is "cpu".
        """
        super().__init__(dim, device)
        half_dim = dim // 2
        
        emb = torch.exp(
            - torch.arange(half_dim) * (math.log(10000.0) / (half_dim - 1))
        ).to(device)
        self.register_buffer("emb", emb)

    def forward(self, t:torch.Tensor)->torch.Tensor:
        """
        Computes sinusoidal positional embeddings.

        Args:
            t (torch.Tensor): Input tensor of shape (batch_size,).

        Returns:
            torch.Tensor: Sinusoidal embedding of shape (batch_size, dim).
        """
        emb = t[:, None] * self.emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class RandomSinPosEmbedding(ABCPosEmbedding):
    """
    Randomized sinusoidal positional embedding with optional learning capability.

    This follows @crowsonkb's method for random sinusoidal embeddings:
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8

    Attributes:
        weight (nn.Parameter): Learnable frequency weights for random sinusoidal embedding.
    """
    def __init__(self, dim, learnable=False, device="cpu"):
        """
        Initializes the randomized sinusoidal positional embedding.

        Args:
            dim (int): Dimension of the embedding.
            learnable (bool, optional): If True, the embedding weights are trainable. Default is False.
            device (str, optional): Device for storing embeddings. Default is "cpu".
        """
        super().__init__(dim, device)
        assert dim % 2 == 0, f'dim {dim} must be even.'
        half_dim = dim // 2
        self.weight = nn.Parameter(
            torch.randn(half_dim), 
            requires_grad = learnable
        ).to(device)

    def forward(self, t:torch.Tensor)->torch.Tensor:
        """
        Computes the random sinusoidal positional embedding.

        Args:
            t (torch.Tensor): Input tensor of shape (batch_size,).

        Returns:
            torch.Tensor: Random sinusoidal embedding of shape (batch_size, dim).
        """
        t = t.unsqueeze(-1).to(self.device)
        freq = t * self.weight.unsqueeze(0) * 2 * math.pi
        return torch.cat((t, freq.sin(), freq.cos()), dim = -1)


