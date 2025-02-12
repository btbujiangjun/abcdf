#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ABC Diffusion Model

This module defines the `ABCDiffusion` and `GaussianDiffusion` classes,
which implement a diffusion-based generative model for image synthesis.

Features:
- Forward diffusion process
- Reverse sampling process
- Exponential moving average (EMA) support
- Noise generation and loss calculation

Author: Jiang Jun  
Date: 2025-02-11  
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from model.config import MODEL_CONFIG
from model.denoise import ABCDenoise
from module.scheduler import LinearScheduler

class ABCDiffusion(ABC, nn.Module):
    """
    Abstract base class for diffusion models.

    This class defines the general diffusion model framework, including:
    - Forward diffusion process
    - Noise generation
    - Loss function
    - Normalization utilities

    Attributes:
        model (ABCDenoise): The denoising neural network.
        scheduler (LinearScheduler): The scheduler for controlling noise levels.
        cfg (MODEL_CONFIG): Configuration dictionary.
    """
    def __init__(self, 
            model: ABCDenoise, 
            scheduler: LinearScheduler,
            cfg: MODEL_CONFIG,
        ):
        """
        Initializes the diffusion model.

        Args:
            model (ABCDenoise): The denoising network.
            scheduler (LinearScheduler): The scheduler controlling noise.
            cfg (MODEL_CONFIG): Configuration dictionary.
        """
        super().__init__()
        self.cfg = cfg
        self.device = cfg["device"]
        self.image_size = cfg["image_size"]
        
        self.model = model.to(self.device)
        self.scheduler = scheduler
        self.loss_fn = F.mse_loss

        self.normalize = self._normalize_fn if cfg["auto_normalize"] else self._identity_fn
        self.denormalize = self._denormalize_fn if cfg["auto_normalize"] else self._identity_fn

    def _normalize_fn(self, img):
        """Normalize image tensor from [0,1] to [-1,1]."""
        return img * 2 - 1

    def _denormalize_fn(self, img):
        """Denormalize image tensor from [-1,1] to [0,1]."""
        return (img + 1) * 0.5

    def _identity_fn(self, img):
        return img
    
    def _clamp_fn(self, img):
        """Clamp image tensor values between -1 and 1."""
        return torch.clamp(img, min=-1., max=1.)

    @torch.no_grad()
    @abstractmethod
    def generate_noise(self, shape:tuple):
        pass

    @torch.no_grad()
    def forward_diffusion(self, img:torch.Tensor, noise=None):
        """
        Forward diffusion process: Generates noisy images.

        x_t = sqrt(α_cumprod) * x_0 + sqrt(1-α_cumprod) * noise

        Args:
            img (torch.Tensor): Input image tensor.
            noise (torch.Tensor, optional): Custom noise tensor. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - x_t (Noisy image)
                - noise (Used noise)
                - t (Time step)
        """
        noise = noise if noise is not None else self.generate_noise(img.shape)
        t = self.scheduler.sampling(img.shape[0])

        x_t = self.scheduler.alpha_cumprod_sqrt(t) * img + \
            self.scheduler.one_minus_alphas_cumprod_sqrt(t) * noise
        
        return x_t, noise, t

    @torch.no_grad()
    @abstractmethod
    def sample(self, batch_size=16, debug=False):
        pass
    
    def loss(self, y, noise, t)->float:
        """
        Compute loss for training.

        Args:
            y (torch.Tensor): Model's predicted noise.
            noise (torch.Tensor): Ground truth noise.
            t (torch.Tensor): Time step tensor.

        Returns:
            float: Mean weighted loss value.
        """
        loss = self.loss_fn(y, noise, reduction='none')
        loss = loss.flatten(start_dim=1).mean(dim=-1)
        return loss.mean()

    @abstractmethod
    def forward(self, img:torch.Tensor, *args, **kwargs):
        pass

class GaussianDiffusion(ABCDiffusion):
    """
    Gaussian diffusion model implementing forward and reverse processes.

    This subclass provides:
    - Noise sampling
    - Reverse sampling process (denoising)
    - Training loss calculation
    """
    def __init__(self, 
            model: ABCDenoise,
            scheduler: LinearScheduler, 
            cfg: MODEL_CONFIG):
        """
        Initializes the Gaussian Diffusion model.

        Args:
            model (ABCDenoise): Denoising network.
            scheduler (LinearScheduler): Noise schedule.
            cfg (MODEL_CONFIG): Configuration dictionary.
        """
        super().__init__(model, scheduler, cfg)
        assert model.in_channels == model.out_channels, \
            f"Error: Mismatch in model input/output channels:{model.in_channels}/{model.out_channels}."
        
    def generate_noise(self, shape)->torch.Tensor:
        """Generate Gaussian noise of a given shape."""
        if not (isinstance(shape, tuple) and len(shape) == 4):
            raise ValueError(f"generate_noise function:shape must be a 4-tuple")
        return torch.randn(shape, device=self.device)

    @torch.no_grad()
    def sample(self, batch_size=16, debug=False):
        """
        Generate images by reversing the diffusion process.

        Args:
            batch_size (int): Number of images to generate.
            debug (bool, optional): Whether to return intermediate steps. Defaults to False.

        Returns:
            torch.Tensor: Generated image(s).
        """
        shape = (batch_size, self.model.in_channels, self.image_size, self.image_size)
        
        x_start = None # 对x_0的估计
        img = self.generate_noise(shape) #noise
        
        imgs = [img] # img of each step
        # from t-1 to 0 逐步反向采样
        for t in reversed(range(0, self.scheduler.time_steps)):
            img, x_start = self._p_sample(img, t)
            imgs.append(img)

        img = torch.stack(imgs, dim=1) if debug else img
        return self.denormalize(img) # [-1, 1] -> [0, 1]

    @torch.no_grad()
    def _p_sample(self, x_t, step_t:int, cliped=True):
        """
        Perform a single step of reverse diffusion.

        Args:
            x_t (torch.Tensor): Current noisy image tensor.
            step_t (int): Current time step.
            cliped (bool, optional): Whether to clamp output. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Next denoised image
                - Estimated original image (x_start)
        """    
        
        """
        1) 将step_t(int)扩展成与batch相同的t(tensor)
        """
        t = torch.full((x_t.shape[0],), step_t, device=self.device, dtype=torch.long)
        
        """
        2)给定当前噪声图x_t和时间步t，通过模型预测噪声pred_noise并得到对x_0的估计x_start 
        """
        pred_noise = self.model(x_t, t)
        
        # 通过x_t和预测噪声，反推x_0的估计值x_start
        # x_start = 1 / sqrt(alpha_cumprod) * x_t - sqrt(1 / alpha_cumprod - 1) * noise_pred
        x_coef = self.scheduler.recip_alpha_cumprod_sqrt(t)
        noise_coef = self.scheduler.recipm1_alpha_cumprod_sqrt(t)

        x_start = x_coef * x_t - noise_coef * pred_noise
        if cliped:
            x_start = self._clamp_fn(x_start)
            # x_0(x_start)被剪裁，为更准确则重新计算一次噪声
            # 通过x_t和x_0的预估值(x_start)反推噪声预测值
            # noise = (1 / sqrt(alpha_cumprod) * x_t - x_start) / sqrt(1 / alpha_cumprod - 1)
            pred_noise = (x_coef * x_t - x_start) / noise_coef

        """
        3) 计算扩散过程p(x_{t-1}|x_t)的均值和方差, 用于反向采样
        """
        # 后验分布相当于q(x_{t-1}|x_t, x_0)
        mean_coef1 = self.scheduler.posterior_mean_coef1(t)
        mean_coef2 = self.scheduler.posterior_mean_coef2(t)
        posterior_mean = mean_coef1 * x_start + mean_coef2 * x_t
        posterior_variance_log = self.scheduler.posterior_variance_log(t)

        """
        4) 采样：x_{t-1} = mean + var * noise
        """
        noise = self.generate_noise(x_t.shape) if step_t > 0 else 0.
        pred_img = posterior_mean + (0.5 * posterior_variance_log).exp() * noise

        return pred_img, x_start


    def forward(self, img:torch.Tensor, *args, ** kwargs)->float:
        b, c, h, w = img.shape
        assert h == self.image_size and w == self.image_size, \
            f"Error:image size dismatch {self.image_size}"
        img = self.normalize(img.to(self.device))
        
        x_t, noise, t = self.forward_diffusion(img)
        y = self.model(x_t, t)
        
        return self.loss(y, noise, t)






