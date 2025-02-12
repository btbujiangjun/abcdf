"""
Linear Noise Scheduler for Diffusion Models

This class implements a linear variance schedule for diffusion models.
It precomputes key statistical values for efficient sampling.

Author: Jiang Jun
Date: 2025-02-11
"""

import torch
import torch.nn as nn

class LinearScheduler(nn.Module):
    """
    Linear variance scheduler for diffusion models.

    Args:
        time_steps (int): Number of time steps in the schedule.
        beta_start (float): Starting beta value for linear schedule.
        beta_end (float): Ending beta value for linear schedule.
        device (str): Device to store tensors (default: "cpu").
    """

    def __init__(self, time_steps:int = 1000,
            beta_start:float = 0.0001,
            beta_end:float = 0.02,
            device="cpu"):
        super().__init__()
        self.device = device
        scale = 1000 / time_steps

        beta = torch.linspace(
            beta_start * scale, 
            beta_end * scale, 
            time_steps, 
            dtype=torch.float,
            device=device
        )
        self.time_steps = beta.shape[0]
        alpha = (1. - beta).to(device)
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1, device=device), alpha_cumprod[:-1]])

        self.register_buffer("alpha_cumprod_sqrt_data", torch.sqrt(alpha_cumprod))
        self.register_buffer("one_minus_alpha_cumprod_sqrt_data", torch.sqrt(1. - alpha_cumprod))
        
        self.register_buffer("recip_alpha_cumprod_sqrt_data", torch.sqrt(1. / alpha_cumprod))
        self.register_buffer("recipm1_alpha_cumprod_sqrt_data", torch.sqrt(1. / alpha_cumprod - 1))
        
        self.register_buffer("posterior_mean_coef1_data", beta * torch.sqrt(alpha_cumprod_prev) / (1. - alpha_cumprod))
        self.register_buffer("posterior_mean_coef2_data", (1. - alpha_cumprod_prev) * torch.sqrt(alpha) / (1. - alpha_cumprod))
        self.register_buffer("posterior_variance_log_data", torch.log((beta * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).clamp(min=1e-20)))

    def sampling(self, size:int)->torch.Tensor:
        return torch.randint(0, 
            self.time_steps, 
            (size,), 
            device=self.device, 
            dtype=torch.long
        )  

    def alpha_cumprod_sqrt(self, t: torch.Tensor)->torch.Tensor:
        return self.alpha_cumprod_sqrt_data[t].view(-1, 1, 1, 1)

    def one_minus_alphas_cumprod_sqrt(self, t:torch.Tensor)->torch.Tensor:
        return self.one_minus_alpha_cumprod_sqrt_data[t].view(-1, 1, 1, 1)

    def recip_alpha_cumprod_sqrt(self, t:torch.Tensor)->torch.Tensor:
        return self.recip_alpha_cumprod_sqrt_data[t].view(-1, 1, 1, 1)

    def recipm1_alpha_cumprod_sqrt(self, t:torch.Tensor)->torch.Tensor:
        return self.recipm1_alpha_cumprod_sqrt_data[t].view(-1, 1, 1, 1)

    def posterior_mean_coef1(self, t:torch.Tensor)->torch.Tensor:
        return self.posterior_mean_coef1_data[t].view(-1, 1, 1, 1)

    def posterior_mean_coef2(self, t:torch.Tensor)->torch.Tensor:
        return self.posterior_mean_coef2_data[t].view(-1, 1, 1, 1)

    def posterior_variance_log(self, t:torch.Tensor)->torch.Tensor:
        return self.posterior_variance_log_data[t].view(-1, 1, 1, 1)

