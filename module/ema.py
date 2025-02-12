#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EMA (Exponential Moving Average) for Model Parameters

This module implements Exponential Moving Average (EMA) for neural networks.
EMA stabilizes training by maintaining a separate set of parameters 
that update more smoothly over iterations.

Author: Jiang Jun
Date: 2025-02-11
""" 

import copy
import torch
import torch.nn as nn

class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) for stabilizing model training.

    This class maintains a copy of the model with parameters updated using an EMA formula:
        ema_param = decay * ema_param + (1 - decay) * model_param

    Attributes:
        model (nn.Module): The original model whose parameters are being tracked.
        decay (float): EMA decay rate (default: 0.999). Higher values mean slower updates.
        ema_model (nn.Module): The EMA version of the model.
    """
    def __init__(self, model:nn.Module, decay:float=0.999):
        """
        Initializes the EMA model.

        Args:
            model (nn.Module): The model whose parameters will be tracked by EMA.
            decay (float, optional): EMA decay factor. Default is 0.999.
        """
        super().__init__()
        self.model = model
        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))
        """
        self.ema_model = type(model)(
            model.model,
            model.scheduler,
            model.cfg
        )
        self.ema_model.load_state_dict(copy.deepcopy(model.state_dict()))
        """
        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)

    def update(self):
        """
        Updates EMA model parameters using the exponential moving average formula.
        """
        for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.decay).add_((1. - self.decay.item()) * model_param.data)
    
    @torch.no_grad()
    def sample(self, batch_size=16, debug=False):
        """
        Generates samples using the EMA model.

        Args:
            batch_size (int, optional): Number of samples to generate. Default is 16.
            debug (bool, optional): Whether to return debug information. Default is False.

        Returns:
            torch.Tensor: Generated samples.
        """
        return self.ema_model.sample(batch_size, debug)

    def to(self, *args, **kwargs):
        """
        Ensures both the main and EMA models are moved to the same device.

        Args:
            *args: Arguments for `torch.nn.Module.to()`.
            **kwargs: Keyword arguments for `torch.nn.Module.to()`.
        """
        self.model.to(*args, **kwargs)
        self.ema_model.to(*args, **kwargs)
        return self
