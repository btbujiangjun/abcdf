#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Manager

This module provides a class for managing model checkpoints, 
including saving (dumping) and loading models with Exponential 
Moving Average (EMA) support.

Usage:
- model_manager = ModelManager(model, ema_model)
- model_manager.dump("checkpoint.ckpt")
- model_manager.load("checkpoint.ckpt")

Author: Jiang Jun  
Date: 2025-02-12  
"""

import os
import torch

class ModelManager:
    """
    A utility class to manage model checkpoint saving and loading.

    Attributes:
        model (torch.nn.Module): The main model.
        ema_model (torch.nn.Module): The Exponential Moving Average (EMA) model.
        ext (str): The file extension for checkpoints.
        device (str): The device where the model is stored.
    """
    def __init__(self, model:torch.nn.Module, ema_model:torch.nn.Module):
        """
        Initializes the ModelManager.

        Args:
            model (torch.nn.Module): The main model.
            ema_model (torch.nn.Module): The EMA model.
        """
        self.model = model
        self.ema_model = ema_model

        self.ext = ".ckpt"
        self.device = model.device

    def dump(self, ckpt:str):
        if not ckpt.endswith(self.ext):
            ckpt += self.ext

        parent_dir = os.path.dirname(ckpt)
        os.makedirs(parent_dir, exist_ok=True)

        data = {
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict()
        }
        torch.save(data, ckpt)
        
        print(f"Dumped checkpoint {ckpt} successfully.", flush=True)
        return ckpt

    def load(self, ckpt:str):
        if not ckpt.endswith(self.ext):
            ckpt += self.ext

        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Error loading model: no such file: {ckpt}.")

        data = torch.load(ckpt, weights_only=False, map_location=self.device)
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])

        print(f"Loaded checkpoint {ckpt} successfully.", flush=True)



