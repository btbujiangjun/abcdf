#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trainer Module for ABCDiffusion Model

This module provides a `Trainer` class that handles the training process
for the ABCDiffusion model. It supports:
- Gradient accumulation
- Exponential Moving Average (EMA)
- Periodic model checkpointing
- Image sampling during training

Author: Jiang Jun  
Date: 2025-02-11  
"""

import os
import torch
from pathlib import Path
from utils import save_images
from module.ema import EMA
from dataset import ABCDataLoader
from model.diffusion import ABCDiffusion
from model.manager import ModelManager
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP

class Trainer:
    """
    Trainer class for training the ABCDiffusion model.

    This class provides utilities for training with gradient accumulation,
    Exponential Moving Average (EMA), and automatic checkpointing.

    Attributes:
        model (ABCDiffusion): The diffusion model to be trained.
        cfg (dict): Configuration dictionary containing training hyperparameters.
        optimizer (torch.optim.Optimizer): Adam optimizer for model training.
        ema (EMA): Exponential Moving Average of the model for better stability.
    """

    def __init__(self, model:ABCDiffusion, cfg):
        """
        Initializes the Trainer.

        Args:
            model (ABCDiffusion): The diffusion model to train.
            cfg (dict): Configuration dictionary with hyperparameters.
        """
        self._model, self.manager = None, None
        self.cfg = cfg
        self.model = model
        self.max_grad_norm = self.model.cfg["max_grad_norm"]
        self.accumulation_steps = self.cfg["accumulation_steps"]

    @property
    def model(self):
        """
        Returns the model, unwrapping from DistributedDataParallel (DDP) if necessary.

        Returns:
            torch.nn.Module: The underlying model.
        """
        if isinstance(self._model, DDP):
            return self._model.module
        else:
            return self._model

    @model.setter
    def model(self, value):
        """
        Sets the model and initializes the optimizer and EMA.

        Args:
            value (torch.nn.Module): The model instance.
        """
        self._model = value
        self.optimizer = torch.optim.Adam(
            self._model.parameters(), 
            lr = self.cfg["lr"], 
            betas = self.cfg["adam_betas"]
        )

        self.ema = EMA(
            self._model,
            decay = self.cfg["ema_decay"],
        ).to(self._model.device)

        self.manager = ModelManager(self._model, self.ema.model)

    def train(self, 
            dataloader:ABCDataLoader,
            epochs:int = 1,
            num_sample = 25,
            results_folder: str = "./results/",
            is_warmup = True):
        """
        Runs the training loop.

        Args:
            dataloader (ABCDataLoader): The data loader for training batches.
            epochs (int, optional): Number of training epochs. Defaults to 1.
            num_sample (int, optional): Number of images to sample at checkpoints. Defaults to 25.
            results_folder (str, optional): Directory to save model checkpoints and images. Defaults to "./results/".
        """
        results_folder = Path(results_folder)
        results_folder.mkdir(parents=True, exist_ok=True)
        
        if is_warmup:
            self.load_lastest(results_folder)

        sample_steps = self.cfg["sample_steps"]
        global_step, samples_seen, local_loss = 0, 0, 0
        dataloader_len = len(dataloader)

        for epoch in range(epochs):
            for i, batch in enumerate(dataloader):
                loss = self.model(batch) / self.accumulation_steps
                loss.backward()
                
                local_loss += loss.item()
                samples_seen += batch.numel()
               
                # Perform gradient accumulation update
                if (i + 1) % self.accumulation_steps == 0 or \
                        i == dataloader_len - 1:
                    clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.ema.update()
                    
                    print(f"Epoch:{epoch}, Global_step:{global_step}, Sample:{samples_seen}, Loss:{local_loss:.3f}")
                    global_step += 1
                    local_loss = 0

                # Save model and generate sample images at intervals
                if global_step % sample_steps == 0 or \
                        i == dataloader_len - 1:
                    milestone = global_step // sample_steps
                    self.save(os.path.join(results_folder, f"model-{milestone}.ckpt"))
                    
                    images = self.ema.sample(batch_size=num_sample)
                    save_images(images, os.path.join(results_folder, f"sample-{milestone}.png"))

        print(f"Training completed.")


    def save(self, ckpt):
        """
        Saves the model and EMA state to a checkpoint.

        Args:
            ckpt (Path): Path to save the checkpoint.
        """
        self.manager.dump(ckpt)

    def load(self, ckpt:str):
        """
        Loads the model and EMA state from a checkpoint.

        Args:
            ckpt (Path): Path to the checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        self.manager.load(ckpt)

    def load_lastest(self, ckpt_dir:str):
        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.exists():
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        ckpts = [p for p in ckpt_dir.glob(f"*{self.manager.ext}")]
        if len(ckpts) > 0:
            lastest_ckpt = max(ckpts, key=os.path.getmtime)
            self.load(str(lastest_ckpt))
