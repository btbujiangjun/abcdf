#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for training a diffusion model using U-Net-based denoising.

Author: Jiang Jun
Date: 2025-02-12
"""

import torch
from dataset import ABCDataset, ABCDataLoader
from module.scheduler import LinearScheduler
from model.config import MODEL_CONFIG
from model.denoise import Unet
from model.diffusion import GaussianDiffusion
from model.trainer import Trainer


def main():
    cfg = MODEL_CONFIG

    dataset = ABCDataset(
        folder = "../my_diffusion/data/faces",
        image_size=cfg["image_size"],
    )
    dataloader = ABCDataLoader(
        dataset, 
        batch_size=cfg["batch_size"], 
        shuffle=True
    )

    denoise_model = Unet(cfg)
    
    diffusion_model = GaussianDiffusion(
        denoise_model,
        LinearScheduler(
            time_steps = cfg["time_steps"],
            device=cfg["device"],
        ),
        cfg
    )
    
    print(diffusion_model)
   
    results_folder = "./results"
    is_warmup = True

    trainer = Trainer(diffusion_model, cfg)
    trainer.train(
        dataloader, 
        epochs=5,
        results_folder=results_folder,
        is_warmup=is_warmup,
    )

if __name__ == "__main__":
    main()

