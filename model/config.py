import torch

class ABCDict(dict):
    def __init__(self, default_value, *args, **kwargs):
        self.default_value = default_value
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        return self.get(key, self.default_value)

MODEL_CONFIG = ABCDict(None, {
    "image_size": 64,
    "time_steps": 100,
    "batch_size": 64,
    "auto_normalize": True,
    "lr": 1e-3,
    "adam_betas": (0.9, 0.99),
    "accumulation_steps": 1,
    "ema_decay": 0.995,
    "max_grad_norm": 1.0,
    "sample_steps": 500,
    "device": torch.device(
        "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
    ),
    "Unet" : {
        "dim": 16,
        "in_channels": 3,
        "out_channels": None,
        "dim_factor": (1, 2, 4),
        "resnet_block_groups": 8,
        "pos_emb": {
            "type": "default", # random or default
            "dim": 32,
            "learnable": True,
        }
    },
})


