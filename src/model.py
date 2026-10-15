"""Residual reconstruction network with a bicubic skip and learned upsampling."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(features, features, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(features, features, 3, padding=1))

    def forward(self, x):
        return x + .1 * self.body(x)


class SuperResolution(nn.Module):
    def __init__(self, bands, scale=4, features=64, blocks=8):
        super().__init__()
        if scale not in (2, 4, 8) or min(bands, features, blocks) < 1:
            raise ValueError("Positive bands/features/blocks and scale 2, 4, or 8 are required.")
        self.config = dict(bands=bands, scale=scale, features=features, blocks=blocks)
        self.scale = scale
        self.head = nn.Conv2d(bands, features, 3, padding=1)
        self.body = nn.Sequential(*(ResidualBlock(features) for _ in range(blocks)),
                                  nn.Conv2d(features, features, 3, padding=1))
        up = []
        for _ in range(int(math.log2(scale))):
            up.extend([nn.Conv2d(features, features * 4, 3, padding=1), nn.PixelShuffle(2), nn.ReLU()])
        self.up = nn.Sequential(*up)
        self.tail = nn.Conv2d(features, bands, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        features = self.head(x)
        residual = self.tail(self.up(features + self.body(features)))
        return F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False) + residual


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format; use a checkpoint from this project.")
    required = {"model_config", "state_dict", "value_scale", "value_offset"}
    if not required.issubset(checkpoint):
        raise ValueError("Checkpoint is missing model configuration, weights or normalization.")
    config = checkpoint["model_config"]
    if not isinstance(config, dict) or set(config) != {"bands", "scale", "features", "blocks"} or not all(type(v) is int and v > 0 for v in config.values()):
        raise ValueError("Invalid model configuration in checkpoint.")
    for key in ("value_scale", "value_offset"):
        if not isinstance(checkpoint[key], (int, float)) or not math.isfinite(checkpoint[key]):
            raise ValueError("Checkpoint normalization must be finite numeric values.")
    if checkpoint["value_scale"] <= 0:
        raise ValueError("Checkpoint value_scale must be positive.")
    if not isinstance(checkpoint["state_dict"], dict) or not all(isinstance(t, torch.Tensor) and torch.isfinite(t).all() for t in checkpoint["state_dict"].values()):
        raise ValueError("Checkpoint contains invalid or non-finite weights.")
    model = SuperResolution(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint
