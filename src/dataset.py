import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch.utils.data import Dataset


def degrade(hr, scale, sigma, noise=0.0, rng=None):
    """Approximate optical blur and pixel-area integration, then sensor noise."""
    blurred = gaussian_filter(hr, sigma=(0, sigma, sigma), mode="reflect")
    c, h, w = hr.shape
    lr = blurred.reshape(c, h // scale, scale, w // scale, scale).mean(axis=(2, 4))
    if noise:
        lr = lr + rng.normal(0, noise, lr.shape)
    return lr.astype("float32")


class RasterPatches(Dataset):
    def __init__(self, root, split, seed=42):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.records = self.manifest["patches"][split]
        self.scale = self.manifest["scale"]
        self.training = split == "train"
        self.rng = np.random.default_rng(seed)
        if self.scale not in (2, 4, 8) or not self.records:
            raise ValueError("Prepared data must contain patches and use scale 2, 4, or 8.")
        for record in self.records:
            path = (self.root / record["file"]).resolve()
            if not path.is_relative_to(self.root.resolve()) or not path.is_file():
                raise ValueError("Every patch must exist inside the prepared dataset directory.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with np.load(self.root / self.records[index]["file"], allow_pickle=False) as item:
            hr = item["hr"].copy()
            paired_lr = item["lr"].copy() if "lr" in item else None
        if (hr.ndim != 3 or hr.shape[0] != self.manifest["bands"] or min(hr.shape) < 1
                or hr.shape[1] % self.scale or hr.shape[2] % self.scale or not np.isfinite(hr).all()):
            raise ValueError("Invalid HR patch shape, band count, scale divisibility or pixel values.")
        if paired_lr is not None and (paired_lr.shape != (hr.shape[0], hr.shape[1] // self.scale, hr.shape[2] // self.scale)
                                     or not np.isfinite(paired_lr).all()):
            raise ValueError("Paired LR patch must have matching bands, dimensions and finite values.")
        hr = hr.astype("float32")
        if paired_lr is not None:
            paired_lr = paired_lr.astype("float32")
        if self.training:
            rotation = int(self.rng.integers(4))
            hr = np.rot90(hr, rotation, axes=(1, 2))
            if paired_lr is not None:
                paired_lr = np.rot90(paired_lr, rotation, axes=(1, 2))
            if self.rng.random() < .5:
                hr = hr[:, :, ::-1]
                if paired_lr is not None:
                    paired_lr = paired_lr[:, :, ::-1]
        if paired_lr is not None:
            return torch.from_numpy(paired_lr.copy()), torch.from_numpy(hr.copy())
        sigma = float(self.rng.uniform(.1, .5) * self.scale) if self.training else .3 * self.scale
        noise = float(self.rng.uniform(0, .003)) if self.training else 0.0
        lr = degrade(hr, self.scale, sigma, noise, self.rng)
        return torch.from_numpy(lr), torch.from_numpy(hr.copy())
