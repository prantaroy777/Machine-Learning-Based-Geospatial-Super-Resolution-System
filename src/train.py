import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dataset import RasterPatches
from .metrics import image_metrics
from .model import SuperResolution, load_model


def select_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Select cpu or install a compatible PyTorch build.")
    return device


def reconstruction_loss(prediction, target):
    pixel = torch.sqrt((prediction - target).square() + 1e-6).mean()
    dx = F.l1_loss(prediction[..., 1:] - prediction[..., :-1], target[..., 1:] - target[..., :-1])
    dy = F.l1_loss(prediction[..., 1:, :] - prediction[..., :-1, :], target[..., 1:, :] - target[..., :-1, :])
    return pixel + .05 * (dx + dy)


def train(data, output, epochs=50, batch_size=8, features=64, blocks=8,
          learning_rate=1e-4, device="auto", seed=42, patience=10):
    if min(epochs, batch_size, features, blocks) < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Training settings must be positive and finite.")
    if patience < 0:
        raise ValueError("Patience must be nonnegative; 0 disables early stopping.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty model output directory to preserve previous runs.")
    device = select_device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    train_set, val_set = RasterPatches(data, "train", seed), RasterPatches(data, "val", seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    model = SuperResolution(train_set.manifest["bands"], train_set.scale, features, blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    output.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_loss(model(lr), hr)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss. Check normalization and raster values.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * lr.shape[0]
        model.eval()
        error = baseline = 0.0
        with torch.inference_mode():
            for lr, hr in val_loader:
                lr, hr = lr.to(device), hr.to(device)
                error += F.l1_loss(model(lr), hr).item() * lr.shape[0]
                baseline += F.l1_loss(F.interpolate(lr, scale_factor=model.scale, mode="bicubic", align_corners=False), hr).item() * lr.shape[0]
        error /= len(val_set)
        baseline /= len(val_set)
        if not np.isfinite(error):
            raise ValueError("Validation produced non-finite predictions.")
        row = {"epoch": epoch, "train_loss": total / len(train_set), "validation_mae": error,
               "bicubic_mae": baseline, "beats_bicubic": error < baseline}
        history.append(row)
        print(json.dumps(row), flush=True)
        if error < best:
            best = error
            stale_epochs = 0
            manifest = train_set.manifest
            checkpoint = {"format_version": 1, "model_config": model.config,
                          "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "value_scale": manifest["value_scale"], "value_offset": manifest["value_offset"],
                          "epoch": epoch, "validation": row, "seed": seed,
                          "sources": manifest["sources"], "training_domain": manifest["training_domain"]}
            for key in ("band_descriptions", "raster_scales", "raster_offsets"):
                if key in manifest:
                    checkpoint[key] = manifest[key]
            temp = output / "best.tmp"
            torch.save(checkpoint, temp)
            temp.replace(output / "best.pt")
        else:
            stale_epochs += 1
        scheduler.step()
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if patience and stale_epochs >= patience:
            print(f"Early stopping after {epoch} epochs; best checkpoint retained.", flush=True)
            break
    return {"checkpoint": str(output / "best.pt"), "best_validation_mae": best,
            "bicubic_mae": baseline, "beats_bicubic": best < baseline}


def evaluate(data, checkpoint, device="auto", data_range=1.0):
    device = select_device(device)
    model, saved = load_model(checkpoint, device)
    dataset = RasterPatches(data, "val")
    manifest = dataset.manifest
    for key in ("band_descriptions", "raster_scales", "raster_offsets"):
        if key in saved and manifest.get(key) != saved[key]:
            raise ValueError(f"Evaluation {key} differs from checkpoint training metadata.")
    if (manifest["bands"] != model.config["bands"] or manifest["scale"] != model.scale or
            manifest["value_scale"] != saved["value_scale"] or manifest["value_offset"] != saved["value_offset"]):
        raise ValueError("Dataset bands, scale and normalization must match the checkpoint.")
    results = {"model": [], "bicubic": []}
    with torch.inference_mode():
        for lr, hr in DataLoader(dataset, batch_size=1):
            lr = lr.to(device)
            predictions = {"model": model(lr), "bicubic": F.interpolate(lr, scale_factor=model.scale, mode="bicubic", align_corners=False)}
            for name, prediction in predictions.items():
                results[name].append(image_metrics(prediction[0].cpu().numpy(), hr[0].numpy(), data_range))
    summary = {}
    for name, rows in results.items():
        summary[name] = {}
        for key in rows[0]:
            values = [r[key] for r in rows if r[key] is not None]
            summary[name]["exact_match_fraction" if key == "exact_match" else key] = float(np.mean(values)) if values else None
    return {"domain": manifest["training_domain"],
            "limitation": "Validation set used for model selection; use an independent reference/test region for final accuracy claims.",
            "patches": len(dataset), "data_range": data_range, "mean_patch_metrics": summary}
