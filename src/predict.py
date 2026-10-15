from pathlib import Path
import uuid
import warnings
import math
from datetime import datetime, timezone

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt, zoom

from .geospatial_utils import read_valid, validate_raster, windows
from .resampling import bicubic
from .reporting import sha256_file


def fill_invalid(values, valid):
    if valid.all():
        return values
    if not valid.any():
        return np.zeros_like(values)
    indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return values[:, indices[0], indices[1]]


def predict(input_path, output_path, scale=4, checkpoint=None, method="model", tile_size=256,
            device="auto", ensemble=False, progress=None, build_overviews=False):
    if scale not in (2, 4, 8) or tile_size < 16:
        raise ValueError("Scale must be 2, 4, or 8 and tile size must be at least 16.")
    if method not in ("model", "bicubic", "bilinear", "nearest"):
        raise ValueError("Unknown prediction method.")
    source, output = Path(input_path), Path(output_path)
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError("Choose a new output path; source and existing outputs are never overwritten.")
    model = saved = None
    halo = 8
    if method == "model":
        if checkpoint is None:
            raise ValueError("Model prediction requires a trained checkpoint. Use bicubic for a labeled baseline.")
        import torch
        from .model import load_model
        from .train import select_device
        device = select_device(device)
        model, saved = load_model(checkpoint, device)
        if model.scale != scale:
            raise ValueError(f"Checkpoint is trained for {model.scale}x; requested {scale}x.")
        halo = 2 * model.config["blocks"] + 8
        if not saved.get("validation", {}).get("beats_bicubic", False):
            warnings.warn("This checkpoint has not beaten bicubic validation MAE. Compare with the baseline before use.", stacklevel=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.stem}-{uuid.uuid4().hex}.tmp.tif")
    lock = output.with_name(output.name + ".lock")
    # Exclusive reservation protects simultaneous jobs that choose the same output.
    with lock.open("x", encoding="utf-8") as stream:
        stream.write("Geo Enhance output in progress. Remove only after confirming the job stopped.")
    try:
        if output.exists():
            raise ValueError("Output was created by another job. Choose a new output path.")
        with rasterio.open(source) as src:
            validate_raster(src)
            if model is not None and src.count != model.config["bands"]:
                raise ValueError("Raster band count differs from checkpoint. Band order must also match training.")
            if saved is not None:
                for key, actual in (("band_descriptions", src.descriptions), ("raster_scales", src.scales), ("raster_offsets", src.offsets)):
                    expected = saved.get(key)
                    if expected is not None and tuple(expected) != tuple(actual):
                        raise ValueError(f"Input {key} differs from checkpoint training metadata.")
            profile = {"driver": "GTiff", "width": src.width * scale, "height": src.height * scale,
                       "count": src.count, "dtype": "float32", "crs": src.crs,
                       "transform": src.transform * Affine.scale(1 / scale), "nodata": float("nan"),
                       "compress": "deflate", "predictor": 3, "tiled": True, "BIGTIFF": "IF_SAFER"}
            with rasterio.open(temp, "w", **profile) as dst:
                dst.scales, dst.offsets = src.scales, src.offsets
                dst.descriptions = src.descriptions
                dst.units = src.units
                tags = {k: v for k, v in src.tags().items() if not k.startswith("STATISTICS_")}
                dst.update_tags(**tags)
                dst.update_tags(SR_METHOD=method, SR_SCALE=str(scale),
                                SR_SOFTWARE="Geo Enhance 0.2.1", SR_CREATED_UTC=datetime.now(timezone.utc).isoformat(),
                                SR_SOURCE=source.name, SR_CHECKPOINT_SHA256=sha256_file(checkpoint) if model else "none",
                                SR_NORMALIZATION=str({"value_scale": saved["value_scale"], "value_offset": saved["value_offset"]}) if model else "none",
                                SR_MASK_POLICY="intersection across all bands; nearest valid fill for context only",
                                SR_DETAIL="Model estimate; nominal pixel spacing does not establish effective resolution" if model else "Interpolation baseline; no learned detail",
                                SR_CHECKPOINT=Path(checkpoint).name if model else "none",
                                SR_TRAINING_DOMAIN=saved.get("training_domain", "unknown") if model else "not applicable",
                                SR_BEATS_BICUBIC=str(saved.get("validation", {}).get("beats_bicubic", False)) if model else "not applicable",
                                SR_ENSEMBLE=str(bool(ensemble and model)))
                for band in range(1, src.count + 1):
                    dst.update_tags(band, **{k: v for k, v in src.tags(band).items() if not k.startswith("STATISTICS_")})
                total_tiles = math.ceil(src.width / tile_size) * math.ceil(src.height / tile_size)
                if progress:
                    progress(0, total_tiles)
                for completed, win in enumerate(windows(src.width, src.height, tile_size), 1):
                    x, y, w, h = map(int, (win.col_off, win.row_off, win.width, win.height))
                    x0, y0 = max(0, x - halo), max(0, y - halo)
                    x1, y1 = min(src.width, x + w + halo), min(src.height, y + h + halo)
                    data, valid = read_valid(src, Window(x0, y0, x1 - x0, y1 - y0))
                    clean = fill_invalid(data, valid)
                    if model is None:
                        order = {"nearest": 0, "bilinear": 1, "bicubic": 3}[method]
                        result = bicubic(clean, scale) if method == "bicubic" else zoom(clean, (1, scale, scale), order=order, mode="nearest", grid_mode=True)
                    else:
                        normalized = (clean - saved["value_offset"]) / saved["value_scale"]
                        tensor = torch.from_numpy(normalized).unsqueeze(0).to(device)
                        with torch.inference_mode():
                            if ensemble:
                                total = None
                                for k in range(4):
                                    for flip in (False, True):
                                        transformed = torch.rot90(tensor, k, (-2, -1))
                                        if flip:
                                            transformed = transformed.flip(-1)
                                        prediction = model(transformed)
                                        if flip:
                                            prediction = prediction.flip(-1)
                                        prediction = torch.rot90(prediction, -k, (-2, -1))
                                        total = prediction if total is None else total + prediction
                                result = (total / 8)[0].cpu().numpy()
                            else:
                                result = model(tensor)[0].cpu().numpy()
                        result = result * saved["value_scale"] + saved["value_offset"]
                    ox, oy = (x - x0) * scale, (y - y0) * scale
                    crop = result[:, oy:oy + h * scale, ox:ox + w * scale].astype("float32")
                    mask = np.repeat(np.repeat(valid[y - y0:y - y0 + h, x - x0:x - x0 + w], scale, axis=0), scale, axis=1)
                    if not np.isfinite(crop[:, mask]).all():
                        raise ValueError("Prediction produced non-finite values. Check input normalization and model.")
                    crop[:, ~mask] = np.nan
                    dst.write(crop, window=Window(x * scale, y * scale, w * scale, h * scale))
                    if progress:
                        progress(completed, total_tiles)
                if build_overviews:
                    from rasterio.enums import Resampling
                    levels = [n for n in (2, 4, 8, 16) if min(dst.width, dst.height) // n >= 64]
                    if levels:
                        dst.build_overviews(levels, Resampling.average)
                        dst.update_tags(ns="rio_overview", resampling="average")
        temp.replace(output)
    finally:
        if temp.exists():
            temp.unlink()
        lock.unlink(missing_ok=True)
    return str(output)
