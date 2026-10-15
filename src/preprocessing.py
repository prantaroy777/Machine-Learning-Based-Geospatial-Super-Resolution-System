"""Prepare spatially separate HR patches; degradation is performed during training."""
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from .geospatial_utils import raster_paths, read_valid, validate_raster


def prepare(train_dir, val_dir, output, scale=4, patch_size=128, stride=128,
            value_scale=10000.0, value_offset=0.0, max_patches=10000,
            lr_train_dir=None, lr_val_dir=None):
    if scale not in (2, 4, 8) or patch_size < 16 or patch_size % scale:
        raise ValueError("Scale must be 2, 4, or 8; patch size must be >=16 and divisible by scale.")
    if stride < 1 or max_patches < 1 or not np.isfinite(value_scale) or value_scale <= 0 or not np.isfinite(value_offset):
        raise ValueError("Stride, patch limit and value scale must be positive; normalization must be finite.")
    sources = {"train": raster_paths(train_dir), "val": raster_paths(val_dir)}
    if bool(lr_train_dir) != bool(lr_val_dir):
        raise ValueError("Paired training requires both LR training and LR validation directories.")
    paired = bool(lr_train_dir)
    lr_sources = {}
    if paired:
        for split, directory in (("train", lr_train_dir), ("val", lr_val_dir)):
            paths = raster_paths(directory)
            lr_sources[split] = {p.name: p for p in paths}
            if len(lr_sources[split]) != len(paths) or len({p.name for p in sources[split]}) != len(sources[split]):
                raise ValueError("Paired scene filenames must be unique within each split.")
            if set(lr_sources[split]) != {p.name for p in sources[split]}:
                raise ValueError("Paired HR/LR scenes must have exactly matching filenames in each split.")
        if stride % scale:
            raise ValueError("Paired patch stride must be divisible by scale.")
    metadata = {}
    bands = None
    signature = None
    for split, paths in sources.items():
        metadata[split] = []
        for path in paths:
            with rasterio.open(path) as ds:
                validate_raster(ds)
                bands = ds.count if bands is None else bands
                if ds.count != bands:
                    raise ValueError("All training and validation scenes must have the same band count and order.")
                current = (ds.descriptions, ds.scales, ds.offsets)
                if signature is not None and current != signature:
                    raise ValueError("Training scenes must have consistent band descriptions, scales and offsets.")
                signature = current
                if paired:
                    with rasterio.open(lr_sources[split][path.name]) as lr:
                        validate_raster(lr)
                        from affine import Affine
                        if (lr.crs != ds.crs or lr.count != ds.count or
                            (lr.width * scale, lr.height * scale) != (ds.width, ds.height) or
                            not np.allclose(tuple(lr.transform), tuple(ds.transform * Affine.scale(scale)), rtol=0, atol=1e-8)):
                            raise ValueError("Paired LR/HR scenes must be exactly aligned with matching CRS, bands and scale.")
                        if (lr.descriptions, lr.scales, lr.offsets) != current:
                            raise ValueError("Paired LR/HR band descriptions and radiometric metadata must match.")
                bounds = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
                metadata[split].append({"path": str(path.resolve()), "bounds_wgs84": list(bounds)})
    for train in metadata["train"]:
        for val in metadata["val"]:
            a, b = train["bounds_wgs84"], val["bounds_wgs84"]
            if train["path"] == val["path"] or (max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])):
                raise ValueError("Training and validation footprints overlap. Use geographically separate scenes.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prepared-data output must be a new or empty directory.")
    output.mkdir(parents=True, exist_ok=True)
    records = {"train": [], "val": []}
    for split, paths in sources.items():
        (output / split).mkdir(exist_ok=True)
        for source_index, path in enumerate(paths):
            with rasterio.open(path) as ds:
                for y in range(0, ds.height - patch_size + 1, stride):
                    for x in range(0, ds.width - patch_size + 1, stride):
                        if len(records[split]) >= max_patches:
                            break
                        data, valid = read_valid(ds, Window(x, y, patch_size, patch_size))
                        if not valid.all():
                            continue
                        hr = ((data - value_offset) / value_scale).astype("float32")
                        arrays = {"hr": hr}
                        if paired:
                            with rasterio.open(lr_sources[split][path.name]) as lr:
                                low, low_valid = read_valid(lr, Window(x // scale, y // scale, patch_size // scale, patch_size // scale))
                            if not low_valid.all():
                                continue
                            arrays["lr"] = ((low - value_offset) / value_scale).astype("float32")
                        if not all(np.isfinite(a).all() for a in arrays.values()):
                            raise ValueError("Normalization produced non-finite values; check value_scale and stored values.")
                        name = f"{split}/{len(records[split]):07d}.npz"
                        np.savez_compressed(output / name, **arrays)
                        records[split].append({"file": name, "scene": source_index, "x": x, "y": y})
        if not records[split]:
            raise ValueError(f"No fully valid {patch_size}-pixel patches in {split}. Check size and NoData/cloud masks.")
    manifest = {"version": 1, "scale": scale, "bands": bands, "patch_size": patch_size,
                "value_scale": value_scale, "value_offset": value_offset,
                "sources": metadata, "patches": records,
                "band_descriptions": list(signature[0]), "raster_scales": list(signature[1]),
                "raster_offsets": list(signature[2]),
                "training_domain": "aligned LR/HR pairs (user-supplied observations)" if paired else "synthetically degraded optical imagery"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {split: len(items) for split, items in records.items()}
