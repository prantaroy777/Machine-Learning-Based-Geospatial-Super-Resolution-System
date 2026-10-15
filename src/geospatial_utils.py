from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def inspect_raster(path):
    with rasterio.open(path) as ds:
        validate_raster(ds)
        return {"path": str(path), "width": ds.width, "height": ds.height,
                "bands": ds.count, "dtype": ds.dtypes[0], "crs": str(ds.crs),
                "pixel_size": list(ds.res), "units": ds.crs.linear_units if ds.crs.is_projected else "degrees",
                "bounds": list(ds.bounds), "nodata": str(ds.nodata),
                "band_descriptions": list(ds.descriptions)}


def validate_raster(ds):
    if ds.crs is None:
        raise ValueError("A georeferenced raster with a CRS is required.")
    if any(np.issubdtype(np.dtype(t), np.complexfloating) for t in ds.dtypes):
        raise ValueError("Complex radar data is not supported by this optical pipeline.")
    if not np.isfinite(tuple(ds.transform)).all() or abs(ds.transform.determinant) < 1e-30:
        raise ValueError("Raster transform must be finite and invertible.")
    if any(item.name in {"palette", "alpha"} for item in ds.colorinterp):
        raise ValueError("Palette/alpha bands are not continuous optical measurements. Convert palette data or remove alpha while preserving its validity mask before use.")


def raster_paths(directory):
    paths = sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in {".tif", ".tiff"})
    if not paths:
        raise ValueError(f"No GeoTIFF files found in {directory}")
    return paths


def windows(width, height, size):
    for y in range(0, height, size):
        for x in range(0, width, size):
            yield Window(x, y, min(size, width - x), min(size, height - y))


def read_valid(ds, window=None):
    array = ds.read(window=window, out_dtype="float32", masked=True)
    values = array.filled(np.nan)
    valid = ~np.ma.getmaskarray(array).any(axis=0) & np.isfinite(values).all(axis=0)
    return values, valid
