"""Portable output provenance; reports omit absolute input paths."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import rasterio


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_report(path):
    with rasterio.open(path) as ds:
        return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "file": Path(path).name, "crs": str(ds.crs), "transform": list(ds.transform),
                "width": ds.width, "height": ds.height, "bands": ds.count,
                "band_descriptions": list(ds.descriptions), "pixel_spacing": list(ds.res),
                "spacing_units": ds.crs.linear_units if ds.crs.is_projected else "degrees",
                "scales": list(ds.scales), "offsets": list(ds.offsets),
                "provenance": {k: v for k, v in ds.tags().items() if k.startswith("SR_")},
                "limitation": "Pixel spacing does not establish effective spatial resolution or accuracy."}


def save_json(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing a raster or previous report.
    text = json.dumps(data, indent=2, allow_nan=False)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(text)
