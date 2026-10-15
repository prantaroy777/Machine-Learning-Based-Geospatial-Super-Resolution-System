"""Generate synthetic fixtures, not real satellite observations or scientific evidence."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/demo")
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists() and any(p.is_file() for p in root.rglob("*")):
        parser.error("Choose a new or empty demo directory.")
    for split, seed, east in (("train", 42, 500000), ("val", 93, 600000)):
        rng = np.random.default_rng(seed)
        y, x = np.mgrid[:256, :256]
        channels = []
        for band in range(3):
            texture = gaussian_filter(rng.random((256, 256)), 2)
            fields = ((x // (24 + band * 4) + y // 32) % 2) * .15
            river = np.exp(-((x - 100 - 20 * np.sin(y / 35)) / 8) ** 2) * .2
            channels.append(.15 + .3 * texture + fields - river + .05 * band)
        hr = np.stack(channels).astype("float32")
        path = root / split / "synthetic_10m.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, hr, east, 10)
        if split == "val":
            blur = gaussian_filter(hr, (0, 1.2, 1.2), mode="reflect")
            lr = blur.reshape(3, 64, 4, 64, 4).mean(axis=(2, 4))
            write(root / "synthetic_40m.tif", lr, east, 40)
    print(f"Synthetic demo data written to {root.resolve()}; normalize with --value-scale 1.")


def write(path, values, east, resolution):
    with rasterio.open(path, "w", driver="GTiff", width=values.shape[2], height=values.shape[1], count=3,
                       dtype="float32", crs="EPSG:32632", transform=from_origin(east, 5200000, resolution, resolution)) as ds:
        ds.write(values)
        ds.descriptions = ("synthetic red", "synthetic green", "synthetic blue")
        ds.update_tags(DATA_ORIGIN="Synthetic software demonstration. Not observed satellite data.")


if __name__ == "__main__":
    main()
