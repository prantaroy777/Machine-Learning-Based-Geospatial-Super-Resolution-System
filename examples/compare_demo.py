"""Render the synthetic demonstration with one shared display range."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
import numpy as np


def main():
    paths = ["data/demo/synthetic_40m.tif", "outputs/demo/bicubic_10m.tif",
             "outputs/demo/model_10m.tif", "data/demo/val/synthetic_10m.tif"]
    names = ["40 m input", "10 m interpolation", "10 m model estimate", "10 m synthetic reference"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    for ax, path, title in zip(axes, paths, names):
        with rasterio.open(path) as ds:
            image = ds.read([1, 2, 3], out_shape=(3, 256, 256), resampling=Resampling.nearest)
        image = np.clip((image.transpose(1, 2, 0) - .05) / .6, 0, 1)
        ax.imshow(image, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
    fig.suptitle("Synthetic software demonstration • shared contrast stretch • not satellite validation", fontsize=13)
    output = Path("outputs/demo/comparison.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
