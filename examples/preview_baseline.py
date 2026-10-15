"""Reproduce the README's synthetic interpolation example without external data."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from src.resampling import bicubic


def main():
    y, x = np.mgrid[:256, :256]
    fields = ((x // 28 + y // 36) % 2) * .15
    river = np.exp(-((x - 100 - 20 * np.sin(y / 35)) / 8) ** 2) * .2
    reference = np.stack([.3 + fields - river, .35 + fields - river * .8,
                          .4 + fields - river * .5]).astype('float32')
    low = gaussian_filter(reference, (0, 1.2, 1.2)).reshape(3, 64, 4, 64, 4).mean(axis=(2, 4))
    prediction = bicubic(low, 4)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, data, title in zip(axes, [low, prediction, reference],
                                 ['40 m simulated input', '10 m interpolation grid', '10 m synthetic reference']):
        axis.imshow(np.clip((data.transpose(1, 2, 0) - .05) / .6, 0, 1), interpolation='nearest')
        axis.set_title(title, fontsize=11)
        axis.set_axis_off()
    figure.suptitle('Synthetic example · same display stretch · not a satellite accuracy benchmark', fontsize=12)
    output = ROOT / 'docs/assets/synthetic-example.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)
    print(output)


if __name__ == '__main__':
    main()
