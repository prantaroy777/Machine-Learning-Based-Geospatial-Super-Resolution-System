"""Keys bicubic interpolation matching PyTorch's align_corners=False convention."""
import numpy as np


def _kernel(distance):
    x = np.abs(distance)
    a = -.75
    return np.where(x <= 1, (a + 2) * x ** 3 - (a + 3) * x ** 2 + 1,
                    np.where(x < 2, a * x ** 3 - 5 * a * x ** 2 + 8 * a * x - 4 * a, 0))


def bicubic(array, scale):
    """CHW float array; half-pixel centers and replicated edge samples."""
    result = array.astype("float32", copy=False)
    for axis in (2, 1):
        length = result.shape[axis]
        positions = (np.arange(length * scale, dtype="float64") + .5) / scale - .5
        base = np.floor(positions).astype("int64")
        shape = [1, 1, 1]
        shape[axis] = len(positions)
        expanded = None
        for offset in (-1, 0, 1, 2):
            indices = base + offset
            weights = _kernel(positions - indices).astype("float32").reshape(shape)
            term = np.take(result, np.clip(indices, 0, length - 1), axis=axis) * weights
            expanded = term if expanded is None else expanded + term
        result = expanded
    return result
