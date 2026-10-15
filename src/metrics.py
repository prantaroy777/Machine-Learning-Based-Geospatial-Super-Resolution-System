import numpy as np
from skimage.metrics import structural_similarity


def image_metrics(prediction, reference, data_range=1.0):
    """CHW arrays, in common normalized units; no per-image rescaling."""
    if prediction.ndim != 3 or prediction.shape != reference.shape or not all(prediction.shape) or not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("Matching CHW shapes and a positive finite data range are required.")
    p, r = prediction.astype("float64"), reference.astype("float64")
    if not (np.isfinite(p).all() and np.isfinite(r).all()):
        raise ValueError("Metrics require finite, valid pixels.")
    mse = float(np.mean((p - r) ** 2))
    win = min(7, p.shape[-2], p.shape[-1])
    win = win if win % 2 else win - 1
    ssim = float(np.mean([structural_similarity(a, b, data_range=data_range, win_size=win)
                          for a, b in zip(p, r)])) if win >= 3 else None
    angle = None
    if p.shape[0] > 1:
        denominator = np.linalg.norm(p, axis=0) * np.linalg.norm(r, axis=0)
        valid = denominator > 1e-12
        if valid.any():
            cosine = np.sum(p * r, axis=0)[valid] / denominator[valid]
            angle = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).mean())
    return {"mae": float(np.abs(p - r).mean()), "rmse": float(np.sqrt(mse)),
            "psnr_db": float(10 * np.log10(data_range ** 2 / mse)) if mse else None,
            "ssim": ssim, "spectral_angle_degrees": angle, "exact_match": mse == 0}
