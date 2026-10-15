"""Windowed evaluation against an independently observed, aligned reference raster."""
from pathlib import Path

import numpy as np
import rasterio

from .geospatial_utils import validate_raster, windows


def require_same_grid(a, b):
    if a.crs != b.crs or (a.width, a.height, a.count) != (b.width, b.height, b.count):
        raise ValueError("Reference must have the same CRS, dimensions and band count as the output.")
    if not np.allclose(tuple(a.transform), tuple(b.transform), rtol=0, atol=1e-8):
        raise ValueError("Reference pixel grid is not aligned. Co-register it before comparison; no automatic resampling is performed.")
    if all(a.descriptions) and all(b.descriptions) and a.descriptions != b.descriptions:
        raise ValueError("Band descriptions/order differ between output and reference.")


def compare_rasters(prediction, reference, data_range, tile_size=512, apply_scale_offset=False):
    """Pixel-weighted metrics over joint validity, with float64 accumulation.

    SSIM is deliberately omitted: computing it independently per tile introduces
    boundary bias and filling invalid pixels would contaminate local statistics.
    """
    if not np.isfinite(data_range) or data_range <= 0 or tile_size < 1:
        raise ValueError("Specify a positive finite data range and positive tile size.")
    if Path(prediction).resolve() == Path(reference).resolve():
        raise ValueError("Use an independent reference, not the prediction itself.")
    with rasterio.open(prediction) as pred, rasterio.open(reference) as ref:
        validate_raster(pred)
        validate_raster(ref)
        require_same_grid(pred, ref)
        if not apply_scale_offset and (pred.scales != ref.scales or pred.offsets != ref.offsets):
            raise ValueError("Scale/offset metadata differ. Use apply_scale_offset to compare physical values.")
        sums = np.zeros((3, pred.count), dtype="float64")
        count = angle_count = 0
        angle_sum = 0.0
        for window in windows(pred.width, pred.height, tile_size):
            a = pred.read(window=window, masked=True, out_dtype="float64")
            b = ref.read(window=window, masked=True, out_dtype="float64")
            av, bv = a.filled(np.nan), b.filled(np.nan)
            valid = (~np.ma.getmaskarray(a).any(axis=0) & ~np.ma.getmaskarray(b).any(axis=0)
                     & np.isfinite(av).all(axis=0) & np.isfinite(bv).all(axis=0))
            av, bv = av[:, valid], bv[:, valid]
            if apply_scale_offset:
                av = av * np.asarray(pred.scales)[:, None] + np.asarray(pred.offsets)[:, None]
                bv = bv * np.asarray(ref.scales)[:, None] + np.asarray(ref.offsets)[:, None]
            if not (np.isfinite(av).all() and np.isfinite(bv).all()):
                raise ValueError("Non-finite physical values after applying scale/offset metadata.")
            error = av - bv
            sums += np.stack([np.abs(error).sum(axis=1), np.square(error).sum(axis=1), error.sum(axis=1)])
            count += av.shape[1]
            if pred.count > 1:
                denominator = np.linalg.norm(av, axis=0) * np.linalg.norm(bv, axis=0)
                nonzero = denominator > 1e-12
                cosine = (av * bv).sum(axis=0)[nonzero] / denominator[nonzero]
                angle_sum += float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).sum())
                angle_count += int(nonzero.sum())
        if not count:
            raise ValueError("The rasters have no jointly valid pixels to compare.")
        if not np.isfinite(sums).all():
            raise ValueError("Metric accumulation overflowed; check your raster values and units.")
        def metrics(abs_sum, squared_sum, bias_sum, n):
            mse = float(squared_sum / n)
            return {"mae": float(abs_sum / n), "rmse": float(np.sqrt(mse)),
                    "bias_prediction_minus_reference": float(bias_sum / n),
                    "psnr_db": float(10 * np.log10(data_range ** 2 / mse)) if mse else None,
                    "exact_match": mse == 0}
        return {"schema_version": 1, "prediction": Path(prediction).name, "reference": Path(reference).name,
                "evaluation": "aligned reference raster; reference independence, timing and sensor compatibility must be verified by the user",
                "units": "scale/offset-adjusted values" if apply_scale_offset else "stored raster values",
                "data_range": data_range, "valid_pixels": count,
                "valid_fraction": count / (pred.width * pred.height),
                "overall": metrics(*sums.sum(axis=1), count * pred.count),
                "per_band": [{"band": i + 1, "description": pred.descriptions[i],
                              **metrics(*sums[:, i], count)} for i in range(pred.count)],
                "spectral_angle_degrees": angle_sum / angle_count if angle_count else None,
                "spectral_angle_valid_pixels": angle_count,
                "notes": ["PSNR null with exact_match=true means infinite PSNR.",
                          "SSIM is omitted for masked full-raster comparisons.",
                          "All bands share the intersection of finite, valid pixels in both rasters."]}
