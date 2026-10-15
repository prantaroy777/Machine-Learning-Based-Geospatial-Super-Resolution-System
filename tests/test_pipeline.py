import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from affine import Affine
from rasterio.transform import from_origin

from src.geospatial_utils import inspect_raster
from src.metrics import image_metrics
from src.predict import predict
from src.preprocessing import prepare

try:
    import torch
    from src.dataset import RasterPatches
    from src.model import SuperResolution
    from src.train import train, evaluate
    TORCH_ERROR = None
    torch.set_num_threads(1)
except (ImportError, OSError) as exc:
    TORCH_ERROR = str(exc)


def write_scene(path, width=40, height=32, x=500000, nodata=False, transform=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:height, :width]
    values = np.stack([xx + yy, 2 * xx + yy + 5, xx + 3 * yy + 10]).astype("float32") / 100
    if nodata:
        values[:, 4:6, 5:8] = -9999
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=3,
                       dtype="float32", crs="EPSG:32632", transform=transform or from_origin(x, 5200000, 40, 40), nodata=-9999) as ds:
        ds.write(values)
        ds.descriptions = ("red", "green", "blue")
        ds.scales = (.01, .01, .01)
        ds.offsets = (1, 1, 1)
        ds.update_tags(SENSOR="synthetic-test")
        ds.update_tags(1, STATISTICS_MINIMUM="0", WAVELENGTH="test")
    return values


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_georeferencing_masks_and_metadata(self):
        source, target = self.root / "source.tif", self.root / "result.tif"
        transform = Affine(40, 2, 500000, 1, -40, 5200000)
        write_scene(source, nodata=True, transform=transform)
        predict(source, target, method="bicubic", scale=4, tile_size=16)
        with rasterio.open(source) as a, rasterio.open(target) as b:
            self.assertEqual((b.width, b.height), (a.width * 4, a.height * 4))
            self.assertEqual(b.crs, a.crs)
            self.assertEqual(b.bounds, a.bounds)
            self.assertEqual(b.transform, a.transform * Affine.scale(.25))
            self.assertEqual(b.descriptions, a.descriptions)
            self.assertEqual(b.scales, a.scales)
            self.assertEqual(b.offsets, a.offsets)
            self.assertEqual(b.tags()["SR_METHOD"], "bicubic")
            self.assertNotIn("STATISTICS_MINIMUM", b.tags(1))
            self.assertEqual(b.tags(1)["WAVELENGTH"], "test")
            mask = np.repeat(np.repeat(a.dataset_mask(), 4, 0), 4, 1)
            np.testing.assert_array_equal(b.dataset_mask(), mask)
            self.assertTrue(np.isfinite(b.read()[:, mask > 0]).all())

    def test_interpolation_tiling_consistency(self):
        source = self.root / "source.tif"
        write_scene(source, width=67, height=53)
        for method in ("nearest", "bilinear", "bicubic"):
            one, tiled = self.root / f"one-{method}.tif", self.root / f"tiles-{method}.tif"
            predict(source, one, method=method, tile_size=128)
            predict(source, tiled, method=method, tile_size=16)
            with rasterio.open(one) as a, rasterio.open(tiled) as b:
                np.testing.assert_allclose(a.read(), b.read(), atol=1e-5, rtol=1e-5)

    def test_all_invalid_and_overwrite_guard(self):
        source, target = self.root / "source.tif", self.root / "out.tif"
        write_scene(source)
        with rasterio.open(source, "r+") as ds:
            ds.write(np.full((3, 32, 40), -9999, dtype="float32"))
        predict(source, target, method="bicubic")
        with rasterio.open(target) as ds:
            self.assertFalse(ds.dataset_mask().any())
        with self.assertRaises(ValueError):
            predict(source, source, method="bicubic")
        with self.assertRaises(ValueError):
            predict(source, target, method="bicubic")

    def test_reject_untrained_model(self):
        with self.assertRaisesRegex(ValueError, "requires a trained checkpoint"):
            predict("unused.tif", self.root / "out.tif")

    def test_overlap_rejected_before_writing(self):
        write_scene(self.root / "train/a.tif")
        write_scene(self.root / "val/b.tif", x=500100)
        with self.assertRaisesRegex(ValueError, "overlap"):
            prepare(self.root / "train", self.root / "val", self.root / "prepared", patch_size=16)
        self.assertFalse((self.root / "prepared").exists())

    def test_prepare_and_normalization(self):
        original = write_scene(self.root / "train/a.tif", width=32, nodata=True)
        write_scene(self.root / "val/b.tif", width=32, x=600000)
        out = self.root / "prepared"
        counts = prepare(self.root / "train", self.root / "val", out, patch_size=16, stride=16, value_scale=2, value_offset=1)
        self.assertEqual(counts, {"train": 3, "val": 4})
        manifest = json.loads((out / "manifest.json").read_text())
        patch = manifest["patches"]["train"][0]
        with np.load(out / patch["file"]) as item:
            x, y = patch["x"], patch["y"]
            np.testing.assert_allclose(item["hr"], (original[:, y:y + 16, x:x + 16] - 1) / 2)

    def test_metrics(self):
        a = np.ones((3, 16, 16), dtype="float32")
        exact = image_metrics(a, a)
        self.assertTrue(exact["exact_match"])
        self.assertEqual(exact["ssim"], 1)
        self.assertAlmostEqual(exact["spectral_angle_degrees"], 0, places=5)
        shifted = image_metrics(a * .5, a)
        self.assertEqual(shifted["mae"], .5)
        self.assertAlmostEqual(shifted["psnr_db"], 6.0206, places=4)

    @unittest.skipIf(TORCH_ERROR is not None, f"PyTorch unavailable: {TORCH_ERROR}")
    def test_model_scales_and_bicubic_initialization(self):
        for scale in (2, 4, 8):
            model = SuperResolution(5, scale, features=8, blocks=1)
            x = torch.rand(1, 5, 8, 11)
            y = model(x)
            self.assertEqual(y.shape, (1, 5, 8 * scale, 11 * scale))
            baseline = torch.nn.functional.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False)
            torch.testing.assert_close(y, baseline)

    @unittest.skipIf(TORCH_ERROR is not None, f"PyTorch unavailable: {TORCH_ERROR}")
    def test_training_evaluation_and_model_tiles(self):
        write_scene(self.root / "train/a.tif", width=32, height=32)
        write_scene(self.root / "val/b.tif", width=32, height=32, x=600000)
        data, run = self.root / "prepared", self.root / "model"
        prepare(self.root / "train", self.root / "val", data, patch_size=32, stride=32, value_scale=1)
        result = train(data, run, epochs=2, batch_size=1, features=8, blocks=1, device="cpu")
        self.assertTrue(Path(result["checkpoint"]).is_file())
        ds = RasterPatches(data, "val")
        torch.testing.assert_close(ds[0][0], ds[0][0])
        report = evaluate(data, result["checkpoint"], device="cpu")
        self.assertEqual(report["patches"], 1)
        self.assertTrue(np.isfinite(report["mean_patch_metrics"]["model"]["rmse"]))
        source = self.root / "source.tif"
        write_scene(source, width=53, height=43)
        for ensemble in (False, True):
            one, tiled = self.root / f"one{ensemble}.tif", self.root / f"tile{ensemble}.tif"
            predict(source, one, checkpoint=result["checkpoint"], device="cpu", tile_size=128, ensemble=ensemble)
            predict(source, tiled, checkpoint=result["checkpoint"], device="cpu", tile_size=16, ensemble=ensemble)
            with rasterio.open(one) as a, rasterio.open(tiled) as b:
                np.testing.assert_allclose(a.read(), b.read(), atol=2e-6)
        with self.assertRaisesRegex(ValueError, "trained for"):
            predict(source, self.root / "bad.tif", scale=2, checkpoint=result["checkpoint"], device="cpu")
        self.assertFalse((self.root / "bad.tif").exists())


if __name__ == "__main__":
    unittest.main()
