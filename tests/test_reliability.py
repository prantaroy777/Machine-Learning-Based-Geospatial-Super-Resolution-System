import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.predict import predict
from src.preprocessing import prepare
from src.reporting import output_report, save_json
from src.resampling import bicubic
from src.validation import compare_rasters

try:
    import torch
    from src.dataset import RasterPatches
    from src.model import SuperResolution, load_model
    from src.train import train
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False


def scene(path, values, resolution=10, x=500000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", count=values.shape[0], height=values.shape[1], width=values.shape[2],
                       dtype="float32", nodata=-9999, crs="EPSG:32632", transform=from_origin(x, 5200000, resolution, resolution)) as ds:
        ds.write(values.astype("float32"))


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.a, self.b = self.root / "a.tif", self.root / "b.tif"

    def test_reference_metrics_ignore_nodata_and_are_tile_invariant(self):
        ref = np.ones((3, 17, 23), dtype="float32")
        pred = ref + .25
        pred[:, 0, 0] = -9999
        ref[:, 1, 1] = -9999
        scene(self.a, pred)
        scene(self.b, ref)
        a = compare_rasters(self.a, self.b, data_range=1, tile_size=7)
        b = compare_rasters(self.a, self.b, data_range=1, tile_size=512)
        self.assertEqual(a['valid_pixels'], 17 * 23 - 2)
        self.assertEqual(a['overall'], b['overall'])
        self.assertEqual(a['overall']['mae'], .25)
        self.assertEqual(a['overall']['bias_prediction_minus_reference'], .25)
        self.assertAlmostEqual(a['overall']['psnr_db'], 12.0411998)
        self.assertAlmostEqual(a['spectral_angle_degrees'], 0, places=5)
        json.dumps(a, allow_nan=False)

    def test_alignment_and_independent_path_guards(self):
        values = np.ones((1, 16, 16), dtype="float32")
        scene(self.a, values)
        scene(self.b, values, x=500001)
        with self.assertRaisesRegex(ValueError, "not aligned"):
            compare_rasters(self.a, self.b, 1)
        with self.assertRaisesRegex(ValueError, "independent"):
            compare_rasters(self.a, self.a, 1)
        with self.assertRaises(ValueError):
            compare_rasters(self.a, self.b, 0)

    def test_physical_scale_offset_and_exact_metrics(self):
        scene(self.a, np.full((1, 16, 16), 100, dtype="float32"))
        scene(self.b, np.ones((1, 16, 16), dtype="float32"))
        with rasterio.open(self.a, "r+") as ds:
            ds.scales = (.01,)
        with self.assertRaisesRegex(ValueError, "metadata differ"):
            compare_rasters(self.a, self.b, 1)
        result = compare_rasters(self.a, self.b, 1, apply_scale_offset=True)
        self.assertTrue(result['overall']['exact_match'])
        self.assertIsNone(result['overall']['psnr_db'])

    def test_empty_validity_rejected(self):
        scene(self.a, np.full((1, 16, 16), -9999, dtype="float32"))
        scene(self.b, np.ones((1, 16, 16), dtype="float32"))
        with self.assertRaisesRegex(ValueError, "no jointly valid"):
            compare_rasters(self.a, self.b, 1)

    def test_progress_provenance_and_overviews(self):
        scene(self.a, np.ones((3, 33, 35), dtype="float32"), resolution=40)
        progress = []
        predict(self.a, self.b, method="bicubic", tile_size=16, build_overviews=True,
                progress=lambda done, total: progress.append((done, total)))
        self.assertEqual(progress[0], (0, 9))
        self.assertEqual(progress[-1], (9, 9))
        report = output_report(self.b)
        self.assertEqual(report['pixel_spacing'], [10., 10.])
        self.assertEqual(report['provenance']['SR_SOURCE'], 'a.tif')
        with rasterio.open(self.b) as ds:
            self.assertEqual(ds.overviews(1), [2])
        target = self.root / 'report.json'
        save_json(target, report)
        with self.assertRaises(FileExistsError):
            save_json(target, report)

    def test_cancelled_job_cleans_partial_files(self):
        scene(self.a, np.ones((1, 32, 32), dtype="float32"))
        def cancel(done, total):
            if done == 1:
                raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            predict(self.a, self.b, method="bicubic", tile_size=16, progress=cancel)
        self.assertFalse(self.b.exists())
        self.assertFalse(list(self.root.glob('*.lock')))
        self.assertFalse(list(self.root.glob('.*.tmp.tif')))

    def test_existing_job_lock_is_respected(self):
        scene(self.a, np.ones((1, 16, 16), dtype="float32"))
        lock = self.b.with_name(self.b.name + '.lock')
        lock.write_text('another job')
        with self.assertRaises(FileExistsError):
            predict(self.a, self.b, method="bicubic")
        self.assertEqual(lock.read_text(), 'another job')

    @unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch unavailable')
    def test_bicubic_matches_training_kernel(self):
        rng = np.random.default_rng(4)
        for shape in ((3, 11, 17), (1, 1, 1), (4, 2, 3)):
            values = rng.random(shape).astype('float32')
            for scale in (2, 4, 8):
                expected = torch.nn.functional.interpolate(torch.from_numpy(values)[None], scale_factor=scale,
                                                          mode='bicubic', align_corners=False)[0].numpy()
                np.testing.assert_allclose(bicubic(values, scale), expected, atol=1e-6)

    def paired_scenes(self):
        low = np.random.default_rng(3).random((3, 8, 8)).astype('float32')
        high = np.repeat(np.repeat(low, 4, 1), 4, 2)
        for split, x in (('train', 500000), ('val', 600000)):
            scene(self.root / f'hr/{split}/scene.tif', high, x=x)
            scene(self.root / f'lr/{split}/scene.tif', low, resolution=40, x=x)
        return low

    def prepare_pairs(self):
        return prepare(self.root / 'hr/train', self.root / 'hr/val', self.root / 'prepared',
                       patch_size=32, stride=32, value_scale=1, lr_train_dir=self.root / 'lr/train', lr_val_dir=self.root / 'lr/val')

    @unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch unavailable')
    def test_real_pair_training_and_synchronized_augmentation(self):
        low = self.paired_scenes()
        self.assertEqual(self.prepare_pairs(), {'train': 1, 'val': 1})
        prepared = self.root / 'prepared'
        dataset = RasterPatches(prepared, 'train')
        for _ in range(8):
            lr, hr = dataset[0]
            torch.testing.assert_close(torch.nn.functional.avg_pool2d(hr, 4), lr)
        np.testing.assert_array_equal(RasterPatches(prepared, 'val')[0][0].numpy(), low)
        result = train(prepared, self.root / 'run', epochs=1, batch_size=1, features=8, blocks=1, device='cpu')
        predict(self.root / 'lr/val/scene.tif', self.b, checkpoint=result['checkpoint'], device='cpu')
        metrics = compare_rasters(self.b, self.root / 'hr/val/scene.tif', 1)
        self.assertTrue(np.isfinite(metrics['overall']['rmse']))

    def test_misaligned_pair_is_rejected(self):
        low = self.paired_scenes()
        scene(self.root / 'lr/val/scene.tif', low, resolution=40, x=600001)
        with self.assertRaisesRegex(ValueError, 'exactly aligned'):
            self.prepare_pairs()

    @unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch unavailable')
    def test_corrupt_checkpoint_normalization_is_rejected(self):
        model = SuperResolution(3, 4, 8, 1)
        checkpoint = self.root / 'bad.pt'
        torch.save({'format_version': 1, 'model_config': model.config, 'state_dict': model.state_dict(),
                    'value_scale': 0, 'value_offset': 0}, checkpoint)
        with self.assertRaisesRegex(ValueError, 'positive'):
            load_model(checkpoint)

    @unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch unavailable')
    def test_invalid_prepared_data_is_rejected(self):
        folder = self.root / 'prepared'
        folder.mkdir()
        manifest = {'scale': 4, 'bands': 3, 'patches': {'train': []}}
        file = folder / 'manifest.json'
        file.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'contain patches'):
            RasterPatches(folder, 'train')
        np.savez(self.root / 'outside.npz', hr=np.ones((3, 16, 16)))
        manifest['patches']['train'] = [{'file': '../outside.npz'}]
        file.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'inside'):
            RasterPatches(folder, 'train')
        np.savez(folder / 'bad.npz', hr=np.ones((3, 17, 16)))
        manifest['patches']['train'] = [{'file': 'bad.npz'}]
        file.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Invalid HR'):
            RasterPatches(folder, 'train')[0]

    @unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch unavailable')
    def test_evaluation_rejects_changed_band_metadata(self):
        from src.train import evaluate
        self.paired_scenes()
        self.prepare_pairs()
        folder = self.root / 'prepared'
        model = SuperResolution(3, 4, 8, 1)
        checkpoint = self.root / 'metadata.pt'
        torch.save({'format_version': 1, 'model_config': model.config, 'state_dict': model.state_dict(),
                    'value_scale': 1, 'value_offset': 0, 'band_descriptions': ['r', 'g', 'b']}, checkpoint)
        with self.assertRaisesRegex(ValueError, 'band_descriptions'):
            evaluate(folder, checkpoint, device='cpu')

    def test_launcher_stops_its_child_on_interrupt(self):
        import launch_gui
        process = MagicMock()
        process.poll.return_value = None
        process.wait.side_effect = [KeyboardInterrupt, 0]
        response = MagicMock()
        response.__enter__.return_value.status = 200
        with patch('sys.argv', ['launch_gui.py', '--no-browser']), patch.object(launch_gui.socket, 'socket'), \
             patch.object(launch_gui.subprocess, 'Popen', return_value=process) as launch, \
             patch.object(launch_gui.urllib.request, 'urlopen', return_value=response):
            launch_gui.main()
        process.terminate.assert_called_once()
        command = launch.call_args.args[0]
        self.assertEqual(command[command.index('--server.headless') + 1], 'true')


if __name__ == '__main__':
    unittest.main()
