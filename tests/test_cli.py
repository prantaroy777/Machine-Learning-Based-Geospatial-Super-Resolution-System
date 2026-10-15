"""User-facing CLI workflow using only public commands and synthetic data."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import rasterio

ROOT = Path(__file__).resolve().parents[1]


class CommandLineTests(unittest.TestCase):
    def command(self, *args, expected=0):
        result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_demo_export_inspection_reference_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            demo, output, report = root / 'demo', root / 'out.tif', root / 'report.json'
            self.command('examples/make_demo.py', '--output', demo)
            source = demo / 'synthetic_40m.tif'
            self.command('main.py', 'predict', source, output, '--method', 'bicubic', '--build-overviews')
            metadata = json.loads(self.command('main.py', 'inspect', output).stdout)
            self.assertEqual(metadata['pixel_size'], [10., 10.])
            self.assertEqual((metadata['width'], metadata['height']), (256, 256))
            with rasterio.open(source) as a, rasterio.open(output) as b:
                self.assertEqual(a.bounds, b.bounds)
            self.command('main.py', 'compare', output, demo / 'val/synthetic_10m.tif',
                         '--data-range', 1, '--report', report)
            saved = report.read_bytes()
            self.assertEqual(json.loads(saved)['valid_pixels'], 256 * 256)
            self.command('main.py', 'compare', output, demo / 'val/synthetic_10m.tif',
                         '--data-range', 1, '--report', report, expected=1)
            self.assertEqual(saved, report.read_bytes())
            self.command('main.py', 'predict', source, root / 'untrained.tif', expected=1)
            self.assertFalse((root / 'untrained.tif').exists())

    def test_dependency_check_does_not_start_a_server(self):
        result = self.command('launch_gui.py', '--check')
        self.assertIn('No server started', result.stdout)
