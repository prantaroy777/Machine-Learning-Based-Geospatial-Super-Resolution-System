from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import shutil

import numpy as np
import rasterio
from rasterio.transform import from_origin
from streamlit.testing.v1 import AppTest

from app.app import save_upload


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'input.tif'
        with rasterio.open(self.source, 'w', driver='GTiff', width=20, height=16, count=3,
                           dtype='float32', crs='EPSG:32632', transform=from_origin(500000, 5200000, 40, 40)) as ds:
            ds.write(np.random.default_rng(1).random((3, 16, 20)).astype('float32'))
        self.app_path = Path(__file__).resolve().parents[1] / 'app/app.py'

    def test_local_file_export_and_persistent_result(self):
        app = AppTest.from_file(str(self.app_path), default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.button(key='generate').disabled)
        app.radio(key='source_mode').set_value('Local file path').run()
        app.text_input(key='source_path').set_value(str(self.source)).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(app.metric[1].value, '10 × 10 metre')
        app.button(key='generate').click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        result = app.session_state['result']
        output = Path(result['path'])
        self.addCleanup(output.unlink, missing_ok=True)
        self.addCleanup(output.with_suffix('.json').unlink, missing_ok=True)
        self.assertTrue(output.exists())
        with rasterio.open(output) as ds:
            self.assertEqual(ds.res, (10.0, 10.0))
        reference = Path(self.temp.name) / 'reference.tif'
        shutil.copyfile(output, reference)
        app.text_input(key='reference').set_value(str(reference)).run()
        app.button(key='compare').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state['result']['comparison']['overall']['exact_match'])
        app.selectbox(key='scale').set_value(2).run()
        self.assertEqual(app.session_state['result']['path'], str(output))
        self.assertEqual(app.session_state['result']['scale'], 4)
        app.radio(key='method').set_value('Trained model — checkpoint required').run()
        self.assertTrue(app.button(key='generate').disabled)
        app.text_input(key='checkpoint').set_value('missing-checkpoint.pt').run()
        self.assertTrue(app.button(key='generate').disabled)
        self.assertTrue(app.error)
        output.unlink()
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any('moved or deleted' in warning.value for warning in app.warning))

    def test_missing_raster_shows_error(self):
        app = AppTest.from_file(str(self.app_path), default_timeout=20).run()
        app.radio(key='source_mode').set_value('Local file path').run()
        app.text_input(key='source_path').set_value(str(Path(self.temp.name) / 'missing.tif')).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertTrue(app.button(key='generate').disabled)

    def test_upload_preserves_geotiff_and_deduplicates(self):
        upload = BytesIO(self.source.read_bytes())
        upload.name = '../../unsafe-name.tif'
        directory = Path(self.temp.name) / 'uploads'
        saved = save_upload(upload, directory)
        self.assertEqual(saved.parent, directory)
        self.assertEqual(saved.read_bytes(), self.source.read_bytes())
        self.assertEqual(save_upload(upload, directory), saved)
        self.assertEqual(len(list(directory.iterdir())), 1)
        with rasterio.open(saved) as ds:
            self.assertEqual(ds.res, (40., 40.))
            self.assertEqual(ds.crs.to_epsg(), 32632)

    def test_demo_works_without_bundled_data(self):
        existing_demo = self.app_path.parents[1] / 'data/demo/synthetic_40m.tif'
        original = Path.is_file
        with patch.object(Path, 'is_file', lambda path: False if path == existing_demo else original(path)):
            app = AppTest.from_file(str(self.app_path), default_timeout=20).run()
            app.radio(key='source_mode').set_value('Try the demo').run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(app.metric[0].value, '40 × 40 metre')
        self.assertFalse(app.button(key='generate').disabled)
        demo = self.app_path.parents[1] / 'outputs/gui' / app.session_state['session_id'] / 'demo/synthetic_40m.tif'
        self.addCleanup(demo.unlink, missing_ok=True)


if __name__ == '__main__':
    unittest.main()
