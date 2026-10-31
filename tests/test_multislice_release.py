"""Publication consistency checks for the unique multislice implementation."""
from pathlib import Path
import importlib.util
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MultiSliceReleaseTests(unittest.TestCase):
    def checker(self):
        path = ROOT / 'tools/check_multislice_release.py'
        self.assertTrue(path.is_file(), 'multislice publication checker is missing')
        spec = importlib.util.spec_from_file_location('multislice_release_check', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.check_multislice_release

    def test_current_release_is_consistent(self):
        self.assertEqual(self.checker()(ROOT), [])

    def test_deleted_asset_reference_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'docs').mkdir()
            (root / 'docs/multislice.rst').write_text(
                'Use result/Embryo_multislice/coherent_boundary/500_epoch/',
                encoding='utf-8')
            failures = self.checker()(root)
            self.assertTrue(any('docs/multislice.rst' in failure and
                                'coherent_boundary' in failure for failure in failures))


if __name__ == '__main__':
    unittest.main()
