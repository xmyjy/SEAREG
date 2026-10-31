"""Single/multislice result-save policy must share the same runtime cadence.

Only stdlib configuration modules are loaded: no Torch, training or GPU access.
"""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_source(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SINGLE = load_source("_save_policy_single", "seareg/config.py").TrainConfig
MULTI = load_source("_save_policy_multi", "seareg/multislice/config.py").MultiSliceConfig


class SavePolicyTests(unittest.TestCase):
    def test_five_hundred_epochs_export_only_300_400_500(self):
        for kind in (SINGLE, MULTI):
            with self.subTest(config=kind.__name__):
                config = kind(epochs=500)
                self.assertEqual((config.export_start_epoch, config.export_every), (300, 100))
                self.assertEqual([epoch for epoch in range(1, 501)
                                  if config.result_export_due(epoch)], [300, 400, 500])

    def test_short_runs_always_export_their_final_epoch(self):
        for kind in (SINGLE, MULTI):
            for epochs in (1, 49, 99, 299):
                with self.subTest(config=kind.__name__, epochs=epochs):
                    config = kind(epochs=epochs)
                    self.assertEqual([epoch for epoch in range(1, epochs + 1)
                                      if config.result_export_due(epoch)], [epochs])

    def test_off_cadence_final_epoch_still_exports(self):
        for kind in (SINGLE, MULTI):
            with self.subTest(config=kind.__name__):
                config = kind(epochs=450)
                self.assertEqual([epoch for epoch in range(1, 451)
                                  if config.result_export_due(epoch)], [300, 400, 450])

    def test_rolling_checkpoint_and_scientific_refresh_stay_separate(self):
        for kind in (SINGLE, MULTI):
            with self.subTest(config=kind.__name__):
                self.assertEqual(kind().checkpoint_every, 10)
        config = MULTI()
        self.assertEqual((config.gate_refresh_every, config.reshard_every,
                          config.negative_refresh, config.updates_per_epoch),
                         (50, 100, 5, 20))

    def test_t6_profile_uses_same_shared_save_policy(self):
        profile = json.loads((ROOT / "tutorials/configs/Embryo_multislice_adaptive_depth_gate.json")
                             .read_text(encoding="utf-8"))
        config = MULTI(**profile["training"])
        self.assertEqual((config.export_start_epoch, config.export_every, config.checkpoint_every),
                         (300, 100, 10))
        self.assertEqual([epoch for epoch in range(1, 501)
                          if config.result_export_due(epoch)], [300, 400, 500])
        self.assertEqual(config.gate_refresh_every, 50)

    def test_publication_check_rejects_old_50_epoch_save_policy(self):
        checker = load_source("_save_policy_checker", "tools/check_multislice_release.py")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for relative in ("seareg/multislice/config.py",
                             "tutorials/configs/Embryo_multislice_adaptive_depth_gate.json",
                             "pyproject.toml", "seareg/__init__.py", "CITATION.cff"):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
            path = root / "tutorials/configs/Embryo_multislice_adaptive_depth_gate.json"
            profile = json.loads(path.read_text(encoding="utf-8"))
            profile["training"].update(export_start_epoch=50, export_every=50)
            path.write_text(json.dumps(profile), encoding="utf-8")
            failures = checker.check_multislice_release(root)
            self.assertTrue(any("export_start_epoch must be 300" in item for item in failures), failures)
            self.assertTrue(any("export_every must be 100" in item for item in failures), failures)


if __name__ == "__main__":
    unittest.main()
