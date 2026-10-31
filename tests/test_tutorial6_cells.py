"""Behavior regressions for T6's lightweight orchestration cells.

These execute the actual notebook sources with tiny handles. They never launch
training or fit labels/embeddings.
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "tutorials/T6_Mouse_Brain_Development_MISAR_seq.ipynb"


def source(cell_id):
    notebook = json.loads(NB_PATH.read_text(encoding="utf-8"))
    return "".join(next(c["source"] for c in notebook["cells"] if c["id"] == cell_id))


class Tutorial6CellsTests(unittest.TestCase):
    def run_reopen(self, state):
        seen = []
        labels = pd.DataFrame({"domain": ["0"]})
        coords = pd.DataFrame({"UMAP1": [0.0], "UMAP2": [1.0]})
        saved = types.SimpleNamespace(
            path=Path("/current/analysis"),
            metadata={"config": {"domains": True, "spot_umap": True, "joint": True}},
            domains=lambda: types.SimpleNamespace(labels=labels),
            spot_embedding=lambda: types.SimpleNamespace(table=coords),
            joint_embedding=lambda: types.SimpleNamespace(table=coords),
        )
        analysis = types.ModuleType("seareg.analysis")
        def load_result(path):
            seen.append(Path(path))
            saved.path = Path(path).resolve()
            return saved
        analysis.load_result = load_result
        state.update(domain_result=types.SimpleNamespace(labels=labels),
                     spot_result=types.SimpleNamespace(table=coords),
                     joint_result=types.SimpleNamespace(table=coords))
        with patch.dict(sys.modules, {"seareg.analysis": analysis}), contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source("t6-reopen"), str(NB_PATH), "exec"), state)
        return seen, state

    def test_reopen_prefers_current_result_over_previous_resolved_path(self):
        seen, state = self.run_reopen({
            "result": types.SimpleNamespace(path=Path("/current/analysis")),
            "ANALYSIS_PATH": Path("/stale/analysis"),
        })
        self.assertEqual(seen, [Path("/current/analysis")])
        self.assertEqual(state["ANALYSIS_PATH"], Path("/current/analysis"))

    def test_reopen_supports_explicit_standalone_override(self):
        seen, state = self.run_reopen({
            "REOPEN_ANALYSIS_PATH": Path("/explicit/saved"),
        })
        self.assertEqual(seen, [Path("/explicit/saved")])


    def test_network_preflight_inspects_saved_graph_and_native_coordinates(self):
        calls = []
        coordinates = object()
        graph = object()
        geometry = types.ModuleType("seareg.multislice.geometry")
        report = dict(total_spots=3, unit_cardinal_distance=1.0, cardinal_tolerance=1e-6,
                      per_slice=[dict(slice_code=0, spots=1, eight_sector_covered=0,
                                      eight_sector_fraction=0.0, cardinal_covered=0, cardinal_fraction=0.0),
                                 dict(slice_code=1, spots=2, eight_sector_covered=0,
                                      eight_sector_fraction=0.0, cardinal_covered=0, cardinal_fraction=0.0)])
        def coverage(xy, spatial, codes):
            calls.append((xy, spatial, list(codes)))
            return report
        geometry.spatial_coverage_report = coverage
        data = types.SimpleNamespace(
            spot_ids=("a", "b", "c"), peak_ids=("p",), coordinates=coordinates,
            slice_ids=("E15", "E11", "E15"), X=types.SimpleNamespace(nnz=3))
        network = types.SimpleNamespace(load_data=lambda: data, load_graphs=lambda: {"spatial": graph},
                                        fingerprint="example", metadata={"graph_report": []})
        with tempfile.TemporaryDirectory() as folder:
            state = dict(RUN_MODE="train", time=time, sr=types.SimpleNamespace(build_multislice_network=lambda *_: network),
                         filtered_counts=object(), training_config=object(), RUN_DIR=Path(folder),
                         stage_seconds={}, pd=pd, display=lambda *_: None, json=json)
            with patch.dict(sys.modules, {"seareg.multislice.geometry": geometry}), contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source("df4b189f"), str(NB_PATH), "exec"), state)
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0][0], coordinates)
            self.assertIs(calls[0][1], graph)
            self.assertEqual(calls[0][2], [1, 0, 1])
            saved = json.loads((Path(folder) / "spatial_coverage.json").read_text())
            self.assertEqual(saved["total_spots"], 3)
            self.assertEqual([x["sample_id"] for x in saved["per_slice"]], ["E11", "E15"])


if __name__ == "__main__":
    unittest.main()
