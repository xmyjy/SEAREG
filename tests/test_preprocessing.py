"""CPU regression tests for peak-frequency filtering at the spot-count cutoff."""

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from seareg.config import PreprocessConfig
from seareg.defaults import training_defaults
from seareg.preprocessing import filter_atac


class PeakFrequencyCutoffTests(unittest.TestCase):
    def test_frequency_bounds_and_retained_peaks(self):
        cases = (
            (2499, 24, 2474),
            (2500, 25, 2475),
            (2501, 12, 2488),
            (2999, 14, 2984),
            (3000, 15, 2985),
        )
        for n_spots, minimum, maximum in cases:
            with self.subTest(n_spots=n_spots):
                frequencies = (minimum - 1, minimum, minimum + 1,
                               maximum - 1, maximum, maximum + 1, n_spots // 2)
                counts = np.zeros((n_spots, len(frequencies)), dtype=np.float32)
                for column, frequency in enumerate(frequencies):
                    counts[:frequency, column] = 7
                spots = tuple(f"spot_{index}" for index in range(n_spots))
                peaks = tuple(f"chr1:{10 * index}-{10 * index + 5}"
                              for index in range(len(frequencies)))
                original = sparse.csr_matrix(counts)
                adata = SimpleNamespace(
                    X=original, obs_names=spots, var_names=peaks,
                    obs=pd.DataFrame(index=spots), obsm={},
                )

                result, observed = filter_atac(
                    adata, PreprocessConfig(use_spatial=False),
                    return_frequency=True,
                )

                report = result.provenance["filter"]
                self.assertEqual((report["min_cells"], report["max_cells"]),
                                 (minimum, maximum))
                kept = (1, 2, 3, 4, 6)
                self.assertEqual(result.peak_ids, tuple(peaks[index] for index in kept))
                self.assertEqual(result.spot_ids, spots)
                np.testing.assert_array_equal(observed, frequencies)
                np.testing.assert_array_equal(result.X.toarray(),
                                              (counts[:, kept] > 0).astype(np.float32))
                np.testing.assert_array_equal(original.toarray(), counts)

    def test_epoch_defaults_use_the_same_inclusive_spot_boundary(self):
        for n_spots, epochs in ((2499, 300), (2500, 300), (2501, 500)):
            with self.subTest(n_spots=n_spots):
                self.assertEqual(training_defaults(n_spots, 700000)["epochs"], epochs)
                self.assertEqual(
                    training_defaults(n_spots, {"slice_a": 700000, "slice_b": 700000})["epochs"],
                    epochs,
                )


if __name__ == "__main__":
    unittest.main()
