"""Spatial continuity of inferred tutorial domains in native section coordinates."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors


def spatial_domain_metrics(domain_result, *, pysal_ties=False):
    """Count spots and compute binary-domain Moran's I within each section.

    Six directed Euclidean neighbors receive equal row-standardized weights.
    Set ``pysal_ties=True`` to use the PySAL cKDTree convention on the E13
    spatial lattice. The default is sklearn's explicit-coordinate query,
    which is used for the P22 figure and the other tutorial datasets.
    """
    labels = domain_result.labels
    coordinates = np.asarray(domain_result.coordinates, dtype=float)[:, :2]
    domains = np.unique(labels["domain"])
    rows = []
    for slice_id in pd.unique(labels["slice_id"]):
        in_slice = labels["slice_id"].eq(slice_id).to_numpy()
        xy = coordinates[in_slice]
        slice_labels = labels["domain"].to_numpy()[in_slice]
        n_spots = len(xy)
        neighbors = None
        if n_spots > 6:
            if pysal_ties:
                _, candidates = cKDTree(xy, leafsize=10).query(xy, k=7)
            else:
                knn = NearestNeighbors(n_neighbors=7, metric="euclidean").fit(xy)
                candidates = knn.kneighbors(xy, return_distance=False)
            # Remove the query point by row identity, including tied coordinates.
            other = candidates != np.arange(n_spots)[:, None]
            other[other.all(axis=1), -1] = False
            neighbors = candidates[other].reshape(n_spots, 6)

        for domain in domains:
            indicator = slice_labels == domain
            count = int(indicator.sum())
            moran_i = np.nan
            if neighbors is not None and 0 < count < n_spots:
                centered = indicator.astype(float) - count / n_spots
                moran_i = float(centered @ centered[neighbors].mean(axis=1)
                                / (centered @ centered))
            rows.append({"slice_id": slice_id, "domain": domain, "n_spots": count,
                         "slice_n_spots": n_spots, "MoranI": moran_i})
    return pd.DataFrame(rows)
