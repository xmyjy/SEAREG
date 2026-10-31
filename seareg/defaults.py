"""Explicit new-data training defaults from filtered binary connections."""

from collections.abc import Mapping
from numbers import Integral
from pathlib import Path


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def training_defaults(n_spots, positive_edges):
    """Return training defaults from total spots and distinct positive edges.

    ``positive_edges`` is the filtered binary spot-peak connection count before
    ``tr`` splitting or ``pn_r`` expansion. A mapping supplies one such count
    per slice; ``n_spots`` remains the total joint spot count.
    """
    n_spots = _positive_integer(n_spots, "n_spots")
    epochs = 500 if n_spots > 2500 else 300

    def shards(edges):
        edges = _positive_integer(edges, "positive_edges")
        return max(10, (edges + 699999) // 700000)

    if isinstance(positive_edges, Mapping):
        if not positive_edges or any(not isinstance(name, str) or not name for name in positive_edges):
            raise ValueError("positive_edges requires nonempty slice names and counts")
        slice_shards = {name: shards(edges) for name, edges in positive_edges.items()}
        return {"epochs": epochs, "slice_shards": slice_shards,
                "updates_per_epoch": max(slice_shards.values())}
    return {"epochs": epochs, "shards": shards(positive_edges)}


def training_config(network, **overrides):
    """Resolve a concrete ``TrainConfig`` from a saved model network.

    The validated network's filtered binary matrix supplies the statistics.
    Explicit options take precedence over the data-derived defaults.
    """
    from .artifacts import ModelNetwork
    from .config import TrainConfig

    data = ModelNetwork.load(network).load_data()
    values = training_defaults(data.X.shape[0], data.X.nnz)
    return TrainConfig(**{**values, **overrides})


def multislice_training_config(counts, **overrides):
    """Resolve a ``MultiSliceConfig`` before creating immutable stages.

    Accept raw AnnData or an H5AD path. Apply the existing slice-aware filter
    with the caller's filtering options, then use each slice's distinct binary
    connections and the total joint spot count. Explicit options always win;
    otherwise updates per epoch follow the largest final slice shard count.
    """
    import anndata as ad
    from .multislice.config import MultiSliceConfig
    from .multislice.inputs import _filter_multislice_data

    a = ad.read_h5ad(counts) if isinstance(counts, (str, Path)) else counts
    if not isinstance(a, ad.AnnData):
        raise TypeError("counts must be AnnData or an h5ad path")
    slice_key = overrides.get("slice_key", "sample_id")
    if not isinstance(slice_key, str) or not slice_key:
        raise ValueError("slice_key must be a string")
    if slice_key not in a.obs:
        raise ValueError(f"missing slice key {slice_key}")
    slice_ids = tuple(a.obs[slice_key].astype(str))
    filter_config = MultiSliceConfig(**{
        "slice_shards": dict.fromkeys(slice_ids, 10), **overrides})
    data = _filter_multislice_data(a, filter_config)
    positions = {}
    for index, name in enumerate(data.slice_ids):
        positions.setdefault(name, []).append(index)
    positive_edges = {name: int(data.X[indices].nnz) for name, indices in positions.items()}
    values = {**training_defaults(data.X.shape[0], positive_edges), **overrides}
    if "updates_per_epoch" not in overrides:
        values["updates_per_epoch"] = max(values["slice_shards"].values())
    return MultiSliceConfig(**values)
