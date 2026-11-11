"""Explicit new-data training defaults from filtered binary connections."""

from collections.abc import Mapping
from numbers import Integral
from pathlib import Path


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _single_slice_settings(n_spots):
    """Resolve the common spot threshold once for training and graph settings."""
    n_spots = _positive_integer(n_spots, "n_spots")
    if n_spots <= 2500:
        return dict(epochs=300, spatial_neighbors=4, negative_refresh=5, dense_mask_mb=512.)
    return dict(epochs=500, spatial_neighbors=3, negative_refresh=10, dense_mask_mb=2048.)


def training_defaults(n_spots, positive_edges, *, pn_r=None):
    """Return single- or multislice defaults from filtered graph statistics.

    ``positive_edges`` is the filtered binary spot-peak connection count before
    ``tr`` splitting or ``pn_r`` expansion. A mapping supplies one such count
    per slice. Multislice epochs are fixed at 500. The effective negative ratio
    (10 for single, 5 for multislice) determines the sampled-triplet budget.
    """
    n_spots = _positive_integer(n_spots, "n_spots")
    multislice = isinstance(positive_edges, Mapping)
    pn_r = _positive_integer((5 if multislice else 10) if pn_r is None else pn_r, "pn_r")

    def shards(edges):
        edges = _positive_integer(edges, "positive_edges")
        return max(10, 10 * ((edges * pn_r + 69_999_999) // 70_000_000))

    if multislice:
        if not positive_edges or any(not isinstance(name, str) or not name for name in positive_edges):
            raise ValueError("positive_edges requires nonempty slice names and counts")
        slice_shards = {name: shards(edges) for name, edges in positive_edges.items()}
        return {"epochs": 500, "pn_r": pn_r, "spatial_weight": .3,
                "spatial_radius_multiplier": 4., "negative_refresh": 5,
                "slice_shards": slice_shards,
                "updates_per_epoch": max(slice_shards.values())}
    values = _single_slice_settings(n_spots)
    values.pop('spatial_neighbors')
    return {**values, "pn_r": pn_r, "spatial_weight": .5, "shards": shards(positive_edges)}


def network_config(filtered, **overrides):
    """Resolve the radius before building a single-slice network.

    Explicit fields win independently. An explicitly selected KNN method keeps
    its neighbor count; the spot-dependent multiplier only applies to radius.
    """
    from .artifacts import FilteredCounts
    from .config import NetworkConfig

    data = FilteredCounts.load(filtered)
    values = {}
    if overrides.get('spatial_method', 'radius') == 'radius':
        values['spatial_neighbors'] = _single_slice_settings(data.metadata['shape'][0])['spatial_neighbors']
    return NetworkConfig(**{**values, **overrides})


def _resolve_workflow_network(config, *, filtered=None, network=None):
    """Share graph-option resolution between preparation and explicit analysis."""
    from dataclasses import asdict
    from .config import NetworkConfig

    current = asdict(config['network_config'])
    if network is not None:
        saved = dict(network.metadata['network_config'])
        explicit = (tuple(current) if config['source_config_version'] == 1
                    else config['network_explicit_fields'])
        if any(saved[name] != current[name] for name in explicit):
            raise ValueError('imported network belongs to different network configuration')
        config['network_config'] = NetworkConfig(**saved)
    elif config.get('network_auto_fields'):
        explicit = {name: value for name, value in current.items()
                    if name not in config['network_auto_fields']}
        config['network_config'] = network_config(filtered, **explicit)
    config['network_defaults_resolved'] = True


def training_config(network, **overrides):
    """Resolve a concrete ``TrainConfig`` from a saved model network.

    The validated network's filtered binary matrix supplies the statistics.
    Explicit options take precedence over the data-derived defaults.
    """
    from .artifacts import ModelNetwork
    from .config import TrainConfig

    data = ModelNetwork.load(network).load_data()
    values = training_defaults(data.X.shape[0], data.X.nnz, pn_r=overrides.get('pn_r', 10))
    return TrainConfig(**{**values, **overrides})


def multislice_training_config(counts, **overrides):
    """Resolve a ``MultiSliceConfig`` before creating immutable stages.

    Accept raw AnnData or an H5AD path. Apply the existing slice-aware filter
    with the caller's filtering options, then use each slice's distinct binary
    connections with its negative-sampling ratio. Explicit options always win;
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
    values = {**training_defaults(data.X.shape[0], positive_edges, pn_r=filter_config.pn_r), **overrides}
    if "updates_per_epoch" not in overrides:
        values["updates_per_epoch"] = max(values["slice_shards"].values())
    return MultiSliceConfig(**values)
