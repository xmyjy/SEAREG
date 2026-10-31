"""Shared analysis metadata and fit-free artifact persistence."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from .config import AnalysisConfig
from .utils import artifact_writer, environment, identity, verify_artifact, write_json
from .utils import protect_output


DISPLAY = {
    "spatial_y_axis": "native_increasing_y",
    "y_axis_inverted": False,
    "spatial_point_size": 20,
    "point_size_units": "points_squared",
    "spot_umap": {
        "figsize_mm": [75, 48.35], "spot_size": 2, "spot_alpha": 1,
        "aspect": "auto", "bbox_inches": None, "dpi": 300,
    },
    "joint_umap": {
        "figsize_mm": [75, 113.2], "spot_size": 5.2, "spot_alpha": 0.65,
        "peak_size": 2.4, "peak_alpha": 0.22, "peak_marker": "^", "peak_color": "#CFCFCF",
        "motif_size": 4.2, "motif_alpha": 0.20, "motif_marker": "s", "motif_color": "#2C2C2C",
        "aspect": "equal", "bbox_inches": None, "dpi": 300, "all_nodes_retained": True,
    },
    "dpi": 450,
    "panel_figsize": [3.3, 3.35],
    "figsize_1x4": [13.2, 3.35],
    "annotate_domains": False,
    "label_smoothing": False,
    "color_alignment": False,
    "domain_labels_scope": "local_to_each_dataset",
}


def _inputs(value):
    from .data import EmbeddingSet

    if not isinstance(value, EmbeddingSet):
        loader = getattr(value, "load_embeddings", None)
        if not callable(loader):
            raise TypeError("inputs must be an EmbeddingSet or expose load_embeddings()")
        value = loader()
    if not isinstance(value, EmbeddingSet):
        raise TypeError("load_embeddings() must return an EmbeddingSet")
    # Revalidate mutable frames and copy them before any numerical operation.
    return EmbeddingSet(value.spot, value.peak, value.motif, value.dna_kind,
                        value.coordinates, value.slice_ids, value.provenance)


def _config(config):
    if not isinstance(config, AnalysisConfig):
        raise TypeError("config must be AnalysisConfig")
    return config


def _output_available(output):
    if output is not None:
        path = Path(output)
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        protect_output(path)


def _metadata(inputs, config):
    metadata = {
        "protocol_id": (
            "copy1_exact_spatial_v1" if config.neighbor_backend == "exact"
            else "copy1_original_spatial_v1"
        ),
        "joint_protocol_id": "direct_joint_v1",
        "config": asdict(config),
        "counts": {"spot": len(inputs.spot), "peak": 0 if inputs.peak is None else len(inputs.peak)},
        "provenance": inputs.provenance,
        "environment": environment(),
        "input_identities": {},
        "display": dict(DISPLAY),
    }
    for kind, frame in (("spot", inputs.spot), ("peak", inputs.peak), (inputs.dna_kind, inputs.motif)):
        if frame is not None:
            metadata["counts"][kind] = len(frame)
            metadata["input_identities"][kind] = {
                "ids_hash": identity(tuple(frame.index)),
                "matrix_sha256": hashlib.sha256(frame.to_numpy(dtype=np.float64).tobytes()).hexdigest(),
                "shape": list(frame.shape),
            }
    metadata["input_identities"]["slice_ids_hash"] = identity(tuple(inputs.slice_ids))
    if inputs.coordinates is not None:
        metadata["input_identities"]["coordinates"] = {
            "matrix_sha256": hashlib.sha256(inputs.coordinates.tobytes()).hexdigest(),
            "shape": list(inputs.coordinates.shape),
            "dtype": str(inputs.coordinates.dtype),
        }
    return metadata


def _write_spots(stage, inputs):
    table = pd.DataFrame({"spot_id": inputs.spot.index, "slice_id": inputs.slice_ids})
    if inputs.coordinates is not None:
        np.save(stage / "spot_coordinates.npy", inputs.coordinates, allow_pickle=False)
        for j in range(inputs.coordinates.shape[1]):
            table[("x", "y")[j] if j < 2 else f"spatial_{j}"] = inputs.coordinates[:, j]
    table.to_csv(stage / "spot_metadata.tsv", sep="\t", index=False)


def _write_graph(stage, graph, metadata):
    np.save(stage / "spot_pca.npy", graph.obsm["X_pca"], allow_pickle=False)
    np.save(stage / "spot_input_l2.npy", np.asarray(graph.X), allow_pickle=False)
    sp.save_npz(stage / "spot_connectivities.npz", graph.obsp["connectivities"])
    if "distances" in graph.obsp:
        sp.save_npz(stage / "spot_distances.npz", graph.obsp["distances"])
    metadata["spatial_preprocessing"] = graph.uns["modular_preprocessing"]
    write_json(stage / "spot_graph_metadata.json", {
        "neighbors": graph.uns.get("neighbors", {}),
        "modular_preprocessing": graph.uns["modular_preprocessing"],
    })


def _read_table(path, required):
    text = {name: str for name in ("spot_id", "slice_id", "domain", "entity_type", "entity_id")}
    table = pd.read_csv(path, sep="\t", dtype=text, keep_default_na=False)
    if not set(required) <= set(table.columns) or table.empty:
        raise ValueError(f"invalid saved table schema: {Path(path).name}")
    for name in set(text) & set(table.columns):
        if table[name].eq("").any():
            raise ValueError(f"empty saved identity: {name}")
    return table


def _read_spots(path):
    table = _read_table(path / "spot_metadata.tsv", ("spot_id", "slice_id"))
    if table.spot_id.duplicated().any():
        raise ValueError("duplicate saved spot IDs")
    coordinates = None
    if (path / "spot_coordinates.npy").is_file():
        coordinates = np.load(path / "spot_coordinates.npy", allow_pickle=False)
        if (coordinates.ndim != 2 or coordinates.shape[0] != len(table)
                or coordinates.shape[1] < 2 or not np.isfinite(coordinates).all()):
            raise ValueError("invalid saved spatial coordinates")
    return table, coordinates


def _read_graph(path, spot_ids, metadata):
    if not (path / "spot_connectivities.npz").is_file():
        return None
    import anndata as ad

    n = len(spot_ids)
    normalized = (np.load(path / "spot_input_l2.npy", allow_pickle=False)
                  if (path / "spot_input_l2.npy").is_file() else np.empty((n, 0)))
    graph = ad.AnnData(normalized)
    graph.obs_names = pd.Index(spot_ids)
    graph.obsm["X_pca"] = np.load(path / "spot_pca.npy", allow_pickle=False)
    graph.obsp["connectivities"] = sp.load_npz(path / "spot_connectivities.npz")
    if (path / "spot_distances.npz").is_file():
        graph.obsp["distances"] = sp.load_npz(path / "spot_distances.npz")
    if (path / "spot_graph_metadata.json").is_file():
        graph.uns.update(json.loads((path / "spot_graph_metadata.json").read_text()))
    else:
        # Historical artifacts have PCA/connectivities but did not serialize X,
        # distances or neighbors. Loading never reconstructs missing science.
        graph.uns["modular_preprocessing"] = metadata.get("spatial_preprocessing", {})
        graph.uns["historical_graph_fields_missing"] = ["X", "distances", "neighbors"]
    return graph


def _figure_metadata(fig, *, export_dpi=450):
    """Record the actual handles rendered, rather than only a style recipe."""
    return {
        "figsize_inches": fig.get_size_inches().tolist(),
        "export_dpi": export_dpi,
        "panels": [{
            "title": ax.get_title(),
            "ylim": [float(x) for x in ax.get_ylim()],
            "y_axis_inverted": bool(ax.yaxis_inverted()),
            "scatter_sizes": [collection.get_sizes().tolist() for collection in ax.collections],
            "points": sum(len(collection.get_offsets()) for collection in ax.collections),
        } for ax in fig.axes],
    }
