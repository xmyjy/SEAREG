"""Explicit spot and joint UMAP fitting and fit-free saved embeddings."""

import numpy as np
import pandas as pd
from .data import EmbeddingSet
from dataclasses import dataclass
from pathlib import Path
import scipy.sparse as sp
from .utils import verify_artifact
from ._analysis_io import _read_graph, _read_spots, _read_table


def compute_spot_umap(graph, *, min_dist=0.08, _config=None):
    """Compute Scanpy spot UMAP from the spatial-domain graph.

    Return spot_id, UMAP1 and UMAP2 in the input graph's spot order.
    """
    import scanpy as sc

    working = graph.copy()
    seed = 1234 if _config is None else _config.seed
    sc.tl.umap(working, min_dist=min_dist, random_state=seed)
    coordinates = working.obsm["X_umap"]
    if _config is not None and not np.isfinite(coordinates).all():
        raise ValueError("non-finite spot UMAP")
    return pd.DataFrame({"spot_id": working.obs_names,
                         "UMAP1": coordinates[:, 0], "UMAP2": coordinates[:, 1]})


def compute_umap(data, *, min_dist=0.08, _config=None, _graph=None):
    """Compute spot-only or joint UMAP with the main-figure workflows.

    Pass a spatial-domain graph for spot-only UMAP, or a dictionary of
    spot/peak and optional motif embedding DataFrames for joint UMAP.
    Return entity_type, entity_id, UMAP1 and UMAP2 in input node order.
    """
    if not isinstance(data, dict) and not (_config is not None and isinstance(data, EmbeddingSet)):
        table = compute_spot_umap(data, min_dist=min_dist, _config=_config).rename(
            columns={"spot_id": "entity_id"})
        table.insert(0, "entity_type", "spot")
        return table

    import umap

    if isinstance(data, dict):
        entities = [("spot", data["spot"]), ("peak", data["peak"])]
        if data.get("motif") is not None:
            entities.append(("motif", data["motif"]))
    else:
        if data.peak is None:
            raise ValueError("direct joint UMAP requires named peak embeddings")
        entities = [("spot", data.spot), ("peak", data.peak)]
        if data.motif is not None:
            entities.append((data.dna_kind, data.motif))

    normalized = []
    for kind, frame in entities:
        if kind == "spot" and _graph is not None:
            normalized.append(np.asarray(_graph.X))
            continue
        values = frame.to_numpy(dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            norm = np.linalg.norm(values, axis=1, keepdims=True)
        if _config is not None and (not np.isfinite(norm).all() or np.any(norm == 0)):
            raise ValueError("row-L2 requires finite nonzero row norms without overflow")
        normalized.append(values / np.maximum(norm, 1e-12))
    values = np.vstack(normalized)
    neighbors = 15 if _config is None else _config.neighbors
    seed = 1234 if _config is None else _config.seed
    if _config is not None and neighbors >= len(values):
        raise ValueError("joint UMAP neighbors exceed node population")
    model = umap.UMAP(
        n_neighbors=neighbors, min_dist=min_dist, n_components=2,
        metric="euclidean", random_state=seed, n_jobs=1,
    )
    coordinates = model.fit_transform(values)
    if _config is not None and not np.isfinite(coordinates).all():
        raise ValueError("non-finite joint UMAP")

    rows = [(kind, name, i) for kind, frame in entities for i, name in enumerate(frame.index)]
    table = pd.DataFrame(rows, columns=["entity_type", "entity_id", "native_row"])
    table["UMAP1"] = coordinates[:, 0]
    table["UMAP2"] = coordinates[:, 1]
    if _config is not None:
        # These are the products of this fit used by the existing artifact writer.
        return EmbeddingResult(table, normalized=values, graph=model.graph_.tocsr())
    return table.drop(columns="native_row")


@dataclass
class EmbeddingResult:
    table: pd.DataFrame
    path: Path | None = None
    normalized: np.ndarray | None = None
    graph: object | None = None


def _read(path, metadata, kind, *, _reader=None):
    if kind == "spot":
        table = _read_table(path / "spot_umap.tsv", ("spot_id", "UMAP1", "UMAP2"))
        spots, _ = _read_spots(path) if _reader is None else _reader.read_spots()
        if table.spot_id.tolist() != spots.spot_id.tolist():
            raise ValueError("saved spot UMAP identity/order mismatch")
        graph = _read_graph(path, spots.spot_id, metadata) if _reader is None else _reader.read_graph()
        normalized = None
        if (path / "spot_input_l2.npy").is_file():
            # The graph has already read this matrix. Preserve the independent
            # array returned by historical loaders without a second disk read.
            normalized = (np.asarray(graph.X).copy() if graph is not None else
                          np.load(path / "spot_input_l2.npy", allow_pickle=False))
    else:
        table = _read_table(path / "joint_umap.tsv",
                            ("entity_type", "entity_id", "native_row", "UMAP1", "UMAP2"))
        if table.duplicated(["entity_type", "entity_id"]).any():
            raise ValueError("duplicate saved typed joint IDs")
        rows = pd.to_numeric(table.native_row, errors="raise").to_numpy()
        if not np.array_equal(rows, rows.astype(np.int64)):
            raise ValueError("invalid saved native row")
        table["native_row"] = rows.astype(np.int64)
        for _, group in table.groupby("entity_type", sort=False):
            if group.native_row.tolist() != list(range(len(group))):
                raise ValueError("saved typed entity rows are out of native order")
        normalized = np.load(path / "joint_input_l2.npy", allow_pickle=False)
        graph = sp.load_npz(path / "joint_connectivities.npz")
        if normalized.ndim != 2 or normalized.shape[0] != len(table) or graph.shape != (len(table), len(table)):
            raise ValueError("saved joint inputs/graph do not match table")
    coordinates = table[["UMAP1", "UMAP2"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("non-finite saved UMAP coordinates")
    return EmbeddingResult(table, path, normalized, graph)


def load_result(path):
    """Verify and load a standalone spot_embedding or joint_embedding artifact."""
    path = Path(path).resolve()
    manifest = verify_artifact(path)
    if manifest["kind"] not in ("spot_embedding", "joint_embedding"):
        raise ValueError("expected standalone spot_embedding or joint_embedding artifact")
    return _read(path, manifest["metadata"], "spot" if manifest["kind"] == "spot_embedding" else "joint")


def _load_from_analysis(path, kind):
    from .analysis import AnalysisResult, load_result as load_analysis
    result = path if isinstance(path, AnalysisResult) else load_analysis(path)
    with result._get_reader().operation() as reader:
        requested = "spot_umap" if kind == "spot" else "joint"
        if not reader.metadata["config"].get(requested, False):
            raise ValueError(f"{kind} UMAP was not saved in this analysis")
        return _read(reader.path, reader.metadata, kind, _reader=reader)
