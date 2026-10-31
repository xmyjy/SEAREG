"""Explicit spot and joint UMAP fitting and fit-free saved embeddings."""

import numpy as np
import pandas as pd
from .data import row_l2
from dataclasses import dataclass
from pathlib import Path
import scipy.sparse as sp
from .utils import artifact_writer, environment, verify_artifact
from ._analysis_io import _config, _inputs, _metadata, _output_available, _read_graph, _read_spots, _read_table, _write_graph, _write_spots


def spot_umap(graph, config):
    import scanpy as sc

    working = graph.copy()
    sc.tl.umap(working, min_dist=config.min_dist, random_state=config.seed)
    values = np.array(working.obsm["X_umap"], copy=True)
    if not np.isfinite(values).all():
        raise ValueError("non-finite spot UMAP")
    return values


def joint_umap(inputs, config):
    import umap

    if inputs.peak is None:
        raise ValueError("direct joint UMAP requires named peak embeddings")
    entities = [("spot", inputs.spot), ("peak", inputs.peak)]
    if inputs.motif is not None:
        entities.append((inputs.dna_kind, inputs.motif))
    values = np.vstack([row_l2(frame.to_numpy()) for _, frame in entities])
    if config.neighbors >= len(values):
        raise ValueError("joint UMAP neighbors exceed node population")
    rows = [(kind, name, i) for kind, frame in entities for i, name in enumerate(frame.index)]
    table = pd.DataFrame(rows, columns=["entity_type", "entity_id", "native_row"])
    model = umap.UMAP(
        n_neighbors=config.neighbors,
        min_dist=config.min_dist,
        n_components=2,
        metric="euclidean",
        random_state=config.seed,
        n_jobs=1,
    )
    coordinates = model.fit_transform(values)
    if not np.isfinite(coordinates).all():
        raise ValueError("non-finite joint UMAP")
    table["UMAP1"] = coordinates[:, 0]
    table["UMAP2"] = coordinates[:, 1]
    return table, values, model


@dataclass
class EmbeddingResult:
    table: pd.DataFrame
    path: Path | None = None
    normalized: np.ndarray | None = None
    graph: object | None = None


def fit_spot(inputs, config, output=None):
    """Run the existing row-L2/PCA/neighbors/spot UMAP method explicitly."""
    _output_available(output)
    config = _config(config)
    inputs = _inputs(inputs)
    from .domains import build_spot_graph

    graph = build_spot_graph(inputs.spot, config)
    coordinates = spot_umap(graph, config)
    table = pd.DataFrame({"spot_id": inputs.spot.index,
                          "UMAP1": coordinates[:, 0], "UMAP2": coordinates[:, 1]})
    result = EmbeddingResult(table, normalized=np.asarray(graph.X).copy(), graph=graph)
    if output is not None:
        metadata = _metadata(inputs, config)
        with artifact_writer(output, "spot_embedding", metadata) as stage:
            _write_spots(stage, inputs)
            _write_graph(stage, graph, metadata)
            table.to_csv(stage / "spot_umap.tsv", sep="\t", index=False)
            metadata["environment"] = environment()
        result.path = Path(output).resolve()
    return result


def fit_joint(inputs, config, output=None):
    """Run direct UMAP over all typed nodes using per-row float64 L2.

    The implementation preserves entity_type/entity_id/native_row and uses no joint
    PCA, feature-to-spot alignment, or subsampling.
    """
    _output_available(output)
    config = _config(config)
    inputs = _inputs(inputs)

    table, normalized, model = joint_umap(inputs, config)
    graph = model.graph_.tocsr()
    result = EmbeddingResult(table, normalized=normalized, graph=graph)
    if output is not None:
        metadata = _metadata(inputs, config)
        metadata["joint_effective_parameters"] = {
            "neighbors": model.n_neighbors, "min_dist": model.min_dist,
            "metric": model.metric, "seed": config.seed, "PCA": False,
            "feature_to_spot_alignment": False,
        }
        with artifact_writer(output, "joint_embedding", metadata) as stage:
            table.to_csv(stage / "joint_umap.tsv", sep="\t", index=False)
            np.save(stage / "joint_input_l2.npy", normalized, allow_pickle=False)
            sp.save_npz(stage / "joint_connectivities.npz", graph)
            metadata["environment"] = environment()
        result.path = Path(output).resolve()
    return result


def _read(path, metadata, kind):
    if kind == "spot":
        table = _read_table(path / "spot_umap.tsv", ("spot_id", "UMAP1", "UMAP2"))
        spots, _ = _read_spots(path)
        if table.spot_id.tolist() != spots.spot_id.tolist():
            raise ValueError("saved spot UMAP identity/order mismatch")
        graph = _read_graph(path, spots.spot_id, metadata)
        normalized = (np.load(path / "spot_input_l2.npy", allow_pickle=False)
                      if (path / "spot_input_l2.npy").is_file() else None)
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
    from .analysis import load_result as load_analysis
    result = load_analysis(path)
    requested = "spot_umap" if kind == "spot" else "joint"
    if not result.metadata["config"].get(requested, False):
        raise ValueError(f"{kind} UMAP was not saved in this analysis")
    return _read(result.path, result.metadata, kind)
