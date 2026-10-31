"""Exact-K or fixed-resolution spatial domains and fit-free saved partitions."""

import inspect
import numpy as np
from .config import AnalysisConfig
from .data import row_l2, validated_frame
from dataclasses import dataclass
import json
from pathlib import Path
import pandas as pd
from .utils import artifact_writer, environment, verify_artifact, write_json
from ._analysis_io import _config, _inputs, _metadata, _output_available, _read_graph, _read_spots, _read_table, _write_graph, _write_spots


def _neighbor_execution(n_obs, config):
    """Resolve the installed Scanpy dispatch for the completed call, without fit.

    Leave the original neighbors call and its shortcut untouched. Scanpy does
    not save its transformer instance; an empty AnnData with the same spot
    count lets its own dispatcher resolve the class and constructor parameters.
    """
    import anndata as ad
    import scanpy as sc
    from importlib.metadata import version

    record = {"requested_backend": config.neighbor_backend,
              "resolution": "same_scanpy_dispatch_without_fit",
              "scanpy_distribution_version": version("scanpy")}
    try:
        resolver = sc.neighbors.Neighbors(ad.AnnData(np.empty((n_obs, 0))))
        method, transformer, shortcut = resolver._handle_transformer(
            "umap", "sklearn" if config.neighbor_backend == "exact" else None,
            knn=True, kwds={"n_neighbors": config.neighbors, "metric": "euclidean",
                            "metric_params": {}, "random_state": config.seed})
        record.update(status="resolved", method=method, scanpy_shortcut=shortcut,
                      transformer_class=f"{type(transformer).__module__}.{type(transformer).__qualname__}",
                      parameters=transformer.get_params(deep=False))
    except Exception as error:
        # Bookkeeping must not turn a supported analysis into a failed run if a
        # future Scanpy release changes this private, fit-free dispatch API.
        record.update(status="unavailable", reason=f"{type(error).__name__}: {error}")
    return record


def build_spot_graph(embedding, config):
    import anndata as ad
    from sklearn.decomposition import PCA
    import scanpy as sc

    frame = validated_frame(embedding, "spot domain input")
    values = row_l2(frame.to_numpy())
    n, d = values.shape
    if n <= config.pca_components or d < config.pca_components:
        raise ValueError("insufficient spots/dimensions for configured PCA")
    if config.neighbors >= n:
        raise ValueError("neighbors must be smaller than the spot population")
    if np.linalg.matrix_rank(values - values.mean(0)) < config.pca_components:
        raise ValueError("insufficient embedding rank for configured PCA")
    pca = PCA(n_components=config.pca_components, random_state=config.seed,
              svd_solver=config.pca_solver)
    features = pca.fit_transform(values)
    result = ad.AnnData(values)
    result.obs_names = frame.index.copy()
    result.obsm["X_pca"] = features
    neighbor_options = {"transformer": "sklearn"} if config.neighbor_backend == "exact" else {}
    sc.pp.neighbors(
        result,
        n_neighbors=config.neighbors,
        n_pcs=config.pca_components,
        use_rep="X_pca",
        metric="euclidean",
        method="umap",
        random_state=config.seed,
        **neighbor_options,
    )
    result.uns["modular_preprocessing"] = {
        "protocol": (
            "copy1_exact_spatial_v1" if config.neighbor_backend == "exact"
            else "copy1_original_spatial_v1"
        ),
        "neighbor_backend": config.neighbor_backend,
        "neighbor_transformer": "sklearn_brute" if config.neighbor_backend == "exact" else "scanpy_auto",
        "neighbor_execution": _neighbor_execution(n, config),
        "seed": config.seed,
        "pca_components": config.pca_components,
        "pca_solver": pca._fit_svd_solver,
        "neighbors": config.neighbors,
        "metric": "euclidean",
    }
    return result


def exact_k(adata, target, config):
    import scanpy as sc

    if not isinstance(target, int) or target < 1 or target > adata.n_obs:
        raise ValueError("invalid target K for spot population")
    if inspect.signature(sc.tl.louvain).parameters["random_state"].default != 0:
        raise ValueError("Scanpy Louvain default seed differs from copy1 protocol")
    working = adata.copy()
    low = 0.0
    high = float(config.resolution_max)
    trace = []
    state = np.random.get_state()
    np.random.seed(config.seed)
    try:
        for step in range(config.max_search_steps):
            resolution = (low + high) / 2
            # Deliberately omit random_state: copy1 uses the checked Scanpy default 0.
            sc.tl.louvain(working, resolution=resolution, key_added="modular_domain")
            labels = working.obs["modular_domain"].astype(str).to_numpy()
            count = len(np.unique(labels))
            trace.append({"iteration": step + 1, "resolution": resolution, "observed_k": count})
            if count == target:
                return np.array(["D" + x for x in labels]), trace
            if count > target:
                high = resolution
            else:
                low = resolution
    finally:
        np.random.set_state(state)
    raise ValueError(
        f'exact K={target} unavailable in [0,{config.resolution_max}] after {config.max_search_steps} steps; observed {[x["observed_k"] for x in trace]}'
    )


def fixed_resolution(adata, config):
    """Run one native Louvain partition, retaining its actual domain count."""
    import scanpy as sc

    if config.fixed_resolution is None:
        raise ValueError("fixed_resolution must be explicit")
    working = adata.copy()
    state = np.random.get_state()
    np.random.seed(config.seed)
    try:
        sc.tl.louvain(working, resolution=config.fixed_resolution,
                      random_state=config.louvain_seed, directed=True,
                      use_weights=False, key_added="modular_domain")
        labels = np.array(["D" + value for value in working.obs["modular_domain"].astype(str)])
    finally:
        np.random.set_state(state)
    return labels, [{"iteration": 1, "resolution": config.fixed_resolution,
                     "observed_k": len(np.unique(labels))}]


def _partition_record(config, values):
    return {"mode": "fixed_resolution", "partition_id": "fixed_resolution",
            "resolution": config.fixed_resolution, "observed_k": len(np.unique(values)),
            "target_k": None}


@dataclass
class DomainResult:
    labels: pd.DataFrame
    graph: object
    trace: list
    target_k: int | None
    coordinates: np.ndarray | None
    path: Path | None = None

    @property
    def observed_k(self):
        """Actual number of saved domains; target_k is None for fixed resolution."""
        return int(self.labels.domain.nunique())


def identify(inputs, config, output=None):
    """Identify one exact-K or fixed-resolution partition from spot embeddings.

    No existing labels are imported and no label smoothing is applied. To fit
    several prespecified partitions in one artifact use analysis.analyze().
    """
    _output_available(output)
    config = _config(config)
    fixed = config.fixed_resolution is not None
    if not fixed and len(config.target_domains) != 1:
        raise ValueError("standalone identify requires exactly one explicit target K")
    inputs = _inputs(inputs)

    graph = build_spot_graph(inputs.spot, config)
    k = None if fixed else config.target_domains[0]
    values, trace = fixed_resolution(graph, config) if fixed else exact_k(graph, k, config)
    labels = pd.DataFrame({"spot_id": inputs.spot.index,
                           "slice_id": inputs.slice_ids, "domain": values})
    result = DomainResult(labels, graph, trace, k, inputs.coordinates)
    if output is not None:
        metadata = _metadata(inputs, config)
        metadata["target_k"] = k
        if fixed:
            metadata["domain_partitions"] = [_partition_record(config, values)]
        with artifact_writer(output, "spatial_domains", metadata) as stage:
            _write_spots(stage, inputs)
            _write_graph(stage, graph, metadata)
            stem = "fixed_resolution" if fixed else f"K{k}"
            labels.to_csv(stage / f"spatial_domains_{stem}.tsv", sep="\t", index=False)
            write_json(stage / ("fixed_resolution_trace.json" if fixed else f"exact_K{k}_trace.json"), trace)
            metadata["environment"] = environment()
        result.path = Path(output).resolve()
    return result


def _read(path, metadata, k):
    fixed = k is None
    spots, coordinates = _read_spots(path)
    stem = "fixed_resolution" if fixed else f"K{k}"
    labels = _read_table(path / f"spatial_domains_{stem}.tsv", ("spot_id", "slice_id", "domain"))
    if (labels.spot_id.tolist() != spots.spot_id.tolist()
            or labels.slice_id.tolist() != spots.slice_id.tolist()):
        raise ValueError("saved domain IDs/slices do not match ordered spot metadata")
    if not fixed and labels.domain.nunique() != k:
        raise ValueError("saved domain partition does not contain the declared exact K")
    trace = json.loads((path / ("fixed_resolution_trace.json" if fixed else f"exact_K{k}_trace.json")).read_text())
    observed = int(labels.domain.nunique())
    if fixed:
        config = AnalysisConfig(**metadata["config"])
        if (config.fixed_resolution is None
                or metadata.get("domain_partitions") != [_partition_record(config, labels.domain.to_numpy())]
                or trace != [{"iteration": 1, "resolution": config.fixed_resolution, "observed_k": observed}]):
            raise ValueError("saved fixed-resolution partition metadata mismatch")
    elif not isinstance(trace, list) or not trace or trace[-1].get("observed_k") != k:
        raise ValueError("saved exact-K trace does not end at the requested K")
    graph = _read_graph(path, spots.spot_id, metadata)
    if graph is None:
        raise ValueError("saved spatial domain graph is missing")
    return DomainResult(labels, graph, trace, k, coordinates, path)


def load_result(path):
    """Verify and load a standalone spatial_domains artifact without fitting."""
    path = Path(path).resolve()
    manifest = verify_artifact(path, "spatial_domains")
    if manifest["metadata"].get("config", {}).get("fixed_resolution") is not None:
        return _read(path, manifest["metadata"], None)
    k = manifest["metadata"].get("target_k")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("invalid saved target K")
    return _read(path, manifest["metadata"], k)


def _load_from_analysis(path, k=None):
    from .analysis import load_result as load_analysis
    result = load_analysis(path)
    config = result.metadata["config"]
    if config.get("domains", False) and config.get("fixed_resolution") is not None:
        if k is not None:
            raise ValueError("fixed-resolution partition has no target K; load with domains()")
        return _read(result.path, result.metadata, None)
    targets = (result.metadata["config"]["target_domains"]
               if result.metadata["config"].get("domains", False) else [])
    if k is None:
        if len(targets) != 1:
            raise ValueError("choose k explicitly when saved analysis has zero or multiple partitions")
        k = targets[0]
    if not isinstance(k, int) or isinstance(k, bool) or k not in targets:
        raise ValueError(f"K={k} is not a saved partition; available: {targets}")
    return _read(result.path, result.metadata, k)
