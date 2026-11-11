"""Exact-K or fixed-resolution spatial domains and fit-free saved partitions."""

import numpy as np
from .config import AnalysisConfig
from .data import row_l2, validated_frame
from dataclasses import dataclass
import json
from pathlib import Path
import pandas as pd
from .utils import verify_artifact
from ._analysis_io import _read_graph, _read_spots, _read_table


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


class _ExactKUnavailable(ValueError):
    def __init__(self, target, config, trace):
        super().__init__(
            f'exact K={target} unavailable in [0,{config.resolution_max}] after '
            f'{config.max_search_steps} steps; observed {[step["observed_k"] for step in trace]}'
        )
        self.trace = trace


def exact_k(adata, target, config):
    import scanpy as sc

    if not isinstance(target, int) or target < 1 or target > adata.n_obs:
        raise ValueError("invalid target K for spot population")
    working = adata.copy()
    low = 0.0
    high = float(config.resolution_max)
    trace = []
    state = np.random.get_state()
    np.random.seed(config.seed)
    try:
        for step in range(config.max_search_steps):
            resolution = (low + high) / 2
            # Deliberately use Scanpy's default random_state.
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
    raise _ExactKUnavailable(target, config, trace)


def identify_spatial_domains(spot_embedding, n_domains, *, domain_map=None, palette=None,
                             _config=None, _graph=None):
    """Identify spatial domains directly from a spot-embedding DataFrame.

    Use the GitHub row-L2, PCA20, 15-neighbor graph and exact-K Louvain
    workflow. Louvain uses Scanpy's default random settings. ``domain_map``
    optionally maps raw D-prefixed labels to display labels; ``palette`` maps
    the final display labels to colors. Without a palette, colors are None.

    Return a dictionary with ``labels``, ``graph`` and ``resolution``.
    The label table preserves spot order and contains ``spot_id``,
    ``raw_domain``, ``domain`` and ``color``.
    """
    config = _config if _config is not None else AnalysisConfig(
        target_domains=(n_domains,),
        pca_components=20,
        neighbors=15,
        seed=1234,
        pca_solver="auto",
        neighbor_backend="auto",
        resolution_max=3.0,
        max_search_steps=20,
    )
    graph = _graph if _graph is not None else build_spot_graph(spot_embedding, config)
    if config.fixed_resolution is None:
        raw_domains, trace = exact_k(graph, n_domains, config)
    else:
        raw_domains, trace = fixed_resolution(graph, config)
    if _config is not None:
        # Keep the native search record for the existing analysis artifact.
        graph.uns.setdefault("_domain_traces", {})[str(n_domains)] = trace
    labels = pd.DataFrame({"spot_id": graph.obs_names, "raw_domain": raw_domains})
    labels["domain"] = labels.raw_domain if domain_map is None else labels.raw_domain.map(domain_map)
    labels["color"] = None if palette is None else labels.domain.map(palette)
    return {"labels": labels, "graph": graph, "resolution": trace[-1]["resolution"]}


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


def _read(path, metadata, k, *, _reader=None):
    fixed = k is None
    spots, coordinates = _read_spots(path) if _reader is None else _reader.read_spots()
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
    graph = _read_graph(path, spots.spot_id, metadata) if _reader is None else _reader.read_graph()
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
    from .analysis import AnalysisResult, load_result as load_analysis
    result = path if isinstance(path, AnalysisResult) else load_analysis(path)
    with result._get_reader().operation() as reader:
        config = reader.metadata["config"]
        if config.get("domains", False) and config.get("fixed_resolution") is not None:
            if k is not None:
                raise ValueError("fixed-resolution partition has no target K; load with domains()")
            return _read(reader.path, reader.metadata, None, _reader=reader)
        targets = config["target_domains"] if config.get("domains", False) else []
        if k is None:
            if len(targets) != 1:
                raise ValueError("choose k explicitly when saved analysis has zero or multiple partitions")
            k = targets[0]
        if not isinstance(k, int) or isinstance(k, bool) or k not in targets:
            raise ValueError(f"K={k} is not a saved partition; available: {targets}")
        return _read(reader.path, reader.metadata, k, _reader=reader)
