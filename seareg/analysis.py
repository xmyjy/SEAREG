"""Downstream analysis orchestration and verified result loading."""

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from threading import RLock
import numpy as np
import pandas as pd
import scipy.sparse as sp
from .config import AnalysisConfig
from .utils import artifact_writer, environment, identity, verify_artifact, write_json
from .utils import protect_output
from ._analysis_io import DISPLAY, _config, _figure_metadata, _inputs, _metadata, _output_available, _read_graph, _read_spots, _read_table, _write_graph, _write_spots


@dataclass
class AnalysisResult:
    path: Path
    metadata: dict
    _reader: object = field(default=None, init=False, repr=False, compare=False)
    _reader_lock: object = field(default_factory=RLock, init=False, repr=False, compare=False)

    def _get_reader(self):
        with self._reader_lock:
            if self._reader is None:
                loaded = load_result(self.path)
                self.path, self.metadata = loaded.path, loaded.metadata
                self._reader = loaded._reader
            if Path(self.path) != self._reader.path:
                raise ValueError("analysis handle path changed; reload with load_result()")
            return self._reader

    def domains(self, k=None):
        """Load a saved partition (fixed resolution or chosen K), never fit."""
        from .domains import _load_from_analysis
        return _load_from_analysis(self, k)

    def spot_embedding(self):
        """Load saved spot UMAP coordinates, never fit UMAP."""
        from .embedding import _load_from_analysis
        return _load_from_analysis(self, "spot")

    def joint_embedding(self):
        """Load saved all-node UMAP coordinates, never fit UMAP."""
        from .embedding import _load_from_analysis
        return _load_from_analysis(self, "joint")


def load_result(path):
    """Verify an immutable analysis handle without fitting or rendering.

    Accessors share verified spot data while checking filesystem identity before
    and after each read. Changed or resealed artifacts require a new load_result
    call. Returned frames, arrays and graphs remain independent mutable copies.
    """
    from ._analysis_io import _AnalysisReader, _artifact_state

    path = Path(path).resolve()
    before = _artifact_state(path)
    manifest = verify_artifact(path, "analysis")
    metadata = json.loads((path / "analysis_manifest.json").read_text())
    if metadata != manifest["metadata"]:
        raise ValueError("analysis manifest metadata mismatch")
    if before != _artifact_state(path):
        raise ValueError("analysis changed while verifying content")
    result = AnalysisResult(path, metadata)
    result._reader = _AnalysisReader(path, metadata, before)
    return result


def spatial_scores(inputs, *, peak_ids=(), motif_ids=(), standardize=False):
    """Calculate named spot–peak and spot–motif cosine profiles.

    ``inputs`` is an EmbeddingSet or completed TrainingRun. Request feature IDs
    from its native peak/motif indices; only the requested profiles are formed.
    The long DataFrame retains spot/slice IDs, feature type/ID, ``cosine`` and
    ``score``, plus native x/y coordinates when available. ``standardize=True``
    makes score a per-feature z-score across all supplied spots (ddof=0), the
    PSS/MSS convention used in Tutorial 1. Constant profiles cannot be
    standardized. Otherwise score is the unmodified cosine association.

    This operation does not train, fit domains/UMAP, smooth, clip or write files.
    """
    from .data import names

    if not isinstance(standardize, bool):
        raise TypeError("standardize must be bool")
    requests = []
    for kind, values in (("peak", peak_ids), ("motif", motif_ids)):
        values = () if values is None else ((values,) if isinstance(values, str) else values)
        requests.append((kind, names(values, f"{kind} feature IDs")))
    if not any(ids for _, ids in requests):
        raise ValueError("request at least one peak or motif feature ID")
    inputs = _inputs(inputs)

    def unit_rows(frame):
        values = frame.to_numpy(dtype=np.float64, copy=True)
        # Scaling before the norm preserves cosine for small/large magnitudes.
        values /= np.max(np.abs(values), axis=1, keepdims=True)
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    spots = unit_rows(inputs.spot)
    tables = []
    for kind, ids in requests:
        if not ids:
            continue
        frame = getattr(inputs, kind)
        if frame is None or (kind == "motif" and inputs.dna_kind != "motif"):
            raise ValueError(f"{kind} embeddings are unavailable for this input")
        missing = [name for name in ids if name not in frame.index]
        if missing:
            raise ValueError(f"unknown {kind} feature IDs: {missing}")
        cosine = spots @ unit_rows(frame.loc[list(ids)]).T
        scores = cosine
        if standardize:
            deviation = cosine.std(axis=0, ddof=0)
            is_constant = (np.ptp(cosine, axis=0) == 0) | (deviation == 0)
            if np.any(is_constant):
                constant = [name for name, value in zip(ids, is_constant) if value]
                raise ValueError(f"cannot standardize constant {kind} profiles: {constant}")
            scores = (cosine - cosine.mean(axis=0)) / deviation
        for column, feature_id in enumerate(ids):
            table = pd.DataFrame({
                "spot_id": inputs.spot.index, "slice_id": inputs.slice_ids,
                "feature_type": kind, "feature_id": feature_id,
                "cosine": cosine[:, column], "score": scores[:, column],
            })
            if inputs.coordinates is not None:
                table["x"] = inputs.coordinates[:, 0]
                table["y"] = inputs.coordinates[:, 1]
            tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    result.attrs.update(metric="cosine", standardize=standardize,
                        population="all supplied spots", ddof=0)
    return result


def _prepare_analysis(inputs, config, output, *, plots=False):
    if not isinstance(plots, bool):
        raise TypeError("plots must be bool")
    _output_available(output)
    config = _config(config)
    inputs = _inputs(inputs)
    if config.domains and inputs.coordinates is None:
        raise ValueError("spatial domain analysis requires ordered spatial coordinates")
    return inputs


def analyze(inputs, config, output, *, plots=True, on_domain_miss="raise"):
    """Fit and save spatial domains and UMAPs from named embeddings.

    ``on_domain_miss="raise"`` preserves strict exact-K failure for Workflow
    and CLI calls. ``"continue"`` records unsuccessful exact-K searches,
    retains successful partitions and computes the requested UMAPs. It does
    not select a different K or change any scientific configuration.
    """
    from .domains import _ExactKUnavailable, build_spot_graph, identify_spatial_domains
    from .embedding import compute_spot_umap, compute_umap

    if not isinstance(on_domain_miss, str) or on_domain_miss not in ("raise", "continue"):
        raise ValueError("on_domain_miss must be 'raise' or 'continue'")
    inputs = _prepare_analysis(inputs, config, output, plots=plots)
    graph = build_spot_graph(inputs.spot, config) if config.domains or config.spot_umap else None
    domains = {}
    domain_misses = {}
    if config.domains:
        targets = (None,) if config.fixed_resolution is not None else config.target_domains
        for k in targets:
            try:
                fit = identify_spatial_domains(inputs.spot, k, _config=config, _graph=graph)
            except _ExactKUnavailable as error:
                if on_domain_miss == "raise":
                    raise
                domain_misses[str(k)] = error.trace
                print(f"Spatial domains: {error}. Continuing with the requested UMAPs.")
            else:
                domains[k] = fit["labels"].raw_domain.to_numpy()
                graph = fit["graph"]

    spot = compute_spot_umap(graph, min_dist=config.min_dist, _config=config) if config.spot_umap else None
    joint = compute_umap(
        inputs, min_dist=config.min_dist, _config=config, _graph=graph
    ) if config.joint else None
    return _save_analysis(inputs, config, output, graph, domains, spot, joint,
                          plots=plots, domain_misses=domain_misses)


def _save_analysis(inputs, config, output, graph, domains, spot, joint, *, plots=False,
                   domain_misses=None):
    """Write the existing analysis artifact from the products of a single fit."""
    from .domains import _partition_record

    if domain_misses:
        requested_targets = config.target_domains
        config = replace(config, target_domains=tuple(domains), domains=bool(domains))
    metadata = _metadata(inputs, config)
    if domain_misses:
        metadata["requested_target_domains"] = requested_targets
        metadata["domain_search_misses"] = domain_misses
    metadata["plots_requested"] = plots
    metadata["plots_exported"] = plots and bool(config.domains or config.spot_umap or config.joint)
    metadata["display"]["rendered"] = metadata["plots_exported"]
    metadata["display"]["rendered_figures"] = {}

    with artifact_writer(output, "analysis", metadata) as stage:
        _write_spots(stage, inputs)
        if graph is not None:
            _write_graph(stage, graph, metadata)
        fixed = config.fixed_resolution is not None
        for k, labels in domains.items():
            trace = graph.uns["_domain_traces"][str(k)]
            stem = "fixed_resolution" if fixed else f"K{k}"
            if fixed:
                metadata["domain_partitions"] = [_partition_record(config, labels)]
            pd.DataFrame({"spot_id": inputs.spot.index, "slice_id": inputs.slice_ids,
                          "domain": labels}).to_csv(stage / f"spatial_domains_{stem}.tsv", sep="\t", index=False)
            write_json(stage / ("fixed_resolution_trace.json" if fixed else f"exact_K{k}_trace.json"), trace)
        if spot is not None:
            spot.to_csv(stage / "spot_umap.tsv", sep="\t", index=False)
        if joint is not None:
            joint.table.to_csv(stage / "joint_umap.tsv", sep="\t", index=False)
            np.save(stage / "joint_input_l2.npy", joint.normalized, allow_pickle=False)
            sp.save_npz(stage / "joint_connectivities.npz", joint.graph)
            metadata["joint_effective_parameters"] = {
                "neighbors": config.neighbors, "min_dist": config.min_dist,
                "metric": "euclidean", "seed": config.seed, "PCA": False,
                "feature_to_spot_alignment": False,
            }
        if plots:
            from .plotting import _render_analysis_products
            metadata["display"]["rendered_figures"] = _render_analysis_products(
                stage, coordinates=inputs.coordinates, slice_ids=inputs.slice_ids,
                domains=domains, fixed=fixed, spot=spot,
                joint=None if joint is None else joint.table,
            )
        metadata["environment"] = environment()
        write_json(stage / "analysis_manifest.json", metadata)
    return load_result(output)
