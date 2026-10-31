"""Downstream analysis orchestration and verified result loading."""

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
from ._analysis_io import DISPLAY, _config, _figure_metadata, _inputs, _metadata, _output_available, _read_graph, _read_spots, _read_table, _write_graph, _write_spots


@dataclass
class AnalysisResult:
    path: Path
    metadata: dict

    def domains(self, k=None):
        """Load a saved partition (fixed resolution or chosen K), never fit."""
        from .domains import _load_from_analysis
        return _load_from_analysis(self.path, k)

    def spot_embedding(self):
        """Load saved spot UMAP coordinates, never fit UMAP."""
        from .embedding import _load_from_analysis
        return _load_from_analysis(self.path, "spot")

    def joint_embedding(self):
        """Load saved all-node UMAP coordinates, never fit UMAP."""
        from .embedding import _load_from_analysis
        return _load_from_analysis(self.path, "joint")


def load_result(path):
    """Verify a new or historical analysis without fitting or rendering."""
    path = Path(path).resolve()
    manifest = verify_artifact(path, "analysis")
    metadata = json.loads((path / "analysis_manifest.json").read_text())
    if metadata != manifest["metadata"]:
        raise ValueError("analysis manifest metadata mismatch")
    return AnalysisResult(path, metadata)


def spatial_scores(inputs, *, peak_ids=(), motif_ids=(), standardize=False):
    """Calculate named spot–peak and spot–motif cosine profiles.

    ``inputs`` is an EmbeddingSet or completed TrainingRun. Request feature IDs
    from its native peak/motif indices; only the requested profiles are formed.
    The long DataFrame retains spot/slice IDs, feature type/ID, ``cosine`` and
    ``score``, plus native x/y coordinates when available. ``standardize=True``
    makes score a per-feature z-score across all supplied spots (ddof=0), the
    PSAS/MSAS convention used in Tutorial 1. Constant profiles cannot be
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


def analyze(inputs, config, output, *, plots=True):
    """Fit requested downstream methods and publish one immutable artifact.

    Plot exports are included by default in this explicit analysis operation.
    Set plots=False to publish numerical tables only. Standalone plotting remains
    interactive and never writes unless save_figure is explicitly called.
    """
    if not isinstance(plots, bool):
        raise TypeError("plots must be bool")
    _output_available(output)
    config = _config(config)
    inputs = _inputs(inputs)
    if config.domains and inputs.coordinates is None:
        raise ValueError("spatial domain analysis requires ordered spatial coordinates")
    metadata = _metadata(inputs, config)
    metadata["plots_requested"] = plots
    metadata["plots_exported"] = plots and bool(config.domains or config.spot_umap or config.joint)
    metadata["display"]["rendered"] = metadata["plots_exported"]
    metadata["display"]["rendered_figures"] = {}
    from .domains import build_spot_graph, exact_k, fixed_resolution, _partition_record
    from .embedding import spot_umap, joint_umap

    with artifact_writer(output, "analysis", metadata) as stage:
        _write_spots(stage, inputs)
        graph = build_spot_graph(inputs.spot, config) if config.domains or config.spot_umap else None
        if graph is not None:
            _write_graph(stage, graph, metadata)
        first_labels = None
        fixed = config.fixed_resolution is not None
        partitions = (None,) if fixed else config.target_domains
        for k in partitions if config.domains else ():
            labels, trace = fixed_resolution(graph, config) if fixed else exact_k(graph, k, config)
            if first_labels is None:
                first_labels = labels
            stem = "fixed_resolution" if fixed else f"K{k}"
            if fixed:
                metadata["domain_partitions"] = [_partition_record(config, labels)]
            pd.DataFrame({"spot_id": inputs.spot.index, "slice_id": inputs.slice_ids,
                          "domain": labels}).to_csv(stage / f"spatial_domains_{stem}.tsv", sep="\t", index=False)
            write_json(stage / ("fixed_resolution_trace.json" if fixed else f"exact_K{k}_trace.json"), trace)
            if plots:
                from .plotting import spatial_domains, spatial_panel, save_figure
                slices = np.asarray(inputs.slice_ids)
                unique = list(dict.fromkeys(slices.tolist()))
                if len(unique) == 1:
                    fig, _ = spatial_domains(inputs.coordinates, labels, title=unique[0])
                else:
                    cases = [{"coordinates": inputs.coordinates[slices == s],
                              "labels": labels[slices == s], "title": s} for s in unique]
                    fig, _ = spatial_panel(cases, figsize=(3.3 * len(unique), 3.35))
                save_figure(fig, stage / f"spatial_domains_{stem}")
                metadata["display"]["rendered_figures"][f"spatial_domains_{stem}"] = _figure_metadata(fig)
                from matplotlib import pyplot as plt
                plt.close(fig)
        if config.spot_umap:
            coordinates = spot_umap(graph, config)
            table = pd.DataFrame({"spot_id": inputs.spot.index,
                                  "UMAP1": coordinates[:, 0], "UMAP2": coordinates[:, 1]})
            table.to_csv(stage / "spot_umap.tsv", sep="\t", index=False)
            if plots:
                from .plotting import spot_umap as plot_spots, save_figure
                fig, _ = plot_spots(table, labels=first_labels if first_labels is not None else inputs.slice_ids)
                save_figure(fig, stage / "spot_umap", dpi=300, bbox_inches=None)
                metadata["display"]["rendered_figures"]["spot_umap"] = _figure_metadata(fig, export_dpi=300)
                from matplotlib import pyplot as plt
                plt.close(fig)
        if config.joint:
            table, normalized, model = joint_umap(inputs, config)
            table.to_csv(stage / "joint_umap.tsv", sep="\t", index=False)
            np.save(stage / "joint_input_l2.npy", normalized, allow_pickle=False)
            sp.save_npz(stage / "joint_connectivities.npz", model.graph_.tocsr())
            metadata["joint_effective_parameters"] = {
                "neighbors": model.n_neighbors, "min_dist": model.min_dist,
                "metric": model.metric, "seed": config.seed, "PCA": False,
                "feature_to_spot_alignment": False,
            }
            if plots:
                from .plotting import joint_umap as plot_joint, save_figure
                fig, _ = plot_joint(table, labels=first_labels)
                save_figure(fig, stage / "joint_umap", dpi=300, bbox_inches=None)
                metadata["display"]["rendered_figures"]["joint_umap"] = _figure_metadata(fig, export_dpi=300)
                from matplotlib import pyplot as plt
                plt.close(fig)
        metadata["environment"] = environment()
        write_json(stage / "analysis_manifest.json", metadata)
    return load_result(output)
