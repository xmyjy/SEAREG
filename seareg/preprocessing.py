"""Input inspection and the unique raw-count filtering implementation."""

from dataclasses import asdict
import numpy as np
import pandas as pd
from .data import PreparedData, canonical_csr, names, InputArtifact, peak_keys, _revcomp
import scipy.sparse as sp
from pathlib import Path
import os
from .utils import artifact_writer, file_sha256, write_json, protect_output, identity
from .compat import _read_dna

def filter_atac(adata, config, *, return_frequency=False):
    spots = names(adata.obs_names, "spots")
    peaks = names(adata.var_names, "peaks")
    x = canonical_csr(adata.X, binary=True)
    n = x.shape[0]
    if n < 1:
        raise ValueError("no spots")
    ratio = 0.01 if n <= 2500 else 0.005
    low = max(1, int(n * ratio))
    high = int(n * (1 - ratio))
    frequency = np.asarray(x.getnnz(axis=0)).ravel()
    keep = np.array([p.startswith("chr") for p in peaks]) & (frequency >= low) & (frequency <= high)
    selected = tuple(p for p, k in zip(peaks, keep) if k)
    peak_keys(selected)
    if config.slice_key is not None:
        if config.slice_key not in adata.obs:
            raise ValueError(f"missing slice column {config.slice_key}")
        labels = tuple(adata.obs[config.slice_key].astype(str))
    else:
        keys = [k for k in ("slice_id", "sample_id") if k in adata.obs]
        if len(keys) == 2:
            a = pd.factorize(adata.obs[keys[0]], sort=False)[0]
            b = pd.factorize(adata.obs[keys[1]], sort=False)[0]
            if not np.array_equal(a, b):
                raise ValueError("conflicting slice_id and sample_id partitions")
        labels = tuple(adata.obs[keys[0]].astype(str)) if keys else ("sample",) * n
    coords = adata.obsm.get("spatial")
    if config.use_spatial and coords is None:
        raise ValueError("spatial enabled but coordinates are missing")
    report = {
        "protocol": config.filter_protocol,
        "original_shape": list(x.shape),
        "filtered_shape": [n, len(selected)],
        "min_cells": low,
        "max_cells": high,
        "nonchromosome_peaks": sum(not p.startswith("chr") for p in peaks),
        "dropped_peak_ids": [p for p, k in zip(peaks, keep) if not k],
        "empty_spot_ids": [spots[i] for i in np.flatnonzero(x[:, keep].getnnz(axis=1) == 0)],
    }
    data = PreparedData(
        x[:, keep],
        spots,
        selected,
        coords,
        labels,
        provenance={"config": asdict(config), "filter": report},
    )
    return (data,frequency) if return_frequency else data




def filter_counts(atac, config, output):
    """Decode/filter a raw count matrix once and preserve its original counts."""
    import anndata as ad
    from .artifacts import FilteredCounts,stage_metadata
    from .config import FilterConfig,PreprocessConfig
    if not isinstance(config,FilterConfig):raise TypeError('filter_counts requires FilterConfig')
    source=Path(atac).resolve()
    if Path(output).exists():raise FileExistsError(output)
    output=protect_output(output,[source])
    before=file_sha256(source);a=ad.read_h5ad(source)
    if file_sha256(source)!=before:raise ValueError('counts changed while reading input')
    data,frequency=filter_atac(a,PreprocessConfig(use_spatial=False,slice_key=config.slice_key,
                                     filter_protocol=config.filter_protocol),return_frequency=True)
    positions={name:i for i,name in enumerate(a.var_names)}
    retained=a[:,[positions[name] for name in data.peak_ids]].copy()
    retained.X=sp.csr_matrix(retained.X)
    retained.X.sum_duplicates();retained.X.eliminate_zeros();retained.X.sort_indices()
    final={name:i for i,name in enumerate(data.peak_ids)};report=data.provenance['filter'];rows=[]
    for index,name in enumerate(a.var_names):
        reason=('non_chromosome' if not name.startswith('chr') else
                'below_min_frequency' if frequency[index]<report['min_cells'] else
                'above_max_frequency' if frequency[index]>report['max_cells'] else 'retained')
        try:interval=peak_keys((name,))[0]
        except ValueError:interval=''
        rows.append({'raw_index':index,'peak_id':name,'canonical_interval':interval,
            'frequency':int(frequency[index]),'retained':name in final,'reason':reason,'final_index':final.get(name,-1)})
    metadata=stage_metadata(output,spot_ids=data.spot_ids,peak_ids=data.peak_ids,
        spot_axis_hash=identity(data.spot_ids),peak_axis_hash=identity(data.peak_ids),slice_ids=data.slice_ids,
        shape=list(data.X.shape),nnz=int(data.X.nnz),filter_report=report,filter_config=asdict(config),
        source={'path':str(source),'sha256':before})
    with artifact_writer(output,'filtered_counts',metadata) as stage:
        retained.write_h5ad(stage/'counts.h5ad')
        pd.DataFrame(rows).to_csv(stage/'peak_filter.tsv.gz',sep='\t',index=False,compression='gzip')
        write_json(stage/'filter_report.json',report)
        if file_sha256(source)!=before:raise ValueError('counts changed during filtering')
    return FilteredCounts.load(output)


def align_dna(data,matrix,source_peak_ids,source_dna_ids,config):
    """Compatibility import for the single network-axis compilation routine."""
    from .graph import align_dna as compile_dna
    return compile_dna(data,matrix,source_peak_ids,source_dna_ids,config)




def prepare_h5ad(path,output,config,motif_path=None):
    """Compatibility adapter to the single filtered/network preparation path."""
    from .compat import network_from_paths
    return network_from_paths(path,motif_path,config,output).path


def load_prepared(path):
    from .artifacts import ModelNetwork
    return ModelNetwork.load(path).load_data()


def inspect_inputs(atac, dna=None, *, use_spatial=True):
    """Read HDF5 headers/identities, not the full count or motif matrices."""
    import h5py
    import numpy as np

    def ids(group):
        key = group.attrs.get("_index", "_index")
        if isinstance(key, bytes):
            key = key.decode()
        if key not in group:
            raise ValueError("H5AD axis has no explicit index")
        values = [v.decode() if isinstance(v, bytes) else str(v) for v in group[key][:]]
        if len(set(values)) != len(values) or any(not v or v in {"nan", "None", "<NA>"} for v in values):
            raise ValueError("H5AD axis IDs must be unique and nonempty")
        return values

    atac = Path(atac).resolve()
    if not atac.is_file():
        raise FileNotFoundError(atac)
    with h5py.File(atac, "r") as h:
        if not all(k in h for k in ["X", "obs", "var"]):
            raise ValueError("ATAC H5AD requires X/obs/var")
        spots, peaks = ids(h["obs"]), ids(h["var"])
        x = h["X"]
        shape = list(map(int, x.shape if isinstance(x, h5py.Dataset) else x.attrs["shape"]))
        dtype = x.dtype if isinstance(x, h5py.Dataset) else x["data"].dtype
        if shape != [len(spots), len(peaks)] or min(shape) < 1 or dtype.kind not in "biuf":
            raise ValueError("ATAC header shape/dtype mismatch")
        spatial = h.get("obsm/spatial")
        if use_spatial and spatial is None:
            raise ValueError("spatial coordinates required")
        if spatial is not None:
            coordinates = spatial[:]
            if coordinates.ndim != 2 or coordinates.shape[0] != len(spots) or coordinates.shape[1] < 2 or not np.isfinite(coordinates).all():
                raise ValueError("spatial coordinate shape/values mismatch")
        result = {"atac": {"path": str(atac), "sha256": file_sha256(atac),
                  "bytes": atac.stat().st_size, "shape": shape, "dtype": str(dtype),
                  "nnz_upper_bound":int(np.prod(shape)) if isinstance(x,h5py.Dataset) else len(x['data'])},
                  "validation_scope": "headers, axis IDs and spatial coordinates; count values validated during prepare-network"}
    if dna is not None:
        dna = Path(dna).resolve()
        if not dna.is_file():
            raise FileNotFoundError(dna)
        with h5py.File(dna, "r") as h:
            if dna.suffix == ".h5ad":
                rows, cols = ids(h["obs"]), ids(h["var"])
                x = h["X"]
                shape = list(map(int, x.shape if isinstance(x, h5py.Dataset) else x.attrs["shape"]))
            else:
                if not all(k in h for k in ["mat", "row_names", "col_names"]):
                    raise ValueError("DNA HDF5 requires mat/row_names/col_names")
                node = h["mat"]
                if isinstance(node, h5py.Group):
                    if node.attrs.get("encoding") != "csr":
                        raise ValueError("unsupported DNA sparse matrix encoding")
                    shape = list(map(int, node.attrs["shape"]))
                else:
                    shape = list(map(int, node.shape))
                rows, cols = h["row_names"], h["col_names"]
            if shape != [len(rows), len(cols)] or min(shape) < 1:
                raise ValueError("DNA header shape mismatch")
        result["dna"] = {"path": str(dna), "sha256": file_sha256(dna),
                         "bytes": dna.stat().st_size, "shape": shape}
    return result


def prepare_inputs(atac, output, dna=None, *, use_spatial=True):
    output = protect_output(output)
    inspected = inspect_inputs(atac, dna, use_spatial=use_spatial)
    files = {key: {**value, "path": os.path.relpath(value["path"], output)}
             for key, value in inspected.items() if key in ("atac", "dna")}
    metadata = {"schema_version": 1, "files": files,
                "validation_scope": inspected["validation_scope"], "counts_transformed": False}
    with artifact_writer(output, "input_identity", {"counts_transformed": False}) as stage:
        write_json(stage / "inputs.json", metadata)
        for key, value in files.items():
            if file_sha256(inspected[key]["path"]) != value["sha256"]:
                raise ValueError("input changed during identity publication")
    return InputArtifact.load(output)


load_inputs = InputArtifact.load
