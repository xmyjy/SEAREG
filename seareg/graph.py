"""Normalized spatial, spot-peak and peak-DNA graphs and public preparation."""

from dataclasses import dataclass,asdict,replace
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from .config import PreprocessConfig
from .data import PreparedNetwork
from .utils import protect_output
from .data import names,canonical_csr,peak_keys,_revcomp
import scipy.sparse as sp
from .artifacts import validated_operation


@validated_operation
def build_network(filtered,selected,config,output,*,_compiled_data=None):
    """Compile final model axes/static graphs once; never read a raw H5AD."""
    from .artifacts import FilteredCounts,ModelNetwork,stage_metadata,save_tensor_graph
    from .config import NetworkConfig
    from .edges import SelectedMotifEdges
    from .utils import artifact_writer,identity,write_json
    import pandas as pd
    if Path(output).exists():raise FileExistsError(output)
    if not isinstance(config,NetworkConfig):raise TypeError('build_network requires NetworkConfig')
    filtered=FilteredCounts.load(filtered)
    selected=(SelectedMotifEdges.load(selected)
              if selected is not None else None)
    if selected is not None and not selected.metadata['nnz']:
        raise ValueError('selected training graph is empty; annotation/selection remain valid')
    if selected is not None and selected.metadata['selection']!='imported_matrix' and selected.peak_ids!=filtered.peak_ids:
        raise ValueError('selected motif peak axis differs from filtered input')
    data=filtered.load_data();pc=PreprocessConfig(**asdict(config))
    if _compiled_data is not None:
        from .data import PreparedData
        if (_compiled_data.fingerprint!=filtered.metadata.get('caller_data_fingerprint') or
                _compiled_data.spot_ids!=data.spot_ids or _compiled_data.peak_ids!=data.peak_ids):
            raise ValueError('compiled prepared input identity differs')
        data=PreparedData(data.X,data.spot_ids,data.peak_ids,data.coordinates,data.slice_ids,
            _compiled_data.dna,_compiled_data.dna_ids,_compiled_data.dna_kind,_compiled_data.provenance)
    data.provenance['config']=asdict(pc)
    if selected is not None:
        if _compiled_data is None:
            data=align_dna(data,selected.load_matrix(),selected.peak_ids,selected.motif_ids,pc)
        else:
            # This is already the model axis, not a graph requiring another
            # binary conversion, empty-column drop or neighborhood dedup.
            data.provenance['dna_alignment']={'original_to_representative':{
                motif:motif for motif in data.dna_ids},'semantics':'frozen_compiled_prepared_input'}
        data.provenance['selected_edges']=selected.fingerprint
    spatial=spatial_graph(data,pc);motif,dp,dk=dna_graph(data)
    upstreams={'filtered':filtered}
    if selected is not None:upstreams['selected_edges']=selected
    output=protect_output(output,[p.path for p in upstreams.values()])
    metadata=stage_metadata(output,upstreams=upstreams,spot_ids=data.spot_ids,peak_ids=data.peak_ids,
        spot_axis_hash=identity(data.spot_ids),peak_axis_hash=identity(data.peak_ids),
        motif_axis_hash=identity(data.dna_ids),motif_ids=data.dna_ids,shape=list(data.X.shape),
        data_fingerprint=data.fingerprint,has_motif=data.dna is not None,
        has_spatial=spatial.adjacency is not None,network_config=asdict(config),
        motif_compilation='selected_edges' if _compiled_data is None else 'frozen_compiled_prepared_input')
    if filtered.metadata.get('caller_data_fingerprint') is not None:
        metadata['caller_data_fingerprint']=filtered.metadata['caller_data_fingerprint']
    with artifact_writer(output,'model_network',metadata) as stage:
        write_json(stage/'data.json',{'spot_ids':data.spot_ids,'peak_ids':data.peak_ids,
            'slice_ids':data.slice_ids,'dna_ids':data.dna_ids,'dna_kind':data.dna_kind,
            'provenance':data.provenance})
        write_json(stage/'graph_report.json',spatial.metadata)
        if data.dna is not None:sp.save_npz(stage/'effective_motif_edges.npz',data.dna)
        if spatial.adjacency is not None:save_tensor_graph(stage/'spatial_adjacency.npz',spatial.adjacency)
        if motif is not None:save_tensor_graph(stage/'motif_adjacency.npz',motif)
        np.savez_compressed(stage/'degrees.npz',spatial_degree=spatial.raw_degree.numpy(),
            peak_degree=np.array([],np.float32) if dp is None else dp.numpy(),
            motif_degree=np.array([],np.float32) if dk is None else dk.numpy())
        if selected is not None:
            report=data.provenance['dna_alignment'];mapping=report['original_to_representative']
            node={name:i for i,name in enumerate(data.dna_ids)}
            pd.DataFrame([{'motif_id':name,'source_index':i,'representative_id':mapping[name] or '',
                'model_node_index':node.get(mapping[name],-1),
                'status':'empty' if mapping[name] is None else 'retained' if mapping[name]==name else 'duplicate'}
                for i,name in enumerate(selected.motif_ids)]).to_csv(stage/'motif_node_map.tsv.gz',sep='\t',index=False)
            coo=data.dna.tocoo();effective={(int(p),int(m)):k for k,(p,m) in enumerate(zip(coo.row,coo.col))}
            lookup={p:i for i,p in enumerate(peak_keys(data.peak_ids))};lineage=[]
            for row in selected.edges().itertuples(index=False):
                p=lookup.get(peak_keys((row.peak_id,))[0],-1);representative=mapping[row.motif_id]
                m=node.get(representative,-1);edge=effective.get((p,m),-1)
                lineage.append({'relation_index':row.relation_index,'peak_id':row.peak_id,'motif_id':row.motif_id,
                    'representative_id':representative or '', 'effective_edge_index':edge,
                    'status':'outside_filtered_axis' if p<0 else 'empty' if m<0 else
                             'retained' if representative==row.motif_id else 'duplicate_to_representative'})
            pd.DataFrame(lineage).to_csv(stage/'motif_edge_lineage.tsv.gz',sep='\t',index=False)
    return ModelNetwork.load(output)


@dataclass
class GraphResult:
    adjacency: torch.Tensor
    raw_degree: torch.Tensor
    metadata: dict


def normalized_adjacency(edges, n_nodes, weights=None):
    raw = torch.as_tensor(edges)
    if raw.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("edge indices must have integer dtype")
    edges = torch.as_tensor(edges, dtype=torch.long)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edges must have shape 2xE")
    if edges.numel() and (edges.min() < 0 or edges.max() >= n_nodes):
        raise ValueError("edge index outside graph")
    values = (
        torch.ones(edges.shape[1], dtype=torch.float32)
        if weights is None
        else torch.as_tensor(weights, dtype=torch.float32)
    )
    if values.shape != (edges.shape[1],) or not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("invalid edge weights")
    both = torch.cat([edges, edges.flip(0)], dim=1)
    a = torch.sparse_coo_tensor(both, torch.cat([values, values]), (n_nodes, n_nodes)).coalesce()
    ei = a.indices()
    w = a.values()
    if not torch.isfinite(w).all():
        raise ValueError("graph coalescing overflow introduced non-finite weights")
    deg = torch.zeros(n_nodes)
    deg.scatter_add_(0, ei[0], w)
    deg.clamp_(min=1e-8)
    denominator = deg[ei[0]] * deg[ei[1]]
    if not torch.isfinite(deg).all() or not torch.isfinite(denominator).all():
        raise ValueError("graph normalization degree overflow; rescale input weights explicitly")
    return torch.sparse_coo_tensor(ei, w / torch.sqrt(denominator), (n_nodes, n_nodes)).coalesce()


def spatial_graph(data, config):
    n = data.X.shape[0]
    if not config.use_spatial:
        return GraphResult(None, torch.zeros(n), {"enabled": False})
    if data.coordinates is None:
        raise ValueError("spatial coordinates are required")
    parts = []
    values = []
    degree = np.zeros(n, dtype=np.float32)
    group_report = []
    labels = np.asarray(data.slice_ids)
    for label in dict.fromkeys(labels.tolist()):
        global_idx = np.flatnonzero(labels == label)
        coords = data.coordinates[global_idx]
        size = len(global_idx)
        if size == 1:
            group_report.append({"slice": label, "spots": 1, "edges": 0})
            continue
        if len(np.unique(coords, axis=0)) != size:
            raise ValueError(f"duplicate/coincident spatial coordinates within slice {label}")
        nn = NearestNeighbors(n_neighbors=2, n_jobs=1).fit(coords)
        nearest = nn.kneighbors(coords)[0][:, 1]
        unit = float(np.median(nearest))
        if config.spatial_method == "knn":
            k = int(config.spatial_neighbors)
            if k >= size:
                raise ValueError(f"spatial k={k} requires more than {size} spots in slice {label}")
            dist, index = (
                NearestNeighbors(n_neighbors=k + 1, n_jobs=1).fit(coords).kneighbors(coords)
            )
            rows = np.repeat(np.arange(size), k)
            cols = index[:, 1:].ravel()
            dist = dist[:, 1:].ravel()
        else:
            radius = float(nearest.min() * config.spatial_neighbors)
            graph = (
                NearestNeighbors(radius=radius, n_jobs=1)
                .fit(coords)
                .radius_neighbors_graph(coords, mode="distance")
                .tocoo()
            )
            mask = graph.row != graph.col
            rows = graph.row[mask]
            cols = graph.col[mask]
            dist = graph.data[mask]
        if config.spatial_distance == "const":
            weight = np.ones(len(rows), dtype=np.float32)
        else:
            d = dist / (unit + 1e-12)
            weight = (
                1 / np.maximum(d, 1e-12) ** (2 if config.spatial_distance == "recip2" else 1)
            ).astype(np.float32)
        parts.append(np.array([global_idx[rows], global_idx[cols]]))
        values.append(weight)
        np.add.at(degree, global_idx[rows], weight)
        group_report.append({"slice": label, "spots": size, "directed_edges": len(rows)})
    edges = np.concatenate(parts, axis=1) if parts else np.empty((2, 0), dtype=np.int64)
    weight = np.concatenate(values) if values else np.empty(0, dtype=np.float32)
    if not edges.size:
        raise ValueError("enabled spatial graph has no edges")
    if not np.all(labels[edges[0]] == labels[edges[1]]):
        raise RuntimeError("cross-slice spatial edge")
    return GraphResult(
        normalized_adjacency(edges, n, weight),
        torch.from_numpy(degree),
        {
            "enabled": True,
            "cross_slice_edges": 0,
            "slices": group_report,
            "symmetrization": "sum",
            "radius_scale": "minimum_nearest_distance",
        },
    )


def bipartite_graph(local_edges, n_left, n_right):
    raw = np.asarray(local_edges)
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("local edge indices must have integer dtype")
    if raw.ndim != 2 or raw.shape[0] != 2:
        raise ValueError("local edges must have shape 2xE")
    if raw.size and (raw.min() < 0 or raw[0].max() >= n_left or raw[1].max() >= n_right):
        raise ValueError("local edge index outside bipartite universe")
    edges = np.asarray(local_edges, dtype=np.int64).copy()
    edges[1] += n_left
    return normalized_adjacency(edges, n_left + n_right)


def dna_graph(data):
    if data.dna is None:
        return None, None, None
    coo = data.dna.tocoo()
    edges = np.array([coo.row, coo.col + data.X.shape[1]])
    adjacency = normalized_adjacency(edges, data.dna.shape[0] + data.dna.shape[1], coo.data)
    return (
        adjacency,
        torch.from_numpy(np.asarray(data.dna.sum(axis=1)).ravel()).float(),
        torch.from_numpy(np.asarray(data.dna.sum(axis=0)).ravel()).float(),
    )


def align_dna(data, matrix, source_peak_ids, source_dna_ids, config):
    source_peaks = peak_keys(source_peak_ids)
    target = peak_keys(data.peak_ids)
    ids = names(source_dna_ids, "DNA")
    x = canonical_csr(matrix, binary=config.dna_weight == "binary")
    if x.shape != (len(source_peaks), len(ids)):
        raise ValueError("DNA source names/shape mismatch")
    lookup = {p: i for i, p in enumerate(source_peaks)}
    missing = [p for p in target if p not in lookup]
    matches = len(target) - len(missing)
    if matches == 0:
        raise ValueError("DNA has zero matching peak overlap")
    if missing and not config.allow_partial_dna:
        raise ValueError(f"DNA coverage missing {len(missing)} peaks: {missing[:5]}")
    row_map = np.array([lookup.get(p, -1) for p in target])
    coo = x.tocoo()
    destination = {lookup[p]: j for j, p in enumerate(target) if p in lookup}
    outrows = np.array([destination.get(int(i), -1) for i in coo.row])
    keep = outrows >= 0
    aligned = sp.csr_matrix(
        (coo.data[keep], (outrows[keep], coo.col[keep])), shape=(len(target), len(ids))
    )
    column_members = {}
    raw_kmer_groups = {}
    if config.dna_kind == "kmer":
        groups = {}
        for i, k in enumerate(ids):
            representative = min(k.upper(), _revcomp(k))
            groups.setdefault(representative, []).append(i)
            raw_kmer_groups[k] = representative
        columns = []
        for rep, index in groups.items():
            v = np.asarray((aligned[:, index] > 0).sum(axis=1)).ravel() > 0
            columns.append(sp.csr_matrix(v[:, None], dtype=np.float32))
            column_members[rep] = [ids[i] for i in index]
        aligned = sp.hstack(columns, format="csr")
        ids = tuple(groups)
    csc = aligned.tocsc()
    retained = []
    new_ids = []
    seen = {}
    mapping = {}
    for j, name in enumerate(ids):
        begin, end = csc.indptr[j : j + 2]
        if begin == end:
            mapping[name] = None
            continue
        signature = (csc.indices[begin:end].tobytes(), csc.data[begin:end].tobytes())
        if config.dna_kind == "motif" and config.motif_deduplicate and signature in seen:
            representative = seen[signature]
            mapping[name] = representative
            column_members[representative].append(name)
        else:
            retained.append(j)
            new_ids.append(name)
            seen[signature] = name
            mapping[name] = name
            column_members.setdefault(name, [name])
    aligned = csc[:, retained].tocsr()
    if config.dna_kind == "kmer":
        mapping = {raw: mapping[representative] for raw, representative in raw_kmer_groups.items()}
    if aligned.nnz == 0:
        raise ValueError("DNA alignment produced an empty graph")
    report = {
        "matched_peaks": matches,
        "target_peaks": len(target),
        "source_peaks": len(source_peaks),
        "missing_peak_ids": missing,
        "coverage": matches / len(target),
        "source_nnz": x.nnz,
        "aligned_nnz": aligned.nnz,
        "deduplicate": config.motif_deduplicate,
        "original_to_representative": mapping,
        "representative_to_members": column_members,
        "weight_mode": config.dna_weight,
        "reverse_complement": "binary_OR" if config.dna_kind == "kmer" else None,
    }
    provenance = {**data.provenance, "dna_alignment": report}
    return replace(
        data, dna=aligned, dna_ids=tuple(new_ids), dna_kind=config.dna_kind, provenance=provenance
    )


def prepare_network(inputs, config, output):
    from .compat import prepare_network_legacy
    return prepare_network_legacy(inputs,config,output)


def load_network(path):
    return PreparedNetwork.load(path)


def inspect_network(path):
    network = PreparedNetwork.load(path)
    data = network.load_data()
    return {"path": str(network.path), "fingerprint": network.fingerprint,
            "spots": len(data.spot_ids), "peaks": len(data.peak_ids),
            "dna_nodes": len(data.dna_ids), "positive_edges": data.X.nnz,
            "dna_edges": 0 if data.dna is None else data.dna.nnz}
