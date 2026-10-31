"""Immutable filtering and graph stages for the original multislice protocol."""
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import scipy.sparse as sp

from ..artifacts import StageArtifact, stage_metadata, validated_operation
from ..data import PreparedData
from ..utils import artifact_writer, protect_output, write_json
from .inputs import MultiSliceInputs, _filter_multislice_data, preparation_config, sampling_config


def _check_config(artifact,config):
    if (dict(artifact.metadata['preparation_config'])!=preparation_config(config) or
            dict(artifact.metadata['sampling_config'])!=sampling_config(config)):
        raise ValueError('stage and multislice config differ')


def _metadata(output,data,config,*,upstreams=None):
    return stage_metadata(output,upstreams=upstreams,shape=list(data.X.shape),
        spot_ids=data.spot_ids,peak_ids=data.peak_ids,slice_ids=data.slice_ids,
        data_fingerprint=data.fingerprint,preparation_config=preparation_config(config),
        sampling_config=sampling_config(config),positive_edges=int(data.X.nnz),
        protocol='slice_union_prevalence_global_negatives_v1')


def _write_axes(stage,data):
    write_json(stage/'data.json',{'spot_ids':data.spot_ids,'peak_ids':data.peak_ids,
        'slice_ids':data.slice_ids,'dna_ids':[],'dna_kind':None})


def _graph_report(data,spatial_edges):
    """Count filtered positives and directed coalesced spatial graph entries."""
    labels=np.asarray(data.slice_ids)
    source,target=spatial_edges
    if not np.array_equal(labels[source],labels[target]):
        raise ValueError('multislice spatial graph contains cross-section edges')
    return [{'sample_id':name,'spots':int(np.count_nonzero(labels==name)),
             'positive_edges':int(data.X[labels==name].nnz),
             'spatial_edges':int(np.count_nonzero(labels[source]==name))}
            for name in sorted(set(data.slice_ids))]


@dataclass(frozen=True)
class MultiSliceFilteredCounts(StageArtifact):
    KIND: ClassVar[str]='multislice_filtered_counts'

    @validated_operation
    def load_data(self):
        self.verify()
        data=PreparedData(sp.load_npz(self.path/'counts.npz'),tuple(self.metadata['spot_ids']),
            tuple(self.metadata['peak_ids']),np.load(self.path/'coordinates.npy',allow_pickle=False),
            tuple(self.metadata['slice_ids']))
        if data.fingerprint!=self.metadata['data_fingerprint']:
            raise ValueError('filtered multislice data fingerprint mismatch')
        return data


@validated_operation
def filter_multislice_counts(counts,config,output):
    """Apply the original slice-union filter without constructing any graphs."""
    if Path(output).exists():raise FileExistsError(output)
    output=protect_output(output,[counts] if isinstance(counts,(str,Path)) else [])
    data=_filter_multislice_data(counts,config)
    with artifact_writer(output,MultiSliceFilteredCounts.KIND,_metadata(output,data,config)) as stage:
        sp.save_npz(stage/'counts.npz',data.X)
        np.save(stage/'coordinates.npy',data.coordinates,allow_pickle=False)
        _write_axes(stage,data)
    return MultiSliceFilteredCounts.load(output)


@dataclass(frozen=True)
class MultiSliceNetwork(StageArtifact):
    KIND: ClassVar[str]='multislice_network'

    @validated_operation
    def load_data(self):
        data=self.upstream('filtered',MultiSliceFilteredCounts).load_data()
        if data.fingerprint!=self.metadata['data_fingerprint']:
            raise ValueError('multislice network data fingerprint mismatch')
        return data

    @validated_operation
    def load_graphs(self):
        """Load exact coalesced graph values saved during network construction."""
        from ..artifacts import load_tensor_graph
        import torch
        self.verify()
        with np.load(self.path/'degrees.npz',allow_pickle=False) as degrees:
            return {'spot_peak':load_tensor_graph(self.path/'spot_peak_adjacency.npz'),
                'spatial':load_tensor_graph(self.path/'spatial_adjacency.npz'),
                'spot_peak_degree':torch.from_numpy(degrees['spot_peak_degree'].copy()),
                'spatial_degree':torch.from_numpy(degrees['spatial_degree'].copy())}


@validated_operation
def build_multislice_network(filtered,config,output):
    """Compile full spot-peak and strictly within-section spatial graphs once."""
    if Path(output).exists():raise FileExistsError(output)
    from ..artifacts import save_tensor_graph
    from .sampling import normalized_adjacency, spatial_graph
    import torch
    filtered=MultiSliceFilteredCounts.load(filtered)
    _check_config(filtered,config)
    output=protect_output(output,[filtered.path])
    data=filtered.load_data();num_spot,num_peak=data.X.shape
    spatial,degree=spatial_graph(data.coordinates,np.asarray(data.slice_ids),config.spatial_radius_multiplier)
    edges=torch.from_numpy(np.stack(data.X.nonzero())).long();edges[1]+=num_spot
    full=normalized_adjacency(edges,num_spot+num_peak)
    peak_degree=torch.from_numpy(np.asarray(data.X.getnnz(axis=1)).ravel()).float()
    metadata=_metadata(output,data,config,upstreams={'filtered':filtered})
    metadata['graph_report']=_graph_report(data,spatial.indices().numpy())
    with artifact_writer(output,MultiSliceNetwork.KIND,metadata) as stage:
        _write_axes(stage,data)
        write_json(stage/'graph_report.json',metadata['graph_report'])
        save_tensor_graph(stage/'spot_peak_adjacency.npz',full)
        save_tensor_graph(stage/'spatial_adjacency.npz',spatial)
        np.savez_compressed(stage/'degrees.npz',spatial_degree=degree.numpy(),spot_peak_degree=peak_degree.numpy())
    return MultiSliceNetwork.load(output)


@validated_operation
def _prepare_network_training(network,config,output):
    if Path(output).exists():raise FileExistsError(output)
    network=MultiSliceNetwork.load(network)
    _check_config(network,config)
    output=protect_output(output,[network.path])
    data=network.load_data()
    metadata=_metadata(output,data,config,upstreams={'network':network})
    with artifact_writer(output,MultiSliceInputs.KIND,metadata) as stage:
        _write_axes(stage,data)
    return MultiSliceInputs.load(output)
