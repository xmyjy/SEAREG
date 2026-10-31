"""Independent immutable inputs with the selected slice-aware peak filter."""
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
import numpy as np
import scipy.sparse as sp
from ..artifacts import StageArtifact, stage_metadata, validated_operation
from ..data import PreparedData, canonical_csr, names, peak_keys
from ..utils import artifact_writer, write_json, protect_output, identity


def preparation_config(config):
    return {k:getattr(config,k) for k in ('slice_key','feature_min_fraction','feature_min_spots','feature_max_fraction')}


def sampling_config(config):
    return {'seed':config.seed,'pn_r':config.pn_r,'slice_shards':dict(config.slice_shards),
            'reshard_every':config.reshard_every,'negative_refresh':config.negative_refresh,
            'spatial_radius_multiplier':config.spatial_radius_multiplier}


@dataclass(frozen=True)
class MultiSliceInputs(StageArtifact):
    KIND: ClassVar[str]='multislice_training_inputs'

    def network(self):
        from .stages import MultiSliceNetwork
        return self.upstream('network',MultiSliceNetwork)

    @validated_operation
    def load_data(self):
        self.verify()
        if 'network' in self.metadata.get('upstreams',{}):
            return self.network().load_data()
        return PreparedData(sp.load_npz(self.path/'counts.npz'),tuple(self.metadata['spot_ids']),
                            tuple(self.metadata['peak_ids']),np.load(self.path/'coordinates.npy',allow_pickle=False),
                            tuple(self.metadata['slice_ids']))


def _filter_multislice_data(counts,config):
    """The unchanged slice-union prevalence/global-maximum preparation rule."""
    import anndata as ad
    a=ad.read_h5ad(counts) if isinstance(counts,(str,Path)) else counts
    if not isinstance(a,ad.AnnData):raise TypeError('counts must be AnnData or an h5ad path')
    spots=names(a.obs_names,'spots');peaks=names(a.var_names,'peaks')
    if config.slice_key not in a.obs:raise ValueError(f'missing slice key {config.slice_key}')
    labels=a.obs[config.slice_key]
    if labels.isna().any():raise ValueError('missing slice labels')
    slices=tuple(labels.astype(str))
    if any(not v or v in ('nan','None','<NA>') for v in slices):raise ValueError('invalid slice labels')
    if set(slices)!=set(config.slice_shards):raise ValueError('slice_shards must match every input slice exactly')
    if any(slices.count(v)<2 for v in set(slices)):raise ValueError('each slice needs at least two spots')
    coords=np.array(a.obsm.get('spatial'),copy=True)
    if coords.shape!=(len(spots),2) or not np.isfinite(coords).all():raise ValueError('finite spatial coordinates of shape (spots,2) required')
    X=canonical_csr(a.X,binary=True)
    if min(X.shape)==0 or not X.nnz:raise ValueError('empty counts')
    chrom=np.array([v.startswith('chr') for v in peaks]);keep_min=np.zeros(X.shape[1],bool)
    labels=np.asarray(slices)
    for label in sorted(set(slices)):
        idx=np.flatnonzero(labels==label)
        threshold=max(config.feature_min_spots,int(np.ceil(len(idx)*config.feature_min_fraction)))
        keep_min|=np.asarray(X[idx].getnnz(axis=0)).ravel()>=threshold
    keep=chrom&keep_min&(np.asarray(X.getnnz(axis=0)).ravel()<=int(np.floor(len(spots)*config.feature_max_fraction)))
    X=X[:,keep].tocsr();selected=tuple(np.asarray(peaks)[keep])
    peak_keys(selected)
    if not X.nnz:raise ValueError('prevalence filter retained no positive peaks')
    degree=np.asarray(X.getnnz(axis=1)).ravel()
    if (degree==0).any():raise ValueError('empty filtered spots have no gate degree')
    if (degree==X.shape[1]).any():raise ValueError('all-positive filtered spots have no legal negative')
    return PreparedData(X,spots,selected,coords,slices)


def prepare_multislice_training(counts, config, output):
    """Prepare saved network inputs, or retain the original raw-count entry.

    Raw AnnData/H5AD preparation keeps its original filtering and payload.
    Saved networks bind training to their already normalized static graphs.
    Reference annotations never enter either training input form.
    """
    if isinstance(counts,StageArtifact) or (isinstance(counts,(str,Path)) and Path(counts).is_dir()):
        from .stages import _prepare_network_training
        return _prepare_network_training(counts,config,output)
    data=_filter_multislice_data(counts,config)
    X=data.X;spots=data.spot_ids;selected=data.peak_ids;coords=data.coordinates;slices=data.slice_ids
    output=protect_output(output,[counts] if isinstance(counts,(str,Path)) else [])
    if output.exists():raise FileExistsError(output)
    metadata=stage_metadata(output,shape=list(X.shape),spot_ids=spots,peak_ids=selected,slice_ids=slices,
        data_fingerprint=data.fingerprint,preparation_config=preparation_config(config),sampling_config=sampling_config(config),
        positive_edges=int(X.nnz),protocol='slice_union_prevalence_global_negatives_v1')
    with artifact_writer(output,'multislice_training_inputs',metadata) as stage:
        sp.save_npz(stage/'counts.npz',X)
        np.save(stage/'coordinates.npy',coords,allow_pickle=False)
        write_json(stage/'data.json',{'spot_ids':spots,'peak_ids':selected,'slice_ids':slices,'dna_ids':[],'dna_kind':None})
    return MultiSliceInputs.load(output)
