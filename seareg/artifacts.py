"""Immutable stage handles, safe relative dependencies and sparse payload IO."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import ClassVar
from functools import wraps
from types import MappingProxyType
from collections.abc import Mapping
import numpy as np
import pandas as pd
from .utils import verify_artifact,file_sha256,identity,json_safe

_STAGE_MODULES={
    'filtered':('preprocessing.py','data.py','artifacts.py','utils.py'),
    'sequences':('sequence.py','data.py','artifacts.py','utils.py'),
    'annotation':('annotation.py','sequence.py','data.py','artifacts.py','utils.py','scanners/legacy.py'),
    'selected_edges':('edges.py','annotation.py','data.py','artifacts.py','utils.py'),
    'network':('graph.py','data.py','artifacts.py','utils.py'),
    'training_inputs':('sampling.py','graph.py','data.py','artifacts.py','utils.py'),
    'training':('train.py','model.py','loss.py','scoring.py','_cuda_scoring.py','resident.py','sampling.py',
                'checkpoint.py','exports.py','_runtime.py','metrics.py','data.py','artifacts.py','utils.py'),
    'analysis':('analysis.py','domains.py','embedding.py','_analysis_io.py',
                '_runtime.py','data.py','artifacts.py','utils.py'),
    'render':('plotting.py','_label_layout.py','_analysis_io.py','artifacts.py','utils.py'),
    'legacy':('scanners/legacy.py','scanners/scan_for_kmers_motifs_new_2024.R',
              'scanners/legacy_motif_recipe.json','data.py','artifacts.py','utils.py'),
}


def implementation_id(stage):
    root=Path(__file__).parent
    return identity({name:file_sha256(root/name) for name in _STAGE_MODULES[stage]})

_ACTIVE_VALIDATION = ContextVar('seareg_validation', default=None)


def _file_state(path, manifest):
    result=[]
    for name in ('manifest.json', *manifest['files']):
        candidate=path/name
        stat=candidate.stat()
        result.append((name,stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns))
    actual=tuple(sorted(str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() or p.is_symlink()))
    return tuple(result),actual


def freeze_manifest(value):
    if isinstance(value,MappingProxyType):return value
    if isinstance(value,Mapping):return MappingProxyType({key:freeze_manifest(item) for key,item in value.items()})
    if isinstance(value,(list,tuple)):return tuple(freeze_manifest(item) for item in value)
    return value


class ValidationContext:
    """One-operation cache, never timestamp-only trust across operations."""
    def __init__(self):
        self.verified={}
        self.visiting=set()

    def verify(self,path,kind=None):
        path=Path(path).resolve()
        if path in self.visiting:raise ValueError('cyclic artifact dependencies')
        cached=self.verified.get(path)
        if cached is None:
            initial=json.loads((path/'manifest.json').read_text())
            before=_file_state(path,initial)
            manifest=verify_artifact(path,kind)
            if before!=_file_state(path,manifest):raise ValueError('artifact changed while verifying content')
            manifest=freeze_manifest(manifest)
            self.visiting.add(path)
            try:
                for entry in manifest['metadata'].get('upstreams',{}).values():
                    parent=self.verify(resolve_reference(path,manifest,entry),entry['kind'])
                    if parent['fingerprint']!=entry['fingerprint']:
                        raise ValueError('upstream artifact identity changed')
            finally:self.visiting.remove(path)
            self.verified[path]=(manifest,_file_state(path,manifest))
        else:
            manifest,state=cached
            if state!=_file_state(path,manifest):raise ValueError('artifact changed inside a verified operation')
            if kind is not None and manifest['kind']!=kind:raise ValueError('artifact kind mismatch')
            for entry in manifest['metadata'].get('upstreams',{}).values():
                parent=self.verify(resolve_reference(path,manifest,entry),entry['kind'])
                if parent['fingerprint']!=entry['fingerprint']:
                    raise ValueError('upstream artifact identity changed')
        return manifest


@contextmanager
def validation_scope():
    existing=_ACTIVE_VALIDATION.get()
    token=None
    if existing is None:
        existing=ValidationContext();token=_ACTIVE_VALIDATION.set(existing)
    try:yield existing
    finally:
        if token is not None:_ACTIVE_VALIDATION.reset(token)


def validated_operation(function):
    @wraps(function)
    def operation(*args,**kwargs):
        with validation_scope():return function(*args,**kwargs)
    return operation


def resolve_reference(owner,manifest,entry):
    relative=Path(entry['path'])
    if relative.is_absolute():raise ValueError('artifact references must be relative')
    bound=manifest['metadata'].get('workspace_root','.')
    if Path(bound).is_absolute():raise ValueError('workspace root must be relative')
    workspace=(Path(owner)/bound).resolve()
    lexical=Path(os.path.abspath(Path(owner)/relative));target=lexical.resolve()
    if workspace==Path('/') or not target.is_relative_to(workspace) or lexical!=target:
        raise ValueError('unsafe artifact reference outside its workspace')
    return target


def stage_metadata(output, *, upstreams=None, **metadata):
    output=Path(output).resolve();parents=dict(upstreams or {})
    with validation_scope():
        parents={name:StageArtifact.load(parent) for name,parent in parents.items()}
    workspace=Path(os.path.commonpath([str(output),*(str(p.path.resolve()) for p in parents.values())]))
    if parents and workspace==Path('/'):raise ValueError('artifact dependencies must share a workspace')
    refs={name:{'path':os.path.relpath(parent.path,output),'kind':parent.manifest['kind'],
                'fingerprint':parent.manifest['fingerprint']} for name,parent in parents.items()}
    return {'artifact_schema_version':2,'workspace_root':os.path.relpath(workspace,output),
            'upstreams':refs,**metadata}


@dataclass(frozen=True)
class StageArtifact:
    path: Path
    manifest: dict
    KIND: ClassVar[str|None]=None

    def __post_init__(self):object.__setattr__(self,'manifest',freeze_manifest(self.manifest))

    @classmethod
    def load(cls,path):
        bound=path.fingerprint if isinstance(path,StageArtifact) else None
        if isinstance(path,StageArtifact):path=path.path
        path=Path(path).resolve()
        with validation_scope() as context:manifest=context.verify(path,cls.KIND)
        if bound is not None and manifest['fingerprint']!=bound:
            raise ValueError('artifact identity replaced at the bound handle path')
        return cls(path,manifest)

    @property
    def metadata(self):return self.manifest['metadata']

    @property
    def fingerprint(self):return self.manifest['fingerprint']

    @property
    def peak_ids(self):return tuple(self.metadata['peak_ids'])

    def upstream(self,name,cls=None):
        self.verify()
        entry=self.metadata['upstreams'][name]
        result=(cls or StageArtifact).load(resolve_reference(self.path,self.manifest,entry))
        if result.fingerprint!=entry['fingerprint']:raise ValueError('upstream artifact identity changed')
        return result

    def verify(self):return type(self).load(self)


@dataclass(frozen=True)
class FilteredCounts(StageArtifact):
    KIND: ClassVar[str]='filtered_counts'

    @validated_operation
    def read_counts(self):
        import anndata as ad
        self.verify();a=ad.read_h5ad(self.path/'counts.h5ad')
        if tuple(a.obs_names)!=tuple(self.metadata['spot_ids']) or tuple(a.var_names)!=self.peak_ids:
            raise ValueError('filtered counts axis identity mismatch')
        return a

    def filter_decisions(self):
        self.verify()
        return pd.read_csv(self.path/'peak_filter.tsv.gz',sep='\t',
                           dtype={'peak_id':str,'canonical_interval':str},keep_default_na=False)

    @validated_operation
    def load_data(self):
        from .data import PreparedData
        a=self.read_counts()
        return PreparedData(a.X,tuple(a.obs_names),tuple(a.var_names),a.obsm.get('spatial'),
            tuple(self.metadata['slice_ids']),provenance=json_safe({'filter':self.metadata['filter_report'],
                'filter_config':self.metadata['filter_config'],'filtered_counts':self.fingerprint}))


def save_tensor_graph(path,tensor):
    """Persist coalesced values exactly, not a recipe for renormalization."""
    tensor=tensor.coalesce().detach().cpu()
    np.savez_compressed(path,indices=tensor.indices().numpy(),values=tensor.values().numpy(),
                        shape=np.asarray(tensor.shape,dtype=np.int64))


@dataclass(frozen=True)
class PeakSequences(StageArtifact):
    KIND: ClassVar[str]='peak_sequences'

    def read_sequences(self):
        self.verify();result=[]
        with (self.path/'peaks.fa').open() as stream:
            token=None
            for line in stream:
                if line.startswith('>'):
                    token=line[1:].strip();result.append('')
                elif token is not None:result[-1]+=line.strip()
                else:raise ValueError('invalid peak sequence FASTA')
        if len(result)!=len(self.peak_ids):raise ValueError('sequence axis mismatch')
        return result


@dataclass(frozen=True)
class ModelNetwork(StageArtifact):
    @classmethod
    def load(cls,path):
        result=super().load(path)
        if result.manifest['kind'] not in ('model_network','prepared'):
            raise ValueError('artifact is not a prepared model network')
        return result

    @validated_operation
    def load_data(self):
        if self.manifest['kind']=='prepared':
            self.verify()
            from .compat import load_prepared_v1
            return load_prepared_v1(self.path)
        from .data import PreparedData
        import scipy.sparse as sp
        filtered=self.upstream('filtered',FilteredCounts);data=filtered.load_data()
        meta=json.loads((self.path/'data.json').read_text())
        result=PreparedData(data.X,tuple(meta['spot_ids']),tuple(meta['peak_ids']),data.coordinates,
            tuple(meta['slice_ids']),sp.load_npz(self.path/'effective_motif_edges.npz')
                if self.metadata['has_motif'] else None,tuple(meta['dna_ids']),meta['dna_kind'],meta['provenance'])
        if result.fingerprint!=self.metadata['data_fingerprint']:raise ValueError('network data fingerprint mismatch')
        return result

    @validated_operation
    def load_graphs(self):
        self.verify()
        if self.manifest['kind']=='prepared':raise ValueError('legacy network requires explicit static-graph migration')
        import torch
        with np.load(self.path/'degrees.npz',allow_pickle=False) as a:
            return {'spatial':load_tensor_graph(self.path/'spatial_adjacency.npz') if self.metadata['has_spatial'] else None,
                'dna':load_tensor_graph(self.path/'motif_adjacency.npz') if self.metadata['has_motif'] else None,
                'spatial_degree':torch.from_numpy(a['spatial_degree'].copy()),
                'peak_degree':torch.from_numpy(a['peak_degree'].copy()) if self.metadata['has_motif'] else None,
                'motif_degree':torch.from_numpy(a['motif_degree'].copy()) if self.metadata['has_motif'] else None}

    def motif_node_map(self):
        self.verify()
        return pd.read_csv(self.path/'motif_node_map.tsv.gz',sep='\t',
            dtype={'motif_id':str,'representative_id':str},keep_default_na=False)

    def edge_lineage(self):
        self.verify()
        return pd.read_csv(self.path/'motif_edge_lineage.tsv.gz',sep='\t',
            dtype={'peak_id':str,'motif_id':str,'representative_id':str},keep_default_na=False)


@dataclass(frozen=True)
class TrainingInputs(StageArtifact):
    KIND: ClassVar[str]='training_inputs'

    def network(self):return self.upstream('network',ModelNetwork)

    @validated_operation
    def load_split(self):
        self.verify()
        with np.load(self.path/'split.npz',allow_pickle=False) as arrays:
            result={key:arrays[key].copy() for key in arrays.files}
        result['sampling_config']=dict(self.metadata['sampling_config'])
        return result

    def load_full_graph(self):
        self.verify()
        return load_tensor_graph(self.path/'full_train_adjacency.npz')


def load_tensor_graph(path):
    import torch
    with np.load(path,allow_pickle=False) as arrays:
        indices=torch.from_numpy(arrays['indices'].copy());values=torch.from_numpy(arrays['values'].copy())
        shape=tuple(map(int,arrays['shape']))
    return torch.sparse_coo_tensor(indices,values,shape).coalesce()
