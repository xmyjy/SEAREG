"""Validated preprocessing, training, analysis and workflow configuration."""

from dataclasses import dataclass, fields
import math
from typing import get_args
from dataclasses import asdict
import json
from pathlib import Path


def _positive(value, name, zero=False):
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')


def _validate_types(config):
    for field in fields(config):
        value = getattr(config, field.name)
        expected = field.type
        options = get_args(expected)
        if type(None) in options:
            if value is None:
                continue
            expected, = (option for option in options if option is not type(None))
        if expected is bool and type(value) is not bool:
            raise ValueError(f"{field.name} must be a JSON boolean")
        if expected is int and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"{field.name} must be an integer")
        if expected is float and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            raise ValueError(f"{field.name} must be numeric")
        if expected is str and not isinstance(value, str):
            raise ValueError(f"{field.name} must be a string")


@dataclass(frozen=True)
class FilterConfig:
    filter_protocol: str = 'copy1_frequency_v1'
    slice_key: str | None = None

    def __post_init__(self):
        _validate_types(self)
        if self.filter_protocol!='copy1_frequency_v1':raise ValueError('unsupported peak filtering protocol')
        if self.slice_key is not None and not self.slice_key:raise ValueError('slice_key must be nonempty')


@dataclass(frozen=True)
class SequenceConfig:
    fasta: str
    assembly: str

    def __post_init__(self):
        _validate_types(self)
        if not self.fasta.strip():raise ValueError('reference FASTA is required')
        if not self.assembly.strip() or any(ord(c)<32 for c in self.assembly):
            raise ValueError('explicit assembly identity is required')


@dataclass(frozen=True)
class AnnotationConfig:
    motifs: str
    fimo: str = 'fimo'
    p_cutoff: float = 1e-5
    motif_pseudocount: float = 0.0
    timeout: float = 3600.0

    def __post_init__(self):
        _validate_types(self)
        if not self.motifs.strip() or not self.fimo.strip():raise ValueError('explicit motif resource/scanner required')
        if not 0<self.p_cutoff<=1 or not math.isfinite(self.p_cutoff):raise ValueError('invalid scan cutoff')
        _positive(self.motif_pseudocount,'motif_pseudocount',zero=True)
        _positive(self.timeout,'timeout')


@dataclass(frozen=True)
class LegacyAnnotationConfig:
    p_cutoff: float = 1e-5
    threads: int = 1
    species_names: tuple[str,...] = ('Homo sapiens','Mus musculus')
    rscript: str = 'Rscript'
    timeout: float = 3600.0
    cache_dir: str | None = None

    def __post_init__(self):
        _validate_types(self)
        if not 0<self.p_cutoff<=1 or not math.isfinite(self.p_cutoff):raise ValueError('invalid legacy scan cutoff')
        if type(self.threads) is not int or self.threads<1:raise ValueError('threads must be a positive integer')
        if not self.species_names or any(not isinstance(s,str) or not s.strip() or ',' in s for s in self.species_names):
            raise ValueError('species_names must contain explicit species names')
        if not self.rscript.strip():raise ValueError('Rscript executable required')
        if self.cache_dir is not None and not self.cache_dir.strip():raise ValueError('cache_dir must be nonempty')
        _positive(self.timeout,'timeout')


@dataclass(frozen=True)
class EdgeSelectionConfig:
    selection: str
    p_cutoff: float | None = None

    def __post_init__(self):
        _validate_types(self)
        if self.selection not in ('all_candidates','site_p'):raise ValueError('unknown edge selection rule')
        if self.selection=='site_p':
            if self.p_cutoff is None or not math.isfinite(self.p_cutoff) or not 0<self.p_cutoff<=1:
                raise ValueError('site_p requires an explicit cutoff')
        elif self.p_cutoff is not None:raise ValueError('all_candidates does not accept another cutoff')


@dataclass(frozen=True)
class PreprocessConfig:
    use_spatial: bool = True
    spatial_method: str = "radius"
    spatial_neighbors: float = 4
    spatial_distance: str = "const"
    slice_key: str | None = None
    dna_kind: str = "motif"
    dna_weight: str = "binary"
    motif_deduplicate: bool = True
    allow_partial_dna: bool = False
    filter_protocol: str = "copy1_frequency_v1"

    def __post_init__(self):
        _validate_types(self)
        if self.spatial_method not in ("radius", "knn") or self.spatial_distance not in (
            "const",
            "recip",
            "recip2",
        ):
            raise ValueError("invalid spatial graph protocol")
        _positive(self.spatial_neighbors, "spatial_neighbors")
        if self.spatial_method == "knn" and int(self.spatial_neighbors) != self.spatial_neighbors:
            raise ValueError("knn requires an integer neighbor count")
        if self.dna_kind not in ("motif", "kmer") or self.dna_weight not in ("binary", "value"):
            raise ValueError("invalid DNA kind/weight")
        if self.filter_protocol != "copy1_frequency_v1":
            raise ValueError("unsupported filter protocol")


@dataclass(frozen=True)
class NetworkConfig:
    use_spatial: bool = True
    spatial_method: str = 'radius'
    spatial_neighbors: float = 4
    spatial_distance: str = 'const'
    dna_kind: str = 'motif'
    dna_weight: str = 'binary'
    motif_deduplicate: bool = True
    allow_partial_dna: bool = False

    def __post_init__(self):
        PreprocessConfig(**asdict(self))


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 300
    hidden_dim: int = 64
    seed: int = 1234
    device: str = "cpu"
    lr: float = 0.001
    decay: float = 0.001
    dna_decay: float = 0.00001
    dna_reg_alpha: float = 0.5
    motif_cap_factor: float = 2.0
    spatial_weight: float = 0.5
    dna_weight: float = 0.1
    pre_train_epochs: int = 0
    shards: int = 10
    pn_r: int = 10
    tr: float = 1.0
    reshard_every: int = 100
    negative_refresh: int = 5
    export_every: int = 100
    export_start_epoch: int = 300
    checkpoint_every: int = 10
    loss_chunk_size: int = 0
    dense_mask_mb: float = 512.0
    mode: str = "single"
    recall_candidates: str = "unseen"
    grouped_loss: bool = False
    loss_backend: str = "torch"
    loss_workspace_mb: float = 256.0
    weighted_ego_l2: bool = False
    resident_cache: bool = True
    resident_cache_budget_gib: float = 0.0
    resident_cache_reserve_gib: float = 4.0

    def __post_init__(self):
        _validate_types(self)
        for name in (
            "epochs",
            "hidden_dim",
            "shards",
            "pn_r",
            "export_every",
            "export_start_epoch",
            "checkpoint_every",
        ):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "seed",
            "pre_train_epochs",
            "reshard_every",
            "negative_refresh",
            "loss_chunk_size",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not 0 < self.tr <= 1:
            raise ValueError("tr must be in (0,1]")
        if self.seed >= 2**32:
            raise ValueError("seed must be in [0, 2**32-1]")
        for name in ("lr",):
            _positive(getattr(self, name), name)
        for name in (
            "decay",
            "dna_decay",
            "motif_cap_factor",
            "spatial_weight",
            "dna_weight",
            "dense_mask_mb",
        ):
            _positive(getattr(self, name), name, zero=True)
        _positive(self.loss_workspace_mb, "loss_workspace_mb")
        for name in ("resident_cache_budget_gib", "resident_cache_reserve_gib"):
            _positive(getattr(self, name), name, zero=True)
        if not math.isfinite(self.dna_reg_alpha):
            raise ValueError("dna_reg_alpha must be finite")
        if self.loss_backend not in ("torch", "torch_streamed", "auto", "cuda_fused"):
            raise ValueError("unsupported loss backend")
        if self.loss_backend in ("torch_streamed", "auto", "cuda_fused") and self.loss_chunk_size:
            raise ValueError("loss_chunk_size must be zero with torch_streamed, auto or cuda_fused")
        if self.grouped_loss and self.loss_chunk_size and self.loss_chunk_size % self.pn_r:
            raise ValueError("loss_chunk_size must be a multiple of pn_r for grouped loss")
        if self.mode != "single":
            raise ValueError("unsupported training mode; only single is supported")
        if self.recall_candidates not in ("unseen", "all"):
            raise ValueError("invalid recall candidates")

    def result_export_due(self, completed_epoch):
        """Save periodic results from the first export epoch, plus completion."""
        return completed_epoch == self.epochs or (
            completed_epoch >= self.export_start_epoch
            and (completed_epoch - self.export_start_epoch) % self.export_every == 0
        )


def training_profile(profile="balanced", **overrides):
    """Explicit execution presets; scientific options remain caller-controlled.

    Workspace is a conservative loss-chunk planning budget, not a bound on total
    process VRAM. Cache budgets are controlled independently.
    """
    presets = {
        "balanced": {"loss_backend": "auto", "loss_workspace_mb": 1024.0},
        "speed": {"loss_backend": "torch", "loss_workspace_mb": 1024.0},
        "memory": {"loss_backend": "torch_streamed", "loss_workspace_mb": 256.0},
    }
    if profile not in presets:
        raise ValueError("profile must be balanced, speed or memory")
    return TrainConfig(**{**presets[profile], "grouped_loss": True,
                         "weighted_ego_l2": True, "resident_cache": True,
                         **overrides})


@dataclass(frozen=True)
class AnalysisConfig:
    target_domains: tuple[int, ...] = ()
    pca_components: int = 20
    neighbors: int = 15
    seed: int = 1234
    louvain_seed: int = 0
    resolution_max: float = 3.0
    max_search_steps: int = 20
    min_dist: float = 0.08
    joint: bool = True
    domains: bool = True
    spot_umap: bool = True
    neighbor_backend: str = "auto"
    fixed_resolution: float | None = None
    pca_solver: str = "auto"

    def __post_init__(self):
        _validate_types(self)
        if self.neighbor_backend not in ("auto", "exact"):
            raise ValueError("neighbor_backend must be auto or exact")
        if self.pca_solver not in ("auto", "full", "covariance_eigh", "arpack", "randomized"):
            raise ValueError("unsupported pca_solver")
        object.__setattr__(self, "target_domains", tuple(self.target_domains))
        if self.fixed_resolution is not None:
            _positive(self.fixed_resolution, "fixed_resolution")
            if self.target_domains:
                raise ValueError("fixed_resolution and target_domains are mutually exclusive")
        if self.domains and not self.target_domains and self.fixed_resolution is None:
            raise ValueError("target_domains must be explicit")
        for k in self.target_domains:
            if not isinstance(k, int) or isinstance(k, bool) or k < 1:
                raise ValueError("invalid target K")
        if len(set(self.target_domains)) != len(self.target_domains):
            raise ValueError("duplicate target K")
        for name in ("pca_components", "neighbors", "max_search_steps"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"invalid {name}")
        if self.neighbors < 2:
            raise ValueError("neighbors must be >=2")
        if not 0 <= self.seed < 2**32 or self.louvain_seed != 0:
            raise ValueError(
                "copy1 spatial requires nonnegative analysis seed and Louvain default seed0"
            )
        _positive(self.resolution_max, "resolution_max")
        if not 0 <= self.min_dist <= 1:
            raise ValueError("min_dist must be in [0,1]")


def from_dict(cls, values):
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def sampling_config(config):
    return {name:getattr(config,name) for name in ('seed','tr','shards','pn_r')}


def load_config(path):
    """Resolve v1/v2 once into the same typed stage configuration."""
    if isinstance(path,dict):
        raw=path;path=Path.cwd()/'in_memory_config.json'
    else:
        path=Path(path).resolve();raw=json.loads(path.read_text())
    if not isinstance(raw,dict):raise ValueError('workflow must be an object')
    version=raw.get('schema_version',1)
    if type(version) is not int or version not in (1,2):raise ValueError('unsupported workflow schema_version')
    common={'schema_version','dataset_id','input','training','training_profile','analysis','execution','output','plots','motif','motif_source'}
    allowed=common|({'network','preprocessing','annotation'} if version==1 else
                    {'filtering','sequence','annotation','edge_selection','network'})
    if set(raw)-allowed:raise ValueError('workflow has unknown sections')
    dataset=raw.get('dataset_id')
    if not isinstance(dataset,str) or not dataset.strip():raise ValueError('dataset_id must be nonempty')
    for name in allowed-{'schema_version','dataset_id','training_profile','output','plots','motif','motif_source'}:
        if name in raw and not isinstance(raw[name],dict):raise ValueError(f'{name} must be an object')
    from .resources import normalize_motif_request
    raw,reference_download=normalize_motif_request(raw,path.parent)
    def resolve(value,label):
        if not isinstance(value,str) or not value.strip():raise ValueError(f'{label} must be a nonempty path')
        return (path.parent/value).resolve()
    def executable(value,label):
        if not isinstance(value,str) or not value.strip():raise ValueError(f'{label} must be an executable path/name')
        return str(resolve(value,label)) if '/' in value else value
    source_input=raw.get('input',{})
    valid_inputs={'atac','dna'} if version==1 else {'atac','dna','filtered','network'}
    if set(source_input)-valid_inputs:raise ValueError('unknown input keys')
    inputs={name:resolve(value,'input.'+name) for name,value in source_input.items()}
    migration=[];network_import=None
    if version==1:
        if 'atac' not in inputs:raise ValueError('input requires atac')
        original_network=raw.get('network',{})
        if set(original_network)-{'prepared'}:raise ValueError('network supports prepared import only in v1')
        if original_network.get('prepared') is not None:network_import=resolve(original_network['prepared'],'network.prepared')
        pc=from_dict(PreprocessConfig,raw.get('preprocessing',{}))
        filtering=FilterConfig(pc.filter_protocol,pc.slice_key)
        nc=NetworkConfig(**{name:getattr(pc,name) for name in NetworkConfig.__dataclass_fields__})
        migration=['v1 preprocessing split into filtering/network views',
                   'v1 annotation graph options moved to explicit edge_selection']
    else:
        choices=sum(name in inputs for name in ('atac','filtered','network'))
        if choices!=1:raise ValueError('input requires exactly one of atac, filtered, network')
        network_import=inputs.get('network')
        filtering=from_dict(FilterConfig,raw.get('filtering',{}))
        nc=from_dict(NetworkConfig,raw.get('network',{}))
        pc=PreprocessConfig(**{**asdict(nc),'filter_protocol':filtering.filter_protocol,'slice_key':filtering.slice_key})
    annotation=raw.get('annotation');ac=None;sequence=None;edge=None
    sequence_raw=raw.get('sequence') if version==2 else None
    if annotation is not None:
        if 'dna' in inputs or network_import is not None:raise ValueError('annotation cannot be combined with input.dna or prepared network import')
        annotation=dict(annotation)
        if annotation.get('method') not in ('fimo','legacy'):raise ValueError('annotation requires method=fimo or legacy')
        if version==1:
            sequence_raw={name:annotation.pop(name,None) for name in ('fasta','assembly')}
            selection=annotation.pop('graph_selection',None);cutoff=annotation.pop('graph_p_cutoff',None)
            if selection is not None:edge=EdgeSelectionConfig(selection,cutoff)
            elif cutoff is not None:raise ValueError('graph_p_cutoff requires graph_selection')
        elif 'edge_selection' in raw:edge=from_dict(EdgeSelectionConfig,raw['edge_selection'])
        method=annotation.pop('method')
        if method=='fimo':
            annotation['motifs']=str(resolve(annotation.get('motifs'),'annotation.motifs'))
            if 'fimo' in annotation:annotation['fimo']=executable(annotation['fimo'],'annotation.fimo')
            ac=from_dict(AnnotationConfig,annotation)
            if edge is not None and edge.selection=='site_p' and edge.p_cutoff>ac.p_cutoff:
                raise ValueError('edge cutoff must be within scan threshold')
        else:
            if edge is not None:raise ValueError('legacy scanner preserves its bundled graph recipe')
            if 'rscript' in annotation:annotation['rscript']=executable(annotation['rscript'],'annotation.rscript')
            if annotation.get('cache_dir') is not None:annotation['cache_dir']=str(resolve(annotation['cache_dir'],'annotation.cache_dir'))
            if 'species_names' in annotation:
                if not isinstance(annotation['species_names'],(list,tuple)):raise ValueError('species_names requires a list')
                annotation['species_names']=tuple(annotation['species_names'])
            ac=from_dict(LegacyAnnotationConfig,annotation)
        annotation={'method':method,**asdict(ac)}
    elif 'edge_selection' in raw:raise ValueError('edge_selection requires a complete annotation')
    if sequence_raw is not None:
        if set(sequence_raw)-{'fasta','assembly'}:raise ValueError('unknown sequence options')
        sequence=SequenceConfig(str(resolve(sequence_raw.get('fasta'),'sequence.fasta')),sequence_raw.get('assembly'))
    if ac is not None and sequence is None:raise ValueError('annotation requires an explicit sequence reference/assembly')
    execution={'physical_gpu':None,'torch_threads':8,'torch_interop_threads':1,'blas_threads':1,'strict_algorithms':True,'tf32':False}
    if set(raw.get('execution',{}))-set(execution):raise ValueError('unknown execution options')
    execution.update(raw.get('execution',{}))
    gpu=execution['physical_gpu']
    if gpu is not None and (type(gpu) is not int or gpu<0):raise ValueError('physical_gpu must be null or nonnegative integer')
    for name in ('torch_threads','torch_interop_threads','blas_threads'):
        if type(execution[name]) is not int or execution[name]<1:raise ValueError(f'{name} must be a positive integer')
    for name in ('strict_algorithms','tf32'):
        if type(execution[name]) is not bool:raise ValueError(f'{name} must be boolean')
    profile=raw.get('training_profile')
    training_options=dict(raw.get('training',{}))
    auto_fields=tuple(name for name in ('epochs','shards')
                      if version==2 and (name not in training_options or training_options[name]=='auto'))
    for name in auto_fields:training_options.pop(name,None)
    training=training_profile(profile,**training_options) if profile is not None else from_dict(TrainConfig,training_options)
    if gpu is not None and training.device!='cuda:0':raise ValueError('physical_gpu requires worker training.device=cuda:0')
    if training.device.startswith('cuda') and gpu is None:raise ValueError('CUDA training requires explicit execution.physical_gpu')
    plots=raw.get('plots',True)
    if type(plots) is not bool:raise ValueError('plots must be boolean')
    return {'schema_version':2,'source_config_version':version,'source':path,'dataset_id':dataset,
        'input':inputs,'filtering':filtering,'sequence':sequence,'annotation':annotation,'annotation_config':ac,
        'edge_selection':edge,'network_config':nc,'network_import':network_import,
        'filter_options_explicit':version==1 or bool(raw.get('filtering',{})),
        'network_options_explicit':bool(raw.get('network',{})) if version==2 else True,
        'preprocessing':pc,'network':{} if network_import is None else {'prepared':network_import},
        'training':training,'training_auto_fields':auto_fields,'training_defaults_resolved':not auto_fields,
        'analysis':from_dict(AnalysisConfig,raw['analysis']) if 'analysis' in raw else None,
        'execution':execution,'plots':plots,'output':resolve(raw.get('output'),'output'),'migration_report':migration,
        'motif':raw['motif'],'reference_download':reference_download}


def effective_config(config):
    def plain(value):
        if hasattr(value,'__dataclass_fields__'):return plain(asdict(value))
        if isinstance(value,Path):return str(value)
        if isinstance(value,dict):return {key:plain(item) for key,item in value.items()}
        if isinstance(value,(list,tuple)):return [plain(item) for item in value]
        return value
    keys=('schema_version','source_config_version','dataset_id','input','filtering','sequence','motif',
          'edge_selection','training','analysis','execution','plots','output','migration_report')
    result={key:plain(config[key]) for key in keys}
    if not config.get('training_defaults_resolved',True):
        for name in config.get('training_auto_fields',()):result['training'][name]='auto'
    result['network']=plain(config['network_config'])
    result['annotation']=None if config['annotation_config'] is None else {
        'method':config['annotation']['method'],**plain(config['annotation_config'])}
    return result
