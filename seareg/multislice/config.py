"""Scientific protocol and separate checkpoint/export management."""
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class MultiSliceConfig:
    slice_shards: dict[str, int] = field(default_factory=lambda: {
        'E11_0-S1':10,'E13_5-S1':10,'E15_5-S1':20,'E18_5-S1':20})
    slice_key: str = 'sample_id'
    epochs: int = 500
    seed: int = 1234
    hidden_dim: int = 64
    lr: float = .001
    decay: float = .001
    pn_r: int = 5
    spatial_weight: float = .3
    gradient_clip: float = .5
    updates_per_epoch: int = 20
    reshard_every: int = 100
    negative_refresh: int = 5
    gate_null_permutations: int = 1000
    gate_refresh_every: int = 50
    feature_min_fraction: float = .005
    feature_min_spots: int = 5
    feature_max_fraction: float = .995
    spatial_radius_multiplier: float = 4.
    device: str = 'cuda:0'
    # Execution/transfer and saving choices do not change the objective.
    loss_backend: str = 'auto'
    loss_workspace_mb: float = 1024.
    grouped_loss: bool = True
    weighted_ego_l2: bool = True
    resident_cache: bool = True
    resident_cache_budget_gib: float = 0.
    resident_cache_reserve_gib: float = 4.
    checkpoint_every: int = 10
    export_start_epoch: int = 300
    export_every: int = 100
    correction_mode: str = 'adaptive_depth_gate'

    def __post_init__(self):
        shards=dict(self.slice_shards)
        if not shards or any(not isinstance(k,str) or not k or type(v) is not int or v<1 for k,v in shards.items()):
            raise ValueError('slice_shards requires named positive integer counts')
        object.__setattr__(self,'slice_shards',shards)
        for name in ('epochs','hidden_dim','pn_r','updates_per_epoch','gate_refresh_every','feature_min_spots',
                     'checkpoint_every','export_start_epoch','export_every'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:raise ValueError(f'{name} must be positive integer')
        for name in ('seed','reshard_every','negative_refresh'):
            if type(getattr(self,name)) is not int or getattr(self,name)<0:raise ValueError(f'{name} must be nonnegative integer')
        if type(self.gate_null_permutations) is not int or self.gate_null_permutations<2:
            raise ValueError('gate_null_permutations must be at least two')
        for name in ('lr','gradient_clip','loss_workspace_mb','spatial_radius_multiplier'):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:raise ValueError(f'{name} must be finite positive')
        for name in ('decay','spatial_weight','resident_cache_budget_gib','resident_cache_reserve_gib'):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:raise ValueError(f'{name} must be finite nonnegative')
        if not 0<=self.feature_min_fraction<=self.feature_max_fraction<=1:raise ValueError('invalid prevalence fractions')
        if self.loss_backend!='auto' or self.grouped_loss is not True or self.weighted_ego_l2 is not True:
            raise ValueError('multislice requires auto grouped scoring and weighted ego L2')
        if type(self.resident_cache) is not bool:raise ValueError('resident_cache must be boolean')
        if not self.slice_key or not isinstance(self.slice_key,str):raise ValueError('slice_key must be a string')
        if self.device!='cpu' and self.device!='cuda:0':raise ValueError('device must be cpu or isolated cuda:0')
        if self.correction_mode != 'adaptive_depth_gate': raise ValueError('correction_mode must be adaptive_depth_gate')
        if self.gate_refresh_every != 50: raise ValueError('gate_refresh_every must be 50')

    def result_export_due(self, completed):
        return completed==self.epochs or (completed>=self.export_start_epoch and
            (completed-self.export_start_epoch)%self.export_every==0)

    @property
    def balanced_updates_per_slice(self):return self.updates_per_epoch
    @property
    def r_spatial(self):return self.spatial_weight
    @property
    def pre_train_epochs(self):return 0
    @property
    def shard_pairing_strategy(self):return 'independent_seeded_shuffle'
