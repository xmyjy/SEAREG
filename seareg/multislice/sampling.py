"""Frozen global-negative streams and slice-local graph construction."""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from .inputs import MultiSliceInputs, sampling_config, preparation_config


def normalized_adjacency(edges,n_nodes,weights=None,undirected=True):
    weights=torch.ones(edges.shape[1],dtype=torch.float32) if weights is None else weights.float()
    if undirected:
        edges=torch.cat([edges,edges.flip(0)],dim=1);weights=torch.cat([weights,weights.clone()])
        hashed=edges[0]*n_nodes+edges[1]
        unique,inverse=torch.unique(hashed,sorted=False,return_inverse=True)
        edges=torch.stack([torch.div(unique,n_nodes,rounding_mode='floor'),unique%n_nodes])
        summed=torch.zeros(unique.numel(),dtype=weights.dtype);summed.scatter_add_(0,inverse,weights);weights=summed
    degree=torch.zeros(n_nodes,dtype=torch.float32)
    degree.scatter_add_(0,edges[0],weights).clamp_(min=1e-8)
    values=weights*(1./torch.sqrt(degree[edges[0]]*degree[edges[1]]))
    return torch.sparse_coo_tensor(edges,values,(n_nodes,n_nodes)).coalesce()


def spatial_graph(coordinates,labels,multiplier):
    rows=[];cols=[]
    for label in pd.unique(labels):
        global_idx=np.flatnonzero(labels==label);coords=coordinates[global_idx]
        if len(np.unique(coords,axis=0))!=len(global_idx):
            raise ValueError(f'duplicate/coincident spatial coordinates within slice {label}')
        dist,_=NearestNeighbors(n_neighbors=2,algorithm='auto',n_jobs=1).fit(coords).kneighbors(coords)
        radius=max(float(np.min(dist[:,1]))*multiplier,1e-12)
        neighbors=NearestNeighbors(radius=radius,algorithm='auto',n_jobs=1).fit(coords)
        try:graph=neighbors.radius_neighbors_graph(coords,mode='distance',include_self=False)
        except TypeError:
            graph=neighbors.radius_neighbors_graph(coords,mode='distance');graph.setdiag(0);graph.eliminate_zeros()
        coo=graph.tocoo();rows.append(global_idx[coo.row]);cols.append(global_idx[coo.col])
    edges=torch.from_numpy(np.stack([np.concatenate(rows),np.concatenate(cols)])).long()
    undirected=torch.cat([edges,edges.flip(0)],dim=1)
    weights=torch.ones(undirected.shape[1],dtype=torch.float32)
    raw_sum=torch.sparse_coo_tensor(undirected,weights,(len(labels),len(labels))).coalesce()
    raw_count=torch.sparse_coo_tensor(undirected,weights,(len(labels),len(labels))).coalesce()
    raw_weight=raw_sum.values()/raw_count.values()
    degree=torch.zeros(len(labels));degree.index_add_(0,raw_sum.indices()[0],raw_weight)
    return normalized_adjacency(raw_sum.indices(),len(labels),raw_weight,False),degree


class ShardedEdgeSampler:
    """Global legal negatives, retaining the original SeedSequence hierarchy."""
    def __init__(self,csr,num_shards,pn_r,base_seed):
        self.csr=csr;self.num_spot,self.num_peak=csr.shape;self.num_shards=num_shards
        self.pn_r=pn_r;self.base_seed=base_seed
        self.pos_mask=np.zeros(csr.shape,bool);u,p=csr.nonzero();self.pos_mask[u,p]=True
        if np.any(self.pos_mask.all(axis=1)):raise ValueError('spot has no legal negative')
        root=np.random.SeedSequence(base_seed);seeds=root.spawn(self.num_spot)
        self.fixed=[];self.initial=[]
        for u in range(self.num_spot):
            rng=np.random.default_rng(seeds[u]);pos=csr.getrow(u).indices.astype(np.int64,copy=False)
            rng.shuffle(pos);self.fixed.append(pos.copy())
            self.initial.append(np.array_split(rng.permutation(len(pos)),num_shards))
        self.reshard_epoch=0;self.negative_epoch=0;self._build(self.initial)

    def _build(self,chunks):
        self.shard_pos=[];self.train_user_ids=[]
        for s in range(self.num_shards):
            self.shard_pos.append(np.concatenate([p[part[s]] for p,part in zip(self.fixed,chunks)]))
            self.train_user_ids.append(np.concatenate([np.full(len(part[s]),u,dtype=np.int64) for u,part in enumerate(chunks)]))

    def reshard(self,epoch):
        seeds=np.random.SeedSequence([self.base_seed,0xABCDEF01,epoch]).spawn(self.num_spot)
        self._build([np.array_split(np.random.default_rng(seeds[u]).permutation(len(p)),self.num_shards) for u,p in enumerate(self.fixed)])
        self.reshard_epoch=epoch

    def make_epoch_batches(self,epoch):
        seeds=np.random.SeedSequence([self.base_seed,0x9E3779B9,epoch]).spawn(self.num_shards)
        batches=[]
        for s,(rawu,rawp) in enumerate(zip(self.train_user_ids,self.shard_pos)):
            u=np.repeat(rawu,self.pn_r);p=np.repeat(rawp,self.pn_r)
            rng=np.random.default_rng(seeds[s]);n=rng.integers(0,self.num_peak,size=len(u),dtype=np.int64)
            collision=self.pos_mask[u,n]
            while collision.any():
                positions=np.nonzero(collision)[0]
                draws=rng.integers(0,self.num_peak,size=collision.sum(),dtype=np.int64)
                n[positions]=draws;collision[positions]=self.pos_mask[u[positions],draws]
            batches.append([torch.from_numpy(v).long() for v in (u,p,n)])
        self.negative_epoch=epoch
        return batches


@dataclass
class SliceShardBundle:
    code:int
    name:str
    global_spots:np.ndarray
    num_shards:int
    sampler:ShardedEdgeSampler
    batch_all:list
    adjacencies:list
    spatial_adjacency:torch.Tensor


class MultiSliceSampler:
    def __init__(self,inputs,config):
        self.inputs=MultiSliceInputs.load(inputs);self.config=config
        if dict(self.inputs.metadata['sampling_config'])!=sampling_config(config) or dict(self.inputs.metadata['preparation_config'])!=preparation_config(config):
            raise ValueError('training inputs and multislice config differ')
        self.data=self.inputs.load_data();self._csr=self.data.X
        self.num_spot,self.num_peak=self._csr.shape
        cat=pd.Categorical(self.data.slice_ids)
        self.spot_slice_codes=torch.from_numpy(cat.codes.astype(np.int64,copy=False)).long()
        self.slice_names=list(cat.categories);self.num_slices=len(self.slice_names)
        saved=(self.inputs.network().load_graphs()
               if 'network' in self.inputs.metadata.get('upstreams',{}) else None)
        if saved is None:
            spatial,degree=spatial_graph(self.data.coordinates,np.asarray(self.data.slice_ids),config.spatial_radius_multiplier)
        else:
            spatial,degree=saved['spatial'],saved['spatial_degree']
        self.data_spatial_graph=SimpleNamespace(edge_index=spatial,train_spot_spot_degree=degree)
        self.train_spot_peak_degree=(torch.from_numpy(np.asarray(self._csr.getnnz(axis=1)).ravel()).float()
                                    if saved is None else saved['spot_peak_degree'])
        self.slice_bundles=[];self.r_by_slice={}
        for code,name in enumerate(self.slice_names):
            spots=np.flatnonzero(np.asarray(cat.codes)==code).astype(np.int64);n=config.slice_shards[name]
            sampler=ShardedEdgeSampler(self._csr[spots].tocsr(),n,config.pn_r,config.seed+100003*(code+1))
            idx=spatial.indices();keep=(self.spot_slice_codes[idx[0]]==code)&(self.spot_slice_codes[idx[1]]==code)
            local_spatial=torch.sparse_coo_tensor(idx[:,keep],spatial.values()[keep],spatial.shape).coalesce()
            b=SliceShardBundle(code,name,spots,n,sampler,[],[],local_spatial)
            self._refresh(b,0,True);self.slice_bundles.append(b)
            self.r_by_slice[code]=torch.sqrt(degree[spots].clamp_min(1e-8)/self.train_spot_peak_degree[spots].clamp_min(1e-8)).mean()
        if saved is None:
            full=torch.from_numpy(np.stack(self._csr.nonzero())).long();full[1]+=self.num_spot
            self.Graph=normalized_adjacency(full,self.num_spot+self.num_peak)
        else:
            self.Graph=saved['spot_peak']
        values=spatial.values().clone()
        scales=torch.tensor([float(self.r_by_slice[i])*config.spatial_weight for i in range(self.num_slices)])
        values*=scales[self.spot_slice_codes[spatial.indices()[0]]]
        self.scaled_full_spatial_adjacency=torch.sparse_coo_tensor(spatial.indices(),values,spatial.shape).coalesce()

    def _refresh(self,b,epoch,adjacency):
        local=b.sampler.make_epoch_batches(epoch);lookup=torch.from_numpy(b.global_spots).long()
        b.batch_all=[[lookup[u],p+self.num_spot,n+self.num_spot] for u,p,n in local]
        if any(len(u)==0 for u,_,_ in b.batch_all):raise ValueError(f'slice {b.name} has empty positive shards; reduce slice_shards')
        if adjacency:
            b.adjacencies=[]
            for u,p,_ in b.batch_all:
                edges=torch.stack([u,p]);hashed=torch.unique(edges[0]*(self.num_spot+self.num_peak)+edges[1],sorted=True)
                unique=torch.stack([torch.div(hashed,self.num_spot+self.num_peak,rounding_mode='floor'),hashed%(self.num_spot+self.num_peak)])
                b.adjacencies.append(normalized_adjacency(unique,self.num_spot+self.num_peak))

    def maybe_refresh_slice_shards(self,epoch):
        if epoch==0:return
        c=self.config;reshard=bool(c.reshard_every and epoch%c.reshard_every==0)
        refresh=bool(c.negative_refresh and epoch%c.negative_refresh==0)
        if reshard or refresh:
            for b in self.slice_bundles:
                if reshard:b.sampler.reshard(epoch)
                self._refresh(b,epoch,reshard)

    def restore_slice_shards_for_resume(self,completed):
        if completed<=0:return
        last=completed-1;c=self.config
        re=last//c.reshard_every*c.reshard_every if c.reshard_every else 0
        ne=last//c.negative_refresh*c.negative_refresh if c.negative_refresh else 0
        for b in self.slice_bundles:
            if re:b.sampler.reshard(re)
            self._refresh(b,max(re,ne),bool(re))

    def ordered_slice_bundles(self,epoch):
        offset=epoch%len(self.slice_bundles)
        return self.slice_bundles[offset:]+self.slice_bundles[:offset]

    def state_dict(self):
        return {'protocol':'deterministic_seed_and_epoch_reconstruction_v1',
                'slices':{b.name:{'reshard_epoch':b.sampler.reshard_epoch,'negative_epoch':b.sampler.negative_epoch,
                                 'seed':b.sampler.base_seed} for b in self.slice_bundles}}
