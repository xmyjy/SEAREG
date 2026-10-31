"""Compare the native sampler with the archived frozen B1 implementation."""
import ast
import contextlib
import importlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from seareg.multislice import MultiSliceConfig, prepare_multislice_training
from seareg.multislice.sampling import MultiSliceSampler
from seareg.multislice.balanced import selected_shard_indices

BASE=Path('/mnt/disk1/xmy/seareg/workspaces/b1_multislice_20261025_v1/runs/gate_cs_local_microgain_7118_r100n5_lab66_20261031_v1/models/cs_matched03')

class FrozenSamplerTests(unittest.TestCase):
    def test_frozen_sampler_four_slices_key_epochs(self):
        if not BASE.exists(): self.skipTest('frozen experiment source unavailable')
        sys.path.insert(0,str(BASE/'source/b1_loader'))
        try:
            frozen=importlib.import_module('slice_local_data')
            source=(BASE/'code/balanced_loop.py').read_text()
            node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='selected_shard_indices')
            env={'np':np}; exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen selector>','exec'),env)
            ref_select=env['selected_shard_indices']
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                x=np.zeros((20,12),dtype=np.float32)
                for i in range(20): x[i,[i%12,(i+1)%12,(i+4)%12,(i+7)%12]]=1
                coords=np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1]]*4,dtype=float)
                labels=sum(([f's{k}']*5 for k in range(4)),[])
                a=ad.AnnData(sp.csr_matrix(x),obs=pd.DataFrame({'sample_id':labels},index=[f'spot{i}' for i in range(20)]),var=pd.DataFrame(index=[f'chr1:{i*100}-{i*100+50}' for i in range(12)])); a.obsm['spatial']=coords
                shards={'s0':2,'s1':2,'s2':3,'s3':3}
                cfg=MultiSliceConfig(slice_shards=shards,epochs=1,hidden_dim=4,updates_per_epoch=20,gate_null_permutations=2,feature_min_spots=1,feature_min_fraction=0,feature_max_fraction=1,device='cpu')
                inputs=prepare_multislice_training(a,cfg,Path(tmp)/'inputs'); new=MultiSliceSampler(inputs,cfg)
                old=frozen.SliceLocalDataLoader.__new__(frozen.SliceLocalDataLoader)
                old.args=SimpleNamespace(tr=1.,pn_r=5,seed=cfg.seed,num_workers=1,max_pos_per_spot=0,negative_peak_scope='global',reshard_every=100,neg_resample_every=5,slice_order='rotating',r_spatial=.3)
                old._csr=new._csr; old.num_spot=new.num_spot; old.num_peak=new.num_peak; old.num_slices=new.num_slices
                old.spot_slice_codes=new.spot_slice_codes; old.slice_names=new.slice_names; old.slice_shard_counts=shards
                old.data_spatial_graph=new.data_spatial_graph; old.train_spot_peak_degree=new.train_spot_peak_degree
                old.slice_bundles=[]; old._build_slice_bundles(0,True); old._build_full_adjacency_from_slice_edges(); old._build_slice_spatial_scales()
                def eq_sparse(x,y,where):
                    x=x.coalesce(); y=y.coalesce(); self.assertTrue(torch.equal(x.indices(),y.indices()),where+' indices'); self.assertTrue(torch.equal(x.values(),y.values()),where+' values')
                eq_sparse(new.Graph,old.Graph,'full adjacency')
                eq_sparse(new.scaled_full_spatial_adjacency,old._scaled_full_spatial_adjacency(),'scaled spatial')
                for code in range(4): self.assertTrue(torch.equal(new.r_by_slice[code],old.r_by_slice[code]),f'r_by_slice {code}')
                for epoch in (0,5,50,100):
                    new.maybe_refresh_slice_shards(epoch); old.maybe_refresh_slice_shards(epoch)
                    self.assertEqual([b.code for b in new.ordered_slice_bundles(epoch)],[b.code for b in old.ordered_slice_bundles(epoch)])
                    for b,ref in zip(new.slice_bundles,old.slice_bundles):
                        self.assertEqual(selected_shard_indices(b,epoch,20,seed=cfg.seed),ref_select(ref,epoch,20,strategy='independent_seeded_shuffle',seed=cfg.seed))
                        for ix,(actual,expected) in enumerate(zip(b.batch_all,ref.batch_all)):
                            for name,x,y in zip(('users','positive','negative'),actual,expected): self.assertTrue(torch.equal(x,y),f'{epoch}/{b.name}/{ix}/{name}')
                            eq_sparse(b.adjacencies[ix],ref.adjacencies[ix],f'{epoch}/{b.name}/{ix}')
        finally:
            sys.path.remove(str(BASE/'source/b1_loader'))
