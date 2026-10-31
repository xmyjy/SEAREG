"""Native CPU recovery checks over real sharded multislice inputs."""
import contextlib
import dataclasses
import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import anndata as ad
from seareg.multislice import MultiSliceConfig, prepare_multislice_training
from seareg.multislice.train import _train
from seareg.multislice.jobs import validated_execution

class NativeResumeTests(unittest.TestCase):
    def test_native_rng_sampling_and_adam_resume_across_refresh_boundaries(self):
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); coords=np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1]]*2,dtype=float)
            x=np.zeros((10,8),dtype=np.float32)
            for i in range(10): x[i,[i%8,(i+1)%8,(i+3)%8]]=1
            a=ad.AnnData(sp.csr_matrix(x),obs=pd.DataFrame({'sample_id':['s0']*5+['s1']*5},index=[f'spot{i}' for i in range(10)]),var=pd.DataFrame(index=[f'chr1:{i*100}-{i*100+50}' for i in range(8)])); a.obsm['spatial']=coords
            cfg=MultiSliceConfig(slice_shards={'s0':2,'s1':2},epochs=101,hidden_dim=4,updates_per_epoch=2,gate_null_permutations=2,feature_min_spots=1,feature_min_fraction=0,feature_max_fraction=1,device='cpu',resident_cache=False)
            inputs=prepare_multislice_training(a,cfg,root/'inputs'); runtime=None
            def train(config,path,resume=None):
                with contextlib.redirect_stdout(io.StringIO()): _train(inputs,config,path,resume_from=resume,execution=runtime)
            def same(x,y,path=''):
                if torch.is_tensor(x): self.assertTrue(torch.equal(x,y),path)
                elif isinstance(x,np.ndarray): np.testing.assert_array_equal(x,y,err_msg=path)
                elif isinstance(x,dict):
                    self.assertEqual(x.keys(),y.keys(),path)
                    for k in x: same(x[k],y[k],f'{path}/{k}')
                elif isinstance(x,(list,tuple)):
                    self.assertEqual(len(x),len(y),path)
                    for i,(v,w) in enumerate(zip(x,y)): same(v,w,f'{path}/{i}')
                else: self.assertEqual(x,y,path)
            train(cfg,root/'continuous')
            run=root/'continuous'
            self.assertEqual([json.loads(line)['epoch'] for line in (run/'gate_states/refresh.jsonl').read_text().splitlines()],[0,50,100])
            for epoch in (0,50,100): self.assertTrue((run/'gate_states'/f'epoch_{epoch:04d}.npz').is_file())
            table=pd.read_csv(run/'per_slice_epoch_diagnostics.tsv',sep='\t'); self.assertEqual(len(table),202)
            manifest=json.loads((run/'method_manifest.json').read_text()); self.assertEqual(manifest['micro_formula_version'],1)
            self.assertEqual(manifest['model_protocol'],'two_layer_sum_adaptive_depth_gate_v1')
            expected=torch.load(root/'continuous/checkpoints/latest.pt',map_location='cpu',weights_only=True)
            for boundary in (49,50,99,100):
                prefix=root/f'prefix{boundary}'; resumed=root/f'resumed{boundary}'
                train(dataclasses.replace(cfg,epochs=boundary),prefix)
                train(cfg,resumed,prefix/'checkpoints/latest.pt')
                actual=torch.load(resumed/'checkpoints/latest.pt',map_location='cpu',weights_only=True)
                for key in ('model_state','optimizer_state','rng_state','sampler_state','optimizer_steps','completed_epochs'):
                    same(expected[key],actual[key],f'{boundary}/{key}')
                for key in ('python','numpy','torch'):
                    self.assertIn(key,actual['rng_state'])
