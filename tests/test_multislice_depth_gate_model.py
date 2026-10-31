import unittest
import importlib
from _scientific_oracle import scientific_oracle
import torch

class DepthGateModelTests(unittest.TestCase):
    def test_single_model_parameters_and_refresh_cadence(self):
        from seareg.multislice.model import AdaptiveDepthGate
        model = AdaptiveDepthGate(5, 3, 4, torch.ones(5), torch.zeros(5,2), torch.zeros(5,dtype=torch.long))
        self.assertEqual(set(dict(model.named_parameters())), {'embedding_spot', 'embedding_peak'})
        with self.assertRaisesRegex(ValueError, '50'):
            model.refresh_if_due(1, None, None, None, every=25)

    def test_config_rejects_old_modes_and_refresh(self):
        from seareg.multislice.config import MultiSliceConfig
        self.assertEqual(MultiSliceConfig().correction_mode, 'adaptive_depth_gate')
        for mode in ('coherent_boundary', 'adaptive_tail_excess'):
            with self.assertRaises(ValueError): MultiSliceConfig(correction_mode=mode)
        with self.assertRaises(ValueError): MultiSliceConfig(gate_refresh_every=25)

    def test_refresh_and_state_roundtrip_preserve_cpu_rng(self):
        from seareg.multislice.model import AdaptiveDepthGate, infer_depth_gate_cpu
        xy=torch.tensor([[0.,0.], [1.,0.], [1.,1.], [0.,1.], [-1.,1.], [-1.,0.], [-1.,-1.], [0.,-1.], [1.,-1.]])
        rows=torch.tensor([0]*8+list(range(1,9))); cols=torch.tensor(list(range(1,9))+[0]*8)
        spatial=torch.sparse_coo_tensor(torch.stack([rows,cols]),torch.ones(16),(9,9)).coalesce()
        u=torch.arange(9).repeat_interleave(3); p=torch.arange(3).repeat(9)+9
        full=torch.sparse_coo_tensor(torch.stack([torch.cat([u,p]),torch.cat([p,u])]),torch.ones(54)/3,(12,12)).coalesce()
        codes=torch.zeros(9,dtype=torch.long)
        torch.manual_seed(314)
        model=AdaptiveDepthGate(9,3,4,torch.ones(9)*3,xy,codes)
        model.initialize_adaptive_gate(spatial,2,42,codes)
        model.refresh_selective_geometry(full,spatial)
        self.assertEqual(int(model.micro_refresh_count),1)
        self.assertFalse(model.refresh_if_due(0,full,spatial,codes))
        model.refresh_selective_geometry(full,spatial)
        self.assertEqual(int(model.micro_refresh_count),1)
        state={k:v.clone() for k,v in model.state_dict().items()}
        restored=AdaptiveDepthGate(9,3,4,torch.ones(9)*3,xy,codes)
        restored.load_state_dict(state)
        for k,v in state.items(): self.assertTrue(torch.equal(v,restored.state_dict()[k]),k)
        rng=torch.get_rng_state().clone()
        a=infer_depth_gate_cpu(model,full,None,spatial,True,1.,0.)
        b=infer_depth_gate_cpu(model,full,None,spatial,True,1.,0.)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for x,y in zip(a,b):
            if x is not None: self.assertTrue(torch.equal(x,y))
        for k,v in state.items(): self.assertTrue(torch.equal(v,model.state_dict()[k]),k)
        self.assertTrue(model.refresh_if_due(50,full,spatial,codes))
        self.assertFalse(model.refresh_if_due(50,full,spatial,codes))
        with self.assertRaisesRegex(ValueError,'stale|backward'): model.refresh_if_due(25,full,spatial,codes)
        self.assertEqual(int(model.micro_refresh_count),2)

    def test_restore_rejects_tampered_formula_axes_and_cached_allocation(self):
        from seareg.multislice.model import AdaptiveDepthGate
        xy=torch.tensor([[0.,0.], [1.,0.], [1.,1.], [0.,1.], [-1.,1.], [-1.,0.], [-1.,-1.], [0.,-1.], [1.,-1.]])
        rows=torch.tensor([0]*8+list(range(1,9))); cols=torch.tensor(list(range(1,9))+[0]*8)
        spatial=torch.sparse_coo_tensor(torch.stack([rows,cols]),torch.ones(16),(9,9)).coalesce()
        u=torch.arange(9).repeat_interleave(3); p=torch.arange(3).repeat(9)+9
        full=torch.sparse_coo_tensor(torch.stack([torch.cat([u,p]),torch.cat([p,u])]),torch.ones(54)/3,(12,12)).coalesce(); codes=torch.zeros(9,dtype=torch.long)
        model=AdaptiveDepthGate(9,3,4,torch.ones(9)*3,xy,codes); model.initialize_adaptive_gate(spatial,2,42,codes); model.refresh_selective_geometry(full,spatial)
        for name in ('geometry_formula_version','micro_formula_version','micro_epoch','geometry_coordinates','protection_slice_codes','micro_alpha','micro_extra','micro_quality'):
            state={k:v.clone() for k,v in model.state_dict().items()}; state[name].view(-1)[0]+=1
            fresh=AdaptiveDepthGate(9,3,4,torch.ones(9)*3,xy,codes)
            with self.assertRaises(ValueError,msg=name): fresh.load_state_dict(state)

    def test_native_checkpoint_carries_formula_identity_and_rejects_wrong_protocol(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from seareg.multislice.checkpoint import save_checkpoint, read_checkpoint
        from seareg.multislice.model import AdaptiveDepthGate
        from seareg.multislice.config import MultiSliceConfig
        cfg=MultiSliceConfig(epochs=2,device='cpu'); inputs=SimpleNamespace(fingerprint='f',metadata={'data_fingerprint':'d'})
        sampler=SimpleNamespace(state_dict=lambda: {'epoch':0}); execution={'numerical':{'device':'cpu'}}
        m=AdaptiveDepthGate(2,2,2,torch.ones(2),torch.zeros(2,2),torch.zeros(2,dtype=torch.long)); opt=torch.optim.Adam(m.parameters())
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.pt'; save_checkpoint(path,m,opt,inputs,cfg,0,0,sampler,[],execution)
            state=torch.load(path,weights_only=True)
            self.assertEqual(state.get('model_protocol'),'two_layer_sum_adaptive_depth_gate_v1')
            self.assertEqual(state.get('micro_formula_version'),1)
            state['model_protocol']='old'; torch.save(state,path)
            with self.assertRaisesRegex(ValueError,'protocol'): read_checkpoint(path,inputs,cfg,execution)
            m.gate_epoch.zero_(); m.micro_epoch.zero_(); m.micro_refresh_count.fill_(1)
            save_checkpoint(path,m,opt,inputs,cfg,0,0,sampler,[],execution)
            original=torch.load(path,weights_only=True)
            self.assertEqual(read_checkpoint(path,inputs,cfg,execution)['completed_epochs'],0)
            for name,value in [('completed_epochs',-1),('completed_epochs',0.0),('next_epoch',1),('optimizer_steps',1),('optimizer_steps',0.0)]:
                state=dict(original); state[name]=value; torch.save(state,path)
                with self.assertRaisesRegex(ValueError,'epoch|step|phase'): read_checkpoint(path,inputs,cfg,execution)
            state=dict(original); state['model_state']=dict(original['model_state'])
            state['model_state']['gate_epoch']=torch.tensor(50)
            state['model_state']['micro_epoch']=torch.tensor(50)
            state['model_state']['micro_refresh_count']=torch.tensor(2)
            torch.save(state,path)
            with self.assertRaisesRegex(ValueError,'epoch|phase'): read_checkpoint(path,inputs,cfg,execution)

    def test_active_state_forward_gradient_and_adam_match_frozen(self):
        import sys
        from pathlib import Path
        from seareg.multislice.model import AdaptiveDepthGate
        oracle = scientific_oracle(self, 'model')
        base = Path(oracle['source_dir'])
        sys.path.insert(0,str(base))
        try:
            Frozen = getattr(importlib.import_module(oracle['module']), oracle['class'])
            xy=torch.tensor([[0.,0.], [1.,0.], [1.,1.], [0.,1.], [-1.,1.], [-1.,0.], [-1.,-1.], [0.,-1.], [1.,-1.]])
            rows=torch.tensor([0]*8+list(range(1,9))); cols=torch.tensor(list(range(1,9))+[0]*8)
            spatial=torch.sparse_coo_tensor(torch.stack([rows,cols]),torch.ones(16),(9,9)).coalesce()
            u=torch.arange(9).repeat_interleave(3); p=torch.arange(3).repeat(9)+9
            full=torch.sparse_coo_tensor(torch.stack([torch.cat([u,p]),torch.cat([p,u])]),torch.ones(54)/3,(12,12)).coalesce(); codes=torch.zeros(9,dtype=torch.long)
            torch.manual_seed(100); a=AdaptiveDepthGate(9,3,4,torch.ones(9)*3,xy,codes)
            torch.manual_seed(100); b=Frozen(9,3,4,torch.ones(9)*3,xy,**oracle['constructor_kwargs'],rule='cross_boundary_risk_selective',gain_rule='coherence_selective',slice_codes=codes)
            for m in (a,b): m.initialize_adaptive_gate(spatial,2,42,codes); m.refresh_selective_geometry(full,spatial)
            for key in ('cached_first_layer_gate','geometry_boundary','geometry_valid_coverage','protection_effective_P','protection_risk','gain_local_delta','gain_displacement_norm','gain_effective_delta','micro_quality','micro_capacity','micro_extra','micro_alpha'):
                self.assertTrue(torch.equal(getattr(a,key),getattr(b,key)),key)
            oa=a.propagate(full,None,spatial,True,.3,0.); ob=b.propagate(full,None,spatial,True,.3,0.)
            for x,y in zip(oa,ob):
                if x is not None: self.assertTrue(torch.equal(x,y))
            la=sum(x.square().sum() for x in oa if x is not None); lb=sum(x.square().sum() for x in ob if x is not None)
            la.backward(); lb.backward()
            for (name,x),(other,y) in zip(a.named_parameters(),b.named_parameters()):
                self.assertEqual(name,other); self.assertTrue(torch.equal(x.grad,y.grad),name)
            op1=torch.optim.Adam(a.parameters(),lr=.001); op2=torch.optim.Adam(b.parameters(),lr=.001); op1.step(); op2.step()
            for (name,x),(_,y) in zip(a.named_parameters(),b.named_parameters()): self.assertTrue(torch.equal(x,y),name)
        finally:
            sys.path.remove(str(base))

    def test_resume_matches_continuous_across_refresh_boundaries(self):
        import copy
        from seareg.multislice.model import AdaptiveDepthGate
        xy=torch.tensor([[0.,0.], [1.,0.], [1.,1.], [0.,1.], [-1.,1.], [-1.,0.], [-1.,-1.], [0.,-1.], [1.,-1.]])
        rows=torch.tensor([0]*8+list(range(1,9))); cols=torch.tensor(list(range(1,9))+[0]*8)
        spatial=torch.sparse_coo_tensor(torch.stack([rows,cols]),torch.ones(16),(9,9)).coalesce()
        u=torch.arange(9).repeat_interleave(3); p=torch.arange(3).repeat(9)+9
        full=torch.sparse_coo_tensor(torch.stack([torch.cat([u,p]),torch.cat([p,u])]),torch.ones(54)/3,(12,12)).coalesce(); codes=torch.zeros(9,dtype=torch.long)
        def make(): return AdaptiveDepthGate(9,3,4,torch.ones(9)*3,xy,codes)
        def step(m,opt,epoch):
            opt.zero_grad(); out=m.propagate(full,None,spatial,True,.3,0.); scale=torch.rand(())
            sum((x*scale).square().sum() for x in out if x is not None).backward(); opt.step()
            if epoch%50==0: m.refresh_if_due(epoch,full,spatial,codes)
        torch.manual_seed(564); m=make(); m.initialize_adaptive_gate(spatial,2,42,codes); m.refresh_selective_geometry(full,spatial)
        opt=torch.optim.Adam(m.parameters(),lr=.001); snapshots={}; expected={}
        for epoch in range(1,102):
            step(m,opt,epoch)
            if epoch in (49,50,99,100): snapshots[epoch]=(copy.deepcopy(m.state_dict()),copy.deepcopy(opt.state_dict()),torch.get_rng_state().clone())
            if epoch in (50,51,100,101): expected[epoch]=(copy.deepcopy(m.state_dict()),copy.deepcopy(opt.state_dict()),torch.get_rng_state().clone())
        for boundary in (49,50,99,100):
            state,adam,rng=snapshots[boundary]; resumed=make(); resumed.load_state_dict(state)
            other=torch.optim.Adam(resumed.parameters(),lr=.001); other.load_state_dict(adam); torch.set_rng_state(rng)
            step(resumed,other,boundary+1); want,want_adam,want_rng=expected[boundary+1]
            for key,val in want.items(): self.assertTrue(torch.equal(val,resumed.state_dict()[key]),f'{boundary}: {key}')
            self.assertTrue(torch.equal(want_rng,torch.get_rng_state()))
            for key,entry in want_adam['state'].items():
                for name,val in entry.items(): self.assertTrue(torch.equal(val,other.state_dict()['state'][key][name]),f'{boundary}: {name}')


    def test_refresh_diagnostics_are_persisted_without_export_side_effects(self):
        import json, tempfile
        from pathlib import Path
        from seareg.multislice.model import AdaptiveDepthGate, infer_depth_gate_cpu
        xy=torch.tensor([[0.,0.], [1.,0.], [1.,1.], [0.,1.], [-1.,1.], [-1.,0.], [-1.,-1.], [0.,-1.], [1.,-1.]])
        rows=torch.tensor([0]*8+list(range(1,9))); cols=torch.tensor(list(range(1,9))+[0]*8)
        spatial=torch.sparse_coo_tensor(torch.stack([rows,cols]),torch.ones(16),(9,9)).coalesce()
        u=torch.arange(9).repeat_interleave(3); p=torch.arange(3).repeat(9)+9
        full=torch.sparse_coo_tensor(torch.stack([torch.cat([u,p]),torch.cat([p,u])]),torch.ones(54)/3,(12,12)).coalesce(); codes=torch.zeros(9,dtype=torch.long)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); m=AdaptiveDepthGate(9,3,4,torch.ones(9)*3,xy,codes,diagnostic_root=root)
            m.initialize_adaptive_gate(spatial,2,42,codes); m.refresh_selective_geometry(full,spatial)
            self.assertTrue((root/'epoch_0000.npz').exists())
            rows=[json.loads(s) for s in (root/'refresh.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),1); row=rows[0]['per_slice'][0]
            for key in ('valid','positive_quality','alpha','local_budget_displacement','allocated_budget_displacement'): self.assertIn(key,row)
            self.assertAlmostEqual(row['local_budget_displacement'],row['allocated_budget_displacement'],places=6)
            before=(root/'refresh.jsonl').read_bytes(); infer_depth_gate_cpu(m,full,None,spatial,True,1.,0.)
            self.assertEqual(before,(root/'refresh.jsonl').read_bytes())



    def test_nonzero_frozen_depth_gate_and_micro_bpr_gradient_clipped_adam(self):
        import sys
        from pathlib import Path
        from seareg.multislice.model import AdaptiveDepthGate
        from seareg.model import compute_loss
        oracle = scientific_oracle(self, 'model')
        base = Path(oracle['source_dir'])
        sys.path.insert(0,str(base))
        try:
            Frozen = getattr(importlib.import_module(oracle['module']), oracle['class'])
            xy=torch.tensor([[0.,0.], [1.,0.], [1.,1.], [0.,1.], [-1.,1.], [-1.,0.], [-1.,-1.], [0.,-1.], [1.,-1.]])
            row=torch.arange(9).repeat_interleave(9); col=torch.arange(9).repeat(9); keep=row!=col
            spatial=torch.sparse_coo_tensor(torch.stack([row[keep],col[keep]]),torch.ones(72)/8,(9,9)).coalesce()
            idx=torch.arange(9); full=torch.sparse_coo_tensor(torch.stack([torch.cat([idx,idx+9]),torch.cat([idx+9,idx])]),torch.ones(18),(18,18)).coalesce(); codes=torch.zeros(9,dtype=torch.long)
            torch.manual_seed(100); a=AdaptiveDepthGate(9,9,4,torch.ones(9),xy,codes)
            torch.manual_seed(100); b=Frozen(9,9,4,torch.ones(9),xy,**oracle['constructor_kwargs'],rule='cross_boundary_risk_selective',gain_rule='coherence_selective',slice_codes=codes)
            h=torch.tensor([[0.,1.,0.,0.],[1.,.02,.01,0.],[1.,.04,-.01,0.],[1.,-.02,.02,0.],[1.,.01,-.02,0.],[1.,-.03,.01,0.],[1.,.03,.02,0.],[1.,-.01,-.01,0.],[1.,.00,.03,0.]])
            g=torch.tensor([.4,.05,.15,.25,.35,.45,.55,.65,.75])
            for m in (a,b):
                m.initialize_adaptive_gate(spatial,2,42,codes)
                with torch.no_grad(): m.embedding_spot.fill_(1); m.embedding_spot[0].fill_(10); m.embedding_peak.copy_(h); m.cached_first_layer_gate.copy_(g)
                m.refresh_selective_geometry(full,spatial)
            self.assertGreater(float(a.gain_effective_delta.max()),0.)
            self.assertGreater(float(a.micro_extra.max()),0.)
            print(f'ACTIVE_FIXTURE gain_max={float(a.gain_effective_delta.max()):.10g} micro_max={float(a.micro_extra.max()):.10g} q_max={float(a.micro_quality.max()):.10g}',flush=True)
            for key in ('cached_first_layer_gate','geometry_boundary','geometry_valid_coverage','protection_effective_P','protection_risk','gain_local_delta','gain_displacement_norm','gain_effective_delta','micro_quality','micro_capacity','micro_extra','micro_alpha'):
                self.assertTrue(torch.equal(getattr(a,key),getattr(b,key)),key)
            oa=a.propagate(full,None,spatial,True,.3,0.); ob=b.propagate(full,None,spatial,True,.3,0.)
            for x,y in zip(oa,ob):
                if x is not None: self.assertTrue(torch.equal(x,y))
            users=torch.arange(9); pos=torch.arange(9); neg=torch.tensor([1,0,0,0,0,0,0,0,0])
            args=dict(spatial=spatial,use_spatial=True,r_spatial=.3,decay=.001,weighted_ego_l2=True)
            la,parts_a=compute_loss(a,users,pos,neg,full,**args); lb,parts_b=compute_loss(b,users,pos,neg,full,**args)
            self.assertTrue(torch.equal(la,lb)); self.assertTrue(torch.equal(parts_a['bpr'],parts_b['bpr'])); self.assertTrue(torch.equal(parts_a['reg'],parts_b['reg']))
            la.backward(); lb.backward()
            for (name,x),(other,y) in zip(a.named_parameters(),b.named_parameters()): self.assertEqual(name,other); self.assertTrue(torch.equal(x.grad,y.grad),name)
            na=torch.nn.utils.clip_grad_norm_(a.parameters(),.5); nb=torch.nn.utils.clip_grad_norm_(b.parameters(),.5); self.assertTrue(torch.equal(na,nb))
            self.assertGreater(float(na),.5)
            print('ACTIVE_OBJECTIVE loss={:.10g} bpr={:.10g} ego_l2={:.10g} preclip_norm={:.10g}'.format(float(la),float(parts_a['bpr']),float(parts_a['reg']),float(na)),flush=True)
            for (name,x),(_,y) in zip(a.named_parameters(),b.named_parameters()): self.assertTrue(torch.equal(x.grad,y.grad),name)
            op1=torch.optim.Adam(a.parameters(),lr=.001); op2=torch.optim.Adam(b.parameters(),lr=.001); op1.step(); op2.step()
            for (name,x),(_,y) in zip(a.named_parameters(),b.named_parameters()): self.assertTrue(torch.equal(x,y),name)
        finally:
            sys.path.remove(str(base))
