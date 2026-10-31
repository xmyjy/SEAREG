"""Single cs_matched03 multislice architecture."""
import time, os, json
from pathlib import Path
import numpy as np
import torch
from .gate import GateBackbone, infer_tailgate_cpu
from .geometry import compute_selective_geometry, prepare_selective_correction, correction_cache_key
from .protection import compute_protection
from .coherence import compute_cs_gain
from .microgain import local_quality, allocate_matched03, EPS

GEOMETRY_NAMES = ('boundary', 'valid_coverage', 'edge_keys', 'edge_sectors', 'edge_active')
GAIN_NAMES = ('local_delta', 'capacity', 'displacement_norm', 'eligible', 'consensus', 'effective_delta')
MICRO_NAMES = ('quality', 'outlier', 'alignment', 'valid', 'center_distance', 'background_residual', 'capacity', 'extra', 'local03', 'matched03')

class MicroGainTailgate(GateBackbone):
    def __init__(self, n_spots, n_peaks, dim, degree, coordinates, slice_codes, *, diagnostic_root=None):
        super().__init__(n_spots, n_peaks, dim, degree)
        xy=torch.as_tensor(coordinates,dtype=torch.float32).clone(); codes=torch.as_tensor(slice_codes,dtype=torch.long).clone()
        if xy.shape!=(n_spots,2) or not torch.isfinite(xy).all() or codes.shape!=(n_spots,): raise ValueError('invalid coordinate/slice axis')
        unique=torch.unique(codes,sorted=True)
        if not torch.equal(unique,torch.arange(len(unique))): raise ValueError('slice codes must be contiguous')
        self.first_layer_gate_mode='cs_matched03'; self._diagnostic_root=diagnostic_root
        self.register_buffer('geometry_coordinates',xy); self.register_buffer('protection_slice_codes',codes)
        self.register_buffer('geometry_ready',torch.tensor(False)); self.register_buffer('geometry_formula_version',torch.tensor(2))
        self.register_buffer('micro_formula_version',torch.tensor(1)); self.register_buffer('micro_epoch',torch.tensor(-1)); self.register_buffer('micro_refresh_count',torch.tensor(0))
        for name in GEOMETRY_NAMES:
            dtype=torch.bool if name in ('valid_coverage','edge_active') else torch.long if name in ('edge_keys','edge_sectors') else torch.float32
            self.register_buffer('geometry_'+name,torch.empty(0,dtype=dtype) if name.startswith('edge_') else torch.zeros(n_spots,dtype=dtype))
        for name in ('effective_P','risk','displacement_norm'): self.register_buffer('protection_'+name,torch.zeros(n_spots))
        for name in GAIN_NAMES: self.register_buffer('gain_'+name,torch.zeros(n_spots,dtype=torch.bool if name=='eligible' else torch.float32))
        for name in MICRO_NAMES: self.register_buffer('micro_'+name,torch.zeros(n_spots,dtype=torch.bool if name=='valid' else torch.float32))
        self.register_buffer('micro_alpha',torch.zeros(len(unique))); self._geometry_spatial_cache={}; self.geometry_update_seconds=0.

    def geometry_state(self):
        state={name:getattr(self,'geometry_'+name) for name in GEOMETRY_NAMES}; state['boundary']=self.protection_effective_P
        return state

    @torch.no_grad()
    def refresh_selective_geometry(self,full,spatial):
        if int(self.micro_epoch)==int(self.gate_epoch) and int(self.micro_epoch)>=0: return
        started=time.perf_counter(); h=torch.sparse.mm(full,torch.cat([self.embedding_spot,self.embedding_peak],0))[:self.num_spot]
        geo=compute_selective_geometry(h,self.geometry_coordinates,spatial,self.cached_first_layer_gate)
        p=compute_protection(h,spatial,self.cached_first_layer_gate,geo,self.protection_slice_codes)
        gain=compute_cs_gain(h,spatial,self.cached_first_layer_gate,p['effective_P'],geo['valid_coverage'],self.protection_slice_codes)
        hc=h.detach().float().cpu(); a=spatial.detach().cpu().coalesce(); g=self.cached_first_layer_gate.detach().float().cpu()
        den=torch.sparse.mm(a,(1-g)[:,None]); clean=torch.sparse.mm(a,hc*(1-g)[:,None])/den.clamp_min(EPS)
        valid=geo['valid_coverage'].cpu() & gain['eligible'] & (den[:,0]>EPS)
        evidence=local_quality(hc,self.geometry_coordinates,a,clean,valid,self.protection_slice_codes)
        cap=(1-p['effective_P']-(g*(1-p['effective_P'])+gain['effective_delta'])).clamp_min(0)
        allocation=allocate_matched03(cap,evidence['quality'],gain['displacement_norm'],evidence['valid'],self.protection_slice_codes)
        micro=dict(evidence,capacity=cap,extra=allocation['matched03'],**allocation)
        for prefix,names,values in [('geometry_',GEOMETRY_NAMES,geo),('protection_',('effective_P','risk','displacement_norm'),p),('gain_',GAIN_NAMES,gain),('micro_',MICRO_NAMES+('alpha',),micro)]:
            for name in names: self._buffers[prefix+name]=values[name].to(self.embedding_spot.device)
        self.micro_epoch.copy_(self.gate_epoch); self.micro_refresh_count.add_(1); self.geometry_ready.fill_(True)
        self._geometry_spatial_cache.clear(); self.geometry_update_seconds=time.perf_counter()-started; self._validate_cs_state(); self._write_diagnostics()

    @torch.no_grad()
    def refresh_if_due(self,completed_epoch,full,spatial,codes,every=50):
        if every!=50: raise ValueError('cs_matched03 refresh cadence must be 50')
        if not torch.equal(torch.as_tensor(codes).cpu(),self.protection_slice_codes.cpu()): raise ValueError('slice axis mismatch')
        if super().refresh_if_due(completed_epoch,full,spatial,codes,every):
            self.refresh_selective_geometry(full,spatial); return True
        return False

    def _validate_cs_state(self):
        if int(self.geometry_formula_version)!=2 or int(self.micro_formula_version)!=1 or int(self.micro_epoch)<0 or int(self.micro_epoch)%50 or int(self.micro_epoch)!=int(self.gate_epoch) or int(self.micro_refresh_count)!=int(self.micro_epoch)//50+1: raise ValueError('micro formula/refresh mismatch')
        expected_q=torch.where(self.micro_valid,self.micro_outlier*self.micro_alignment,torch.zeros_like(self.micro_quality))
        if not torch.allclose(expected_q,self.micro_quality,atol=2e-6,rtol=2e-5): raise ValueError('micro quality mismatch')
        g=self.cached_first_layer_gate; p=self.protection_effective_P
        capacity=(1-p-(g*(1-p)+self.gain_effective_delta)).clamp_min(0)
        if not torch.allclose(capacity,self.micro_capacity,atol=2e-6,rtol=2e-5): raise ValueError('micro capacity mismatch')
        expected=allocate_matched03(self.micro_capacity,self.micro_quality,self.gain_displacement_norm,self.micro_valid,self.protection_slice_codes)
        for name in ('local03','matched03','alpha'):
            if not torch.equal(expected[name].cpu(),getattr(self,'micro_'+name).cpu()): raise ValueError('micro allocation mismatch: '+name)
        if not torch.equal(expected['matched03'].cpu(),self.micro_extra.cpu()): raise ValueError('micro extra mismatch')
        if (g*(1-p)+self.gain_effective_delta+self.micro_extra>1-p+1e-7).any(): raise ValueError('receiver exceeds protection ceiling')
        if not torch.equal(self.gain_effective_delta,self.gain_local_delta*self.gain_consensus): raise ValueError('coherence gain mismatch')

    def _write_diagnostics(self):
        if self._diagnostic_root is None: return
        root=Path(self._diagnostic_root); root.mkdir(parents=True,exist_ok=True)
        final=root / f'epoch_{int(self.micro_epoch):04d}.npz'; temporary=final.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,**{name:value.detach().cpu().numpy() for name,value in self.named_buffers()})
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,final)
        rows=[]
        for code in range(len(self.micro_alpha)):
            selected=self.protection_slice_codes==code
            norm=self.gain_displacement_norm[selected].double()
            local=(self.micro_local03[selected].double()*norm).sum()
            matched=(self.micro_matched03[selected].double()*norm).sum()
            rows.append(dict(slice_code=code,spots=int(selected.sum()),geometry_covered=int(self.geometry_valid_coverage[selected].sum()),
                valid=int(self.micro_valid[selected].sum()),positive_quality=int((self.micro_quality[selected]>0).sum()),
                positive_extra=int((self.micro_extra[selected]>0).sum()),alpha=float(self.micro_alpha[code]),
                local03_displacement=float(local),matched03_displacement=float(matched)))
        record=dict(epoch=int(self.micro_epoch),micro_formula_version=int(self.micro_formula_version),
                    geometry_formula_version=int(self.geometry_formula_version),refresh_count=int(self.micro_refresh_count),per_slice=rows)
        with (root/'refresh.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
    def _correct_first_layer_peak_message(self,p1,spatial):
        if not bool(self.geometry_ready) or int(self.micro_epoch)!=int(self.gate_epoch): raise RuntimeError('CS geometry is stale')
        state=self.geometry_state(); key=correction_cache_key(spatial,self.cached_first_layer_gate,state)
        prep=self._geometry_spatial_cache.get(key)
        if prep is None:
            prep=prepare_selective_correction(spatial,self.cached_first_layer_gate,state); self._geometry_spatial_cache[key]=prep
        g=self.cached_first_layer_gate.detach().to(p1)
        clean=torch.sparse.mm(spatial,p1*(1-g)[:,None])/prep['denominator'].to(p1).clamp_min(EPS)
        valid=self.geometry_valid_coverage & (prep['denominator'][:,0]>EPS)
        delta=torch.where(valid,self.gain_effective_delta,torch.zeros_like(self.gain_effective_delta)).to(p1)
        first=p1+(prep['receiver'].to(p1)+delta)[:,None]*(clean-p1)
        extra=torch.where(valid & self.micro_valid,self.micro_extra,torch.zeros_like(self.micro_extra)).to(p1)
        micro_clean=torch.sparse.mm(spatial,p1*(1-g)[:,None])/prep['denominator'].to(p1).clamp_min(EPS)
        return torch.where((extra!=0)[:,None],first+extra[:,None]*(micro_clean-p1),first)

    def load_state_dict(self,state_dict,strict=True,assign=False):
        for name in ('geometry_coordinates','protection_slice_codes'):
            if name not in state_dict or not torch.equal(state_dict[name].cpu(),getattr(self,name).cpu()): raise ValueError('checkpoint coordinate/slice axis mismatch')
        for name in GEOMETRY_NAMES: self._buffers['geometry_'+name]=torch.empty_like(state_dict['geometry_'+name],device=self.embedding_spot.device)
        result=super().load_state_dict(state_dict,strict=strict,assign=assign)
        if not all(torch.isfinite(b).all() for b in self.buffers()): raise ValueError('nonfinite restored state')
        self._geometry_spatial_cache.clear(); self._validate_cs_state(); return result

    def __getstate__(self):
        state=super().__getstate__(); state['_geometry_spatial_cache']={}; state['_diagnostic_root']=None; return state


def geometry_summary(model):
    g=model.cached_first_layer_gate; p=model.protection_effective_P
    return dict(geometry_formula_version=2, geometry_epoch=int(model.gate_epoch), valid_coverage_fraction=float(model.geometry_valid_coverage.float().mean()), boundary_mean=float(model.geometry_boundary.mean()), receiver_gate_mean=float((g*(1-p)+model.gain_effective_delta+model.micro_extra).mean()), micro_formula_version=int(model.micro_formula_version), micro_epoch=int(model.micro_epoch), micro_alpha=model.micro_alpha.detach().cpu().tolist(), micro_refresh_count=int(model.micro_refresh_count), geometry_update_seconds=model.geometry_update_seconds)
