"""Detached coherence selective gain for cs_matched03."""
import torch
EPS=1e-8
@torch.no_grad()
def gain_from_messages(h,clean,uniform,g,p,valid,codes):
 if h.shape!=clean.shape or h.shape!=uniform.shape or h.ndim!=2:raise ValueError('gain message shape mismatch')
 n=len(h)
 if any(v.shape!=(n,) for v in [g,p,valid,codes]):raise ValueError('gain point axis mismatch')
 if not all(torch.isfinite(v).all() for v in [h,clean,uniform,g,p]):raise ValueError('nonfinite gain inputs')
 if (g<0).any() or (g>1).any() or (p<0).any() or (p>1).any():raise ValueError('gain probability outside bounds')
 norm=(clean-h).norm(dim=1);shift=(uniform-clean).norm(dim=1);base=g*(1-p);ceiling=1-p
 eligible=valid.bool()&(norm>EPS);capacity=torch.where(eligible,torch.minimum(base,ceiling-base).clamp_min(0),torch.zeros_like(base))
 local=torch.minimum(capacity,base*shift/norm.clamp_min(EPS));local=torch.where(eligible,local,torch.zeros_like(local))
 unique=torch.unique(codes,sorted=True)
 if not torch.equal(unique,torch.arange(len(unique),device=codes.device)):raise ValueError('slice codes must be contiguous')
 alpha=torch.zeros(len(unique),dtype=h.dtype,device=h.device);uniform_delta=torch.zeros_like(base);rows=[]
 for code in unique.tolist():
  mask=codes==code;target=(local[mask]*norm[mask]).sum();den=(capacity[mask]*norm[mask]).sum()
  coefficient=(target/den).clamp(0,1) if den>EPS else den.new_zeros(())
  alpha[code]=coefficient;uniform_delta[mask]=coefficient*capacity[mask];matched=(uniform_delta[mask]*norm[mask]).sum()
  if not torch.isclose(target,matched,atol=2e-6,rtol=2e-5):raise ValueError('gain displacement matching failed')
  rows.append(dict(slice_code=code,n=int(mask.sum()),eligible=int(eligible[mask].sum()),positive=int((local[mask]>0).sum()),base_receiver_mean=float(base[mask].mean()),local_delta_mean=float(local[mask].mean()),capacity_mean=float(capacity[mask].mean()),alpha=float(coefficient),local_displacement_sum=float(target),slice_displacement_sum=float(matched),donor_counterfactual_displacement_sum=float((base[mask]*shift[mask]).sum()),capped=int(((local>=capacity)&(capacity>0)&mask).sum())))
 return dict(local_delta=local,slice_delta=uniform_delta,capacity=capacity,displacement_norm=norm,donor_shift_norm=shift,alpha=alpha,eligible=eligible,diagnostics=rows)

@torch.no_grad()
def compute_cs_gain(h, spatial, g, p, valid, codes):
    h=h.detach().float().cpu(); s=spatial.detach().cpu().coalesce(); g=g.detach().float().cpu(); p=p.detach().float().cpu(); valid=valid.detach().bool().cpu(); codes=codes.detach().long().cpu()
    den=torch.sparse.mm(s,(1-g)[:,None]); clean=torch.sparse.mm(s,h*(1-g)[:,None])/den.clamp_min(EPS)
    mass=torch.sparse.mm(s,torch.ones(len(h),1)); uniform=torch.sparse.mm(s,h)/mass.clamp_min(EPS)
    gain=gain_from_messages(h,clean,uniform,g,p,valid&(den[:,0]>EPS)&(mass[:,0]>EPS),codes)
    row,col=s.indices(); denominator=torch.sparse.mm(s,(1-g)[:,None])[:,0]
    clean=torch.sparse.mm(s,h*(1-g)[:,None])/denominator.clamp_min(EPS)[:,None]; v=clean-h; norm=v.norm(dim=1)
    diff=h[col]-h[row]; edge_norm=diff.norm(dim=1); unit=diff/edge_norm.clamp_min(EPS)[:,None]
    unit=torch.where((edge_norm>EPS)[:,None],unit,torch.zeros_like(unit))
    weight=s.values()*(1-g[col])/denominator[row].clamp_min(EPS)
    mean=torch.zeros_like(h).index_add_(0,row,unit*weight[:,None]); direction=v/norm.clamp_min(EPS)[:,None]
    consensus=(mean*direction).sum(1).clamp(0,1)
    eligible=gain['eligible']&(denominator>EPS)&(norm>EPS)
    consensus=torch.where(eligible,consensus,torch.zeros_like(consensus))
    gain['consensus']=consensus
    gain['effective_delta']=torch.where(eligible,gain['local_delta'],torch.zeros_like(gain['local_delta']))*consensus
    return gain
