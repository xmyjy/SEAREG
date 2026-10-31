"""Detached cross-boundary risk protection for adaptive_depth_gate."""
import torch
import torch.nn.functional as F
EPS = 1e-8

@torch.no_grad()
def compute_protection(h, spatial, g, geometry, codes):
    h=h.detach().float().cpu(); a=spatial.detach().cpu().coalesce(); g=g.detach().float().cpu(); codes=codes.detach().long().cpu()
    geo={k:v.detach().cpu() for k,v in geometry.items()}; P=geo['boundary'].float(); valid=geo['valid_coverage'].bool()
    n,dim=h.shape; row,col=a.indices(); w=a.values().float()
    if codes.shape!=(n,) or not torch.equal(row*n+col,geo['edge_keys']): raise ValueError('protection graph/axis mismatch')
    if not all(torch.isfinite(t).all() for t in [h,g,P,w]): raise ValueError('nonfinite protection input')
    if ((codes[row] != codes[col]) & (w > 0)).any(): raise ValueError('cross-slice spatial edge')
    sector=geo['edge_sectors']; active=geo['edge_active']; bucket=row*8+sector; weights=w*active
    mass=torch.zeros(n*8).index_add_(0,bucket,weights).reshape(n,8)
    def sector_mean(v):
        return torch.zeros(n*8,dim).index_add_(0,bucket,v[col]*weights[:,None]).reshape(n,8,dim)/mass.clamp_min(EPS)[:,:,None]
    z=F.normalize(h,dim=1,eps=EPS); means=sector_mean(z)
    distance=torch.linalg.vector_norm(z[row]-z[col],dim=1)
    distances=torch.zeros(n*8).index_add_(0,bucket,distance*weights).reshape(n,8)/mass.clamp_min(EPS)
    scatter=torch.linalg.vector_norm(z[col]-means[row,sector],dim=1)
    dispersion=torch.zeros(n*8).index_add_(0,bucket,scatter*weights).reshape(n,8)/mass.clamp_min(EPS)
    donor=1-g; clean=torch.sparse.mm(a,h*donor[:,None])/torch.sparse.mm(a,donor[:,None]).clamp_min(EPS)
    correction=clean-h; displacement=torch.linalg.vector_norm(correction,dim=1)
    halves=(torch.arange(8)[:,None]+torch.arange(4)[None,:]).remainder(8)
    halfmean=means[:,halves].mean(2); halfd=distances[:,halves].mean(2)
    halfsigma=(dispersion[:,halves]+torch.linalg.vector_norm(means[:,halves]-halfmean[:,:,None,:],dim=3)).mean(2)
    contrast=(halfd.roll(-4,1)-halfd).clamp_min(0)/(halfd.roll(-4,1)+halfd+halfsigma+halfsigma.roll(-4,1)).clamp_min(EPS)
    rawmean=sector_mean(h)[:,halves].mean(2); normals=rawmean.roll(-4,1)-rawmean
    vn=torch.linalg.vector_norm(correction,dim=1); nn=torch.linalg.vector_norm(normals,dim=2)
    nonzero=(vn[:,None]>EPS)&(nn>EPS); denominator=torch.where(nonzero,vn[:,None]*nn,torch.ones_like(nn))
    cosine=((correction[:,None,:]*normals).sum(2)/denominator).clamp(-1,1)
    cosine=torch.where(nonzero,cosine,torch.zeros_like(cosine)); tied=contrast==contrast.max(1,keepdim=True).values
    risk=torch.where(valid,(cosine.clamp_min(0)*tied).sum(1)/tied.sum(1).clamp_min(1),torch.zeros_like(g))
    effective=torch.where(valid,P*risk,torch.zeros_like(P))
    if not all(torch.isfinite(t).all() for t in [effective,risk,displacement]): raise FloatingPointError('nonfinite protection state')
    if (effective<0).any() or (effective>P+1e-7).any(): raise ValueError('effective protection outside original bounds')
    return dict(effective_P=effective,risk=risk,displacement_norm=displacement)
