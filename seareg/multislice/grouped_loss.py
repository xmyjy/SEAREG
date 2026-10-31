"""Same LightGCN/BPR operators with one linear ego regularizer per update group."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from ..model import _edge_sums, batch_edge_weights
from ..scoring import fused_sampled_scores


def bpr_only(model,users,positive,negative,adjacency,spatial,use_spatial,r_spatial,plan,group_size):
    spots,peaks,_=model.propagate(adjacency,None,spatial,use_spatial,r_spatial,0.)
    weights=batch_edge_weights(users,model.num_spot,dtype=spots.dtype)
    bpr=spots.new_zeros(())
    if plan['effective_backend']=='cuda_fused':
        if torch.are_deterministic_algorithms_enabled():
            raise ValueError('CUDA fused scoring requires non-deterministic execution')
        a,b=fused_sampled_scores(spots,peaks,users,positive,negative,validate=False)
        return (-F.logsigmoid(a-b)*weights).sum()
    for start in range(0,len(users),plan['chunk_triplets']):
        part=slice(start,start+plan['chunk_triplets'])
        args=(spots,peaks,model.embedding_spot,model.embedding_peak,
              users[part],positive[part],negative[part],weights[part],group_size,False)
        value=(checkpoint(_edge_sums,*args,use_reentrant=False,preserve_rng_state=False,
                          determinism_check='none') if plan['checkpoint'] and torch.is_grad_enabled()
               else _edge_sums(*args))
        bpr=bpr+value[0]
    return bpr


def group_regularization(model,coefficients):
    """Average coefficients; keep per-slice diagnostic values detached."""
    spot_sq=model.embedding_spot.square().sum(1)
    peak_sq=model.embedding_peak.square().sum(1)
    bar_spot=torch.zeros_like(coefficients[0].spot)
    bar_peak=torch.zeros_like(coefficients[0].peak)
    scale=1./len(coefficients)
    for value in coefficients:
        bar_spot.add_(value.spot,alpha=scale)
        bar_peak.add_(value.peak,alpha=scale)
    reg=.5*((bar_spot*spot_sq).sum()+(bar_peak*peak_sq).sum())
    diagnostics=[.5*((value.spot*spot_sq.detach()).sum()+(value.peak*peak_sq.detach()).sum())
                 for value in coefficients]
    return reg,diagnostics
