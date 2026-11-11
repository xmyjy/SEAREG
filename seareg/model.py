"""SEAREG LightGCN model and sampled BPR objectives."""

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .loss import (
    prepare_ego_l2_coefficients,
    weighted_ego_regularization,
    _weighted_ego_regularization,
    execution_plan,
)
from .scoring import fused_sampled_scores


class LightGCN(nn.Module):
    def __init__(self, n_spots, n_peaks, n_dna, dim, n_layers=2):
        super().__init__()
        if n_layers != 2:
            raise ValueError("this model explicitly supports exactly two layers")
        if min(n_spots, n_peaks, dim) < 1 or (n_dna is not None and n_dna < 0):
            raise ValueError("invalid model dimensions")
        self.num_spot, self.num_peak, self.num_kmer = n_spots, n_peaks, n_dna
        self.embedding_spot = nn.Parameter(torch.zeros(n_spots, dim))
        self.embedding_peak = nn.Parameter(torch.zeros(n_peaks, dim))
        self.embedding_kmer = nn.Parameter(torch.zeros(n_dna, dim)) if n_dna else None
        for parameter in (self.embedding_spot, self.embedding_peak, self.embedding_kmer):
            if parameter is not None:
                nn.init.xavier_uniform_(parameter)

    def propagate(self, A, dna=None, spatial=None, use_spatial=True, r_spatial=0.0, r_dna=0.0):
        u0, p0, k0 = self.embedding_spot, self.embedding_peak, self.embedding_kmer
        x1 = torch.sparse.mm(A, torch.cat([u0, p0], dim=0))
        u1, p1 = torch.split(x1, [self.num_spot, self.num_peak], dim=0)
        if use_spatial and spatial is not None:
            u1 = u1 + r_spatial * torch.sparse.mm(spatial, u0)
        if k0 is not None and dna is not None:
            p1d, k1 = torch.split(
                torch.sparse.mm(dna, torch.cat([p0, k0], dim=0)), [self.num_peak, len(k0)], dim=0
            )
            p1 = p1 + r_dna * p1d
        x2 = torch.sparse.mm(A, torch.cat([u1, p1], dim=0))
        u2, p2 = torch.split(x2, [self.num_spot, self.num_peak], dim=0)
        if use_spatial and spatial is not None:
            u2 = u2 + r_spatial * torch.sparse.mm(spatial, u1)
        kout = None
        if k0 is not None and dna is not None:
            p2d, k2 = torch.split(
                torch.sparse.mm(dna, torch.cat([p1, k1], dim=0)), [self.num_peak, len(k0)], dim=0
            )
            p2 = p2 + r_dna * p2d
            kout = k1 + k2
        return u1 + u2, p1 + p2, kout


def _edge_sums(u, p, u0, p0, users, pos, neg, weights, group=1, include_reg=True):
    # Preserve copy_1's shared gather and gradient accumulation order.
    if group > 1:
        # The sampler expands every unique (user, positive) pair pn_r times as
        # one contiguous block. Gather that stable pair once and broadcast it
        # against the group's negatives. The forward objective is unchanged;
        # backward reductions can differ at floating-point roundoff scale.
        grouped_users = users.reshape(-1, group)
        grouped_pos = pos.reshape(-1, group)
        grouped_neg = neg.reshape(-1, group)
        grouped_weights = weights.reshape(-1, group)
        unique_users = grouped_users[:, 0]
        unique_pos = grouped_pos[:, 0]
        user_embedding = u[unique_users].unsqueeze(1)
        positive = (user_embedding * p[unique_pos].unsqueeze(1)).sum(2)
        negative = (user_embedding * p[grouped_neg]).sum(2)
        bpr = -F.logsigmoid(positive - negative)
        if include_reg:
            spot_ego, peak_ego = u0.square(), p0.square()
            reg = 0.5 * (
                spot_ego[unique_users].sum(1, keepdim=True)
                + peak_ego[unique_pos].sum(1, keepdim=True)
                + peak_ego[grouped_neg].sum(2)
            )
            reg_sum = (reg * grouped_weights).sum()
        else:
            reg_sum = bpr.new_zeros(())
        return (bpr * grouped_weights).sum(), reg_sum
    u_emb, p_emb, n_emb = u[users], p[pos], p[neg]
    positive = (u_emb * p_emb).sum(1)
    negative = (u_emb * n_emb).sum(1)
    bpr = -F.logsigmoid(positive - negative)
    if include_reg:
        spot_ego, peak_ego = u0.square(), p0.square()
        reg = 0.5 * (
            spot_ego[users].sum(1) + peak_ego[pos].sum(1) + peak_ego[neg].sum(1)
        )
        reg_sum = (reg * weights).sum()
    else:
        reg_sum = bpr.new_zeros(())
    return (bpr * weights).sum(), reg_sum


def prepare_dna_regularization_weight(degree, alpha, *, device, dtype):
    """Validate and normalize the static motif-degree weight once per run."""
    if degree is None:
        return None
    degree = torch.as_tensor(degree)
    if degree.ndim != 1 or degree.numel() == 0:
        raise ValueError("DNA degree must be a nonempty 1D vector")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not math.isfinite(alpha):
        raise ValueError("DNA regularization alpha must be finite")
    if not torch.isfinite(degree).all() or (degree < 0).any():
        raise ValueError("invalid DNA degree vector")
    reference = torch.empty((), device=device, dtype=dtype)
    if not reference.is_floating_point():
        raise ValueError("DNA regularization requires a floating-point model dtype")
    degree = degree.to(device=device, dtype=dtype)
    weight = degree.clamp_min(1).pow(float(alpha))
    weight = weight / (weight.mean() + 1e-12)
    return weight.detach()


def validate_training_indices(users, pos, neg, n_spots, n_peaks, *, check_range=True):
    """Validate a fresh batch; cached training batches need only metadata checks."""
    if any(x.dtype not in (torch.int32, torch.int64) for x in (users, pos, neg)):
        raise ValueError("training indices must have integer dtype")
    if users.ndim != 1 or pos.shape != users.shape or neg.shape != users.shape or users.numel() == 0:
        raise ValueError("training batch must be nonempty equal-length 1D local-index tensors")
    if check_range and (
        users.min() < 0 or users.max() >= n_spots or pos.min() < 0 or neg.min() < 0
        or max(pos.max(), neg.max()) >= n_peaks
    ):
        raise ValueError("training index outside local node universe")


def batch_edge_weights(users, n_spots, *, dtype):
    """Uniform occurrence weights for the single-slice BPR objective."""
    return torch.full((1,), 1 / len(users), dtype=dtype, device=users.device).expand(len(users))


def compute_loss(
    model,
    users,
    pos,
    neg,
    A,
    dna=None,
    spatial=None,
    use_spatial=True,
    r_spatial=0.0,
    r_dna=0.0,
    decay=0.001,
    dna_decay=1e-5,
    dna_degree=None,
    alpha=0.5,
    chunk_size=0,
    group_size=0,
    *,
    loss_backend="torch",
    loss_workspace_mb=256.0,
    weighted_ego_l2=False,
    ego_coefficients=None,
    dna_reg_weight=None,
    validate_index_range=True,
    loss_plan=None,
):
    # The owned loop validates fresh CPU batches and compiles its plan once.
    # Standalone calls keep validation at the public boundary.
    validate = validate_index_range or loss_plan is None
    if validate and (not isinstance(group_size, int) or isinstance(group_size, bool) or group_size < 0):
        raise ValueError("group size must be a nonnegative integer")
    group = group_size if group_size > 1 else 1
    if validate:
        validate_training_indices(users, pos, neg, model.num_spot, model.num_peak,
                                  check_range=validate_index_range)
        if chunk_size < 0:
            raise ValueError("chunk size must be nonnegative")
        if group > 1 and len(users) % group:
            raise ValueError("grouped loss requires a multiple of the group size")
        if group > 1 and chunk_size and chunk_size % group:
            raise ValueError("loss chunk size must be a multiple of the grouped loss size")
        if loss_backend not in ("torch", "torch_streamed", "auto", "cuda_fused"):
            raise ValueError("unsupported loss backend")
        if loss_backend in ("torch_streamed", "auto", "cuda_fused") and chunk_size:
            raise ValueError("chunk_size must be zero with torch_streamed, auto or cuda_fused")
        if not isinstance(weighted_ego_l2, bool):
            raise ValueError("weighted_ego_l2 must be boolean")
    u, p, _ = model.propagate(A, dna, spatial, use_spatial, r_spatial, r_dna)
    weights = batch_edge_weights(users, model.num_spot, dtype=u.dtype)
    bpr = u.new_zeros(())
    reg = u.new_zeros(())
    plan = loss_plan if loss_plan is not None else execution_plan(
        len(users), u.shape[1], u.element_size(), loss_backend, loss_workspace_mb, group, chunk_size,
        device=u.device, dtype=u.dtype, allow_fused=True)
    fused = plan["effective_backend"] == "cuda_fused"
    if fused and torch.are_deterministic_algorithms_enabled():
        raise ValueError("CUDA fused scoring requires non-deterministic single-slice execution")
    step = plan["chunk_triplets"]
    use_weighted_ego = weighted_ego_l2 or ego_coefficients is not None or fused
    if use_weighted_ego and ego_coefficients is None:
        ego_coefficients = prepare_ego_l2_coefficients(
            users,
            pos,
            neg,
            weights,
            n_spots=model.num_spot,
            n_peaks=model.num_peak,
            assume_uniform=True,
        )
    if fused:
        positive_scores, negative_scores = fused_sampled_scores(
            u, p, users, pos, neg, validate=validate)
        bpr = (-F.logsigmoid(positive_scores - negative_scores) * weights).sum()
    else:
        for start in range(0, len(users), step):
            sl = slice(start, start + step)
            args = (
                u,
                p,
                model.embedding_spot,
                model.embedding_peak,
                users[sl],
                pos[sl],
                neg[sl],
                weights[sl],
                group,
                not use_weighted_ego,
            )
            parts = (
                # This scorer draws no random numbers; shapes are fixed by the batch plan.
                checkpoint(_edge_sums, *args, use_reentrant=False,
                           preserve_rng_state=False, determinism_check="none")
                if plan["checkpoint"] and torch.is_grad_enabled()
                else _edge_sums(*args)
            )
            bpr = bpr + parts[0]
            reg = reg + parts[1]
    if use_weighted_ego:
        regularize = weighted_ego_regularization if validate else _weighted_ego_regularization
        reg = regularize(
            model.embedding_spot,
            model.embedding_peak,
            ego_coefficients,
        )
    reg_dna = u.new_zeros(())
    if model.embedding_kmer is not None:
        norm = model.embedding_kmer.square().sum(1)
        if dna_reg_weight is not None:
            if validate and (
                dna_reg_weight.shape != norm.shape
                or dna_reg_weight.device != norm.device
                or dna_reg_weight.dtype != norm.dtype
            ):
                raise ValueError("DNA regularization weight must match motif embeddings")
            reg_dna = 0.5 * (dna_reg_weight * norm).mean()
        elif dna_degree is None:
            reg_dna = 0.5 * norm.mean()
        else:
            degree = dna_degree.to(device=norm.device, dtype=norm.dtype)
            if degree.shape != norm.shape or not torch.isfinite(degree).all() or (degree < 0).any():
                raise ValueError("invalid DNA degree vector")
            w = degree.clamp_min(1).pow(float(alpha))
            w = w / (w.mean() + 1e-12)
            reg_dna = 0.5 * (w * norm).mean()
    elif dna_reg_weight is not None:
        raise ValueError("DNA regularization weight supplied without motif embeddings")
    loss = bpr + decay * reg + dna_decay * reg_dna
    return loss, {"bpr": bpr, "reg": reg, "reg_dna": reg_dna}



@torch.no_grad()
def clip_motif_rows(model, factor):
    if factor <= 0 or model.embedding_kmer is None:
        return None
    matrix = model.embedding_kmer
    row_norm = matrix.norm(dim=1, keepdim=True)
    limit = (row_norm.mean() * factor).clamp_min(6.0)
    matrix.mul_((limit / row_norm.clamp_min(1e-8)).clamp(max=1))
    return limit


# The public model name preserves the existing LightGCN implementation.
SEAREG = LightGCN
