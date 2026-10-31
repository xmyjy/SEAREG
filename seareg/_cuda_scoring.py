"""Fused CUDA kernels for the same sampled-dot operation (float32).

Each program processes a bounded tile of triplets. No dense pair matrix or
triplet-by-feature tensor is allocated in device global memory.
"""
import triton
import triton.language as tl


@triton.jit
def _forward(S, P, U, POS, NEG, OP, ON, N, D: tl.constexpr,
             B: tl.constexpr, K: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    j = tl.arange(0, K)
    u = tl.load(U + i, i < N, other=0)
    p = tl.load(POS + i, i < N, other=0)
    n = tl.load(NEG + i, i < N, other=0)
    valid = (i[:, None] < N) & (j[None, :] < D)
    s = tl.load(S + u[:, None] * D + j[None, :], valid, other=0)
    a = tl.load(P + p[:, None] * D + j[None, :], valid, other=0)
    b = tl.load(P + n[:, None] * D + j[None, :], valid, other=0)
    tl.store(OP + i, tl.sum(s * a, 1), i < N)
    tl.store(ON + i, tl.sum(s * b, 1), i < N)


@triton.jit
def _backward(S, P, U, POS, NEG, GP, GN, DS, DP,
              N, D: tl.constexpr, NS,
              HAS_POS: tl.constexpr, HAS_NEG: tl.constexpr,
              B: tl.constexpr, K: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    j = tl.arange(0, K)
    u = tl.load(U + i, i < N, other=0)
    p = tl.load(POS + i, i < N, other=0)
    n = tl.load(NEG + i, i < N, other=0)
    valid = (i[:, None] < N) & (j[None, :] < D)
    s = tl.load(S + u[:, None] * D + j[None, :], valid, other=0)
    ds = tl.full((B, K), 0, tl.float32)
    if HAS_POS:
        g = tl.load(GP + i, i < N, other=0)
        a = tl.load(P + p[:, None] * D + j[None, :], valid, other=0)
        ds += g[:, None] * a
        tl.atomic_add(DP + p[:, None] * D + j[None, :], g[:, None] * s, valid,
                      sem="relaxed")
    if HAS_NEG:
        g = tl.load(GN + i, i < N, other=0)
        a = tl.load(P + n[:, None] * D + j[None, :], valid, other=0)
        ds += g[:, None] * a
        tl.atomic_add(DP + n[:, None] * D + j[None, :], g[:, None] * s, valid,
                      sem="relaxed")
    lo = tl.min(tl.where(i < N, u, NS), 0)
    hi = tl.max(tl.where(i < N, u, -1), 0)
    if lo == hi:
        # Samplers group triplets by spot. Combine locally before accumulation.
        tl.atomic_add(DS + lo * D + j, tl.sum(ds, 0), j < D, sem="relaxed")
    else:
        tl.atomic_add(DS + u[:, None] * D + j[None, :], ds, valid, sem="relaxed")


def forward(spots, peaks, users, positive, negative, pos_scores, neg_scores):
    n, d = users.numel(), spots.shape[1]
    _forward[(triton.cdiv(n, 32),)](
        spots, peaks, users, positive, negative, pos_scores, neg_scores,
        n, d, 32, triton.next_power_of_2(d), num_warps=4, enable_fp_fusion=False)


def backward(spots, peaks, users, positive, negative, gp, gn, ds, dp):
    n, d = users.numel(), spots.shape[1]
    _backward[(triton.cdiv(n, 32),)](
        spots, peaks, users, positive, negative, gp, gn, ds, dp,
        n, d, spots.shape[0], gp is not None, gn is not None,
        32, triton.next_power_of_2(d), num_warps=4, enable_fp_fusion=False)
