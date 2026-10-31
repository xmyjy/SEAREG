"""Deterministic workspace planning and memory-efficient ego regularization."""

from dataclasses import dataclass
import math

import torch


def workspace_plan(
    n_triplets,
    dimension,
    element_size,
    workspace_mb=256.0,
    group_size=0,
):
    """Return a deterministic chunk schedule that never splits a sampler group."""
    for value, name in (
        (n_triplets, "triplet count"),
        (dimension, "feature dimension"),
        (element_size, "element size"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size < 0:
        raise ValueError("group size must be a nonnegative integer")
    if not math.isfinite(workspace_mb) or workspace_mb <= 0:
        raise ValueError("loss workspace must be finite and positive")
    group = group_size if group_size > 1 else 1
    if n_triplets % group:
        raise ValueError("triplet count must contain only complete loss groups")

    # Conservative bound for gathers, products, checkpoint inputs and indices.
    per_triplet = 8 * dimension * element_size + 64
    budget = int(workspace_mb * 1024**2)
    capacity = budget // per_triplet
    capacity = (capacity // group) * group
    if capacity < group:
        raise ValueError("loss workspace is too small for one complete loss group")
    chunk = min(n_triplets, capacity)
    return {
        "implementation": "torch_checkpoint_streamed_v1",
        "workspace_bytes": budget,
        "chunk_triplets": chunk,
        "estimated_chunk_bytes": chunk * per_triplet,
        "chunks": math.ceil(n_triplets / chunk),
        "group_size": group,
        "dense_spot_peak_matrix": False,
    }


def execution_plan(n_triplets, dimension, element_size, backend="auto",
                   workspace_mb=256.0, group_size=0, chunk_size=0, *,
                   device=None, dtype=None, allow_fused=True):
    """Choose eager or checkpointed loss without altering samples or updates."""
    if backend not in ("torch", "torch_streamed", "auto", "cuda_fused"):
        raise ValueError("unsupported loss backend")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 0:
        raise ValueError("invalid chunk size")
    if backend != "torch" and chunk_size:
        raise ValueError("explicit chunks require the torch backend")
    if not math.isfinite(workspace_mb) or workspace_mb <= 0:
        raise ValueError("loss workspace must be finite and positive")
    fused = False
    if backend in ("auto", "cuda_fused"):
        from .scoring import fused_support
        fused, reason, _ = fused_support(device, dtype, dimension, allow_fused=allow_fused)
        if backend == "cuda_fused" and not fused:
            raise ValueError(reason)
    # Explicit eager execution historically has no workspace cap. Still use
    # the same argument/group validation without imposing the auto policy.
    planning_budget = workspace_mb
    if backend == "torch":
        planning_budget = max(workspace_mb, (8 * dimension * element_size + 64) * max(1, group_size) / 1024**2)
    plan = workspace_plan(n_triplets, dimension, element_size, planning_budget, group_size)
    plan.update(workspace_bytes=int(workspace_mb * 1024**2), budget_enforced=(backend != "torch"))
    if fused:
        # The fused kernel tiles internally; workspace is the PyTorch fallback
        # planning budget, not a limit on total process memory or score vectors.
        plan.update(effective_backend="cuda_fused", implementation="cuda_fused_sampled_dot_v1",
                    checkpoint=False, chunk_triplets=n_triplets, chunks=1,
                    estimated_chunk_bytes=2*n_triplets*element_size,
                    budget_enforced=False, workspace_role="torch_fallback_planning")
        return plan
    if backend == "torch":
        step = min(chunk_size or n_triplets, n_triplets)
        group = group_size if group_size > 1 else 1
        if step % group:
            raise ValueError("chunk size must preserve complete groups")
        plan.update(chunk_triplets=step, chunks=math.ceil(n_triplets / step),
                    estimated_chunk_bytes=step * (8 * dimension * element_size + 64))
    effective = "torch" if backend == "torch" or (backend == "auto" and plan["chunks"] == 1) else "torch_streamed"
    plan.update(effective_backend=effective, checkpoint=(effective == "torch_streamed" or bool(chunk_size)))
    return plan


@dataclass(frozen=True)
class EgoL2Coefficients:
    """Per-node sums of the exact edge weights used by sampled BPR."""

    spot: torch.Tensor
    peak: torch.Tensor

    def to(self, device=None, dtype=None):
        return EgoL2Coefficients(
            self.spot.to(device=device, dtype=dtype),
            self.peak.to(device=device, dtype=dtype),
        )


def prepare_ego_l2_coefficients(
    users,
    positive,
    negative,
    weights,
    *,
    n_spots,
    n_peaks,
    assume_uniform=False,
):
    """Collapse the weighted triplet L2 objective into per-node coefficients."""
    if any(index.dtype not in (torch.int32, torch.int64) for index in (users, positive, negative)):
        raise ValueError("ego coefficient indices must have integer dtype")
    if (
        users.ndim != 1
        or positive.shape != users.shape
        or negative.shape != users.shape
        or weights.shape != users.shape
        or users.numel() == 0
    ):
        raise ValueError("ego coefficients require nonempty equal-length vectors")
    if not weights.is_floating_point():
        raise ValueError("ego coefficient weights must be floating point")
    if any(index.device != weights.device for index in (users, positive, negative)):
        raise ValueError("ego coefficient indices and weights must share a device")
    for value, name in ((n_spots, "spot count"), (n_peaks, "peak count")):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(assume_uniform) is not bool:
        raise ValueError("assume_uniform must be boolean")

    def deterministic_sum(index, values, size):
        # The single-slice path has one exact edge weight.  Integer bincount is
        # deterministic on CUDA and avoids both weighted CUDA atomics and a sort.
        if assume_uniform:
            return torch.bincount(index.long(), minlength=size).to(values.dtype) * values[0]
        order = torch.argsort(index, stable=True)
        sorted_index = index[order].long()
        sorted_values = values[order]
        unique, counts = torch.unique_consecutive(sorted_index, return_counts=True)
        result = values.new_zeros(size)
        result[unique] = torch.segment_reduce(
            sorted_values, reduce="sum", lengths=counts
        )
        return result

    spot = deterministic_sum(users, weights, n_spots)
    peak = deterministic_sum(positive, weights, n_peaks)
    peak = peak + deterministic_sum(negative, weights, n_peaks)
    return EgoL2Coefficients(spot=spot, peak=peak)


def weighted_ego_regularization(spots, peaks, coefficients):
    """Evaluate the original weighted triplet ego L2 without T-by-D gathers."""
    if not isinstance(coefficients, EgoL2Coefficients):
        raise TypeError("coefficients must be EgoL2Coefficients")
    if spots.ndim != 2 or peaks.ndim != 2 or spots.shape[1] != peaks.shape[1]:
        raise ValueError("spot and peak embeddings require the same feature dimension")
    if coefficients.spot.shape != (spots.shape[0],) or coefficients.peak.shape != (
        peaks.shape[0],
    ):
        raise ValueError("ego coefficient shape mismatch")
    if (
        coefficients.spot.device != spots.device
        or coefficients.peak.device != peaks.device
        or coefficients.spot.dtype != spots.dtype
        or coefficients.peak.dtype != peaks.dtype
    ):
        raise ValueError("ego coefficients must match embedding device and dtype")
    return _weighted_ego_regularization(spots, peaks, coefficients)


def _weighted_ego_regularization(spots, peaks, coefficients):
    """Nodewise L2 for coefficients already validated at batch preparation."""
    spot_row_sq = spots.square().sum(1)
    peak_row_sq = peaks.square().sum(1)
    return 0.5 * (
        (coefficients.spot * spot_row_sq).sum()
        + (coefficients.peak * peak_row_sq).sum()
    )
