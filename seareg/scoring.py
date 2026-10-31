"""Optional single-slice CUDA scoring; never change numerical execution mode."""

from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable


@lru_cache(maxsize=4)
def _runtime_support(cuda_version, compiler_override):
    """Resolve optional resources once, outside the optimizer hot path."""
    if importlib.util.find_spec('triton') is None:
        return False, 'Triton is unavailable', None
    compiler = compiler_override
    if cuda_version and cuda_version.startswith('11.') and not compiler:
        bundled = Path(__file__).parent / '_compiler/cuda11/ptxas'
        if not bundled.is_file():
            return False, 'CUDA 11 requires a compatible ptxas resource', None
        compiler = str(bundled.resolve())
        os.environ['TRITON_PTXAS_PATH'] = compiler
    if compiler and (not Path(compiler).is_file() or not os.access(compiler, os.X_OK)):
        return False, 'configured ptxas is not an executable file', None
    info = {'backend': 'cuda_fused_sampled_dot_v1',
            'triton': importlib.metadata.version('triton'),
            'cuda': cuda_version, 'compiler': compiler}
    if compiler:
        info['compiler_sha256'] = hashlib.sha256(Path(compiler).read_bytes()).hexdigest()
    return True, None, info


def fused_support(device, dtype, dimension, *, allow_fused=True):
    """Report eligibility without importing kernels or relaxing determinism."""
    if not allow_fused:
        return False, 'CUDA fused scoring is restricted to single-slice training', None
    if torch.are_deterministic_algorithms_enabled():
        return False, 'CUDA fused scoring requires non-deterministic execution', None
    if device is None or torch.device(device).type != 'cuda':
        return False, 'CUDA fused scoring requires CUDA tensors', None
    if dtype != torch.float32 or not 0 < dimension <= 256:
        return False, 'CUDA fused scoring requires Float32 width 1..256', None
    if torch.version.hip is not None or not torch.cuda.is_available():
        return False, 'compatible CUDA execution is unavailable', None
    return _runtime_support(torch.version.cuda, os.environ.get('TRITON_PTXAS_PATH'))


class _FusedSampledScores(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spots, peaks, users, positive, negative):
        from ._cuda_scoring import forward
        positive_scores = spots.new_empty(users.numel())
        negative_scores = spots.new_empty(users.numel())
        forward(spots, peaks, users, positive, negative, positive_scores, negative_scores)
        ctx.save_for_backward(spots, peaks, users, positive, negative)
        ctx.set_materialize_grads(False)
        return positive_scores, negative_scores

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_positive, grad_negative):
        if torch.are_deterministic_algorithms_enabled():
            raise ValueError('CUDA fused backward requires non-deterministic execution')
        from ._cuda_scoring import backward
        spots, peaks, users, positive, negative = ctx.saved_tensors
        grad_spots, grad_peaks = torch.zeros_like(spots), torch.zeros_like(peaks)
        backward(spots, peaks, users, positive, negative,
                 None if grad_positive is None else grad_positive.contiguous(),
                 None if grad_negative is None else grad_negative.contiguous(),
                 grad_spots, grad_peaks)
        return grad_spots, grad_peaks, None, None, None


def fused_sampled_scores(spots, peaks, users, positive, negative, *, validate=True):
    """Score only requested pairs with the verified fused forward/backward.

    Direct calls validate index bounds. The owned loop validates immutable CPU
    batches before upload and passes validate=False, avoiding per-step GPU sync.
    Higher-order gradients are intentionally unsupported, as in the original
    sampled scorer. First-order optimization retains the sampled-dot objective.
    """
    if validate:
        if spots.ndim != 2 or peaks.ndim != 2 or spots.shape[1] != peaks.shape[1]:
            raise ValueError('spot/peak embeddings require the same feature width')
        if spots.device != peaks.device or spots.dtype != peaks.dtype:
            raise ValueError('spot/peak embeddings require the same device and dtype')
        supported, reason, _ = fused_support(spots.device, spots.dtype, spots.shape[1])
        if not supported:
            raise ValueError(reason)
        for index in (users, positive, negative):
            if (index.ndim != 1 or index.shape != users.shape or not index.numel()
                    or index.dtype not in (torch.int32, torch.int64) or index.device != spots.device):
                raise ValueError('triplets require nonempty equal-length CUDA integer vectors')
        bounds = torch.stack([bound for index in (users, positive, negative)
                              for bound in torch.aminmax(index)]).tolist()
        for (low, high), size in zip(zip(bounds[::2], bounds[1::2]),
                                    (spots.shape[0], peaks.shape[0], peaks.shape[0])):
            if low < 0 or high >= size:
                raise ValueError('triplet index outside local node universe')
    return _FusedSampledScores.apply(spots.contiguous(), peaks.contiguous(),
                                    users.long().contiguous(), positive.long().contiguous(),
                                    negative.long().contiguous())
