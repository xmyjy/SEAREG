"""Original within-slice adaptive-tail-excess first-layer correction.

The gate is detached and introduces no learnable parameters. Raw embeddings
remain S1+S2 and P1+P2; correction is explicitly enabled by the multi API.
"""
import copy
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from ..model import LightGCN

class _DepthGateLightGCN(LightGCN):
    def __init__(self, n_spots, n_peaks, dim, degree):
        super().__init__(n_spots, n_peaks, 0, dim)
        self.first_layer_gate_mode = 'adaptive_tail_excess'
        degree = torch.as_tensor(degree, dtype=torch.float32)
        if degree.shape != (n_spots,) or torch.any(degree <= 0):
            raise ValueError('depth gate requires positive full spot-peak degrees')
        self.register_buffer('spot_peak_degree', degree.clone())
        self.register_buffer('cached_first_layer_gate', torch.zeros(n_spots))
        self.register_buffer('cached_degree_gate', torch.zeros(n_spots))
        self.register_buffer('_adaptive_degree_observed', None, persistent=False)
        self.register_buffer('_adaptive_degree_null', None, persistent=False)
        self._last_gate_summary = None

    @staticmethod
    def _tail_excess_gate(
            observed_score: torch.Tensor, null_score: torch.Tensor
    ) -> torch.Tensor:
        """Estimate a monotone non-null probability without a fixed q cutoff."""
        observed = observed_score.detach().float()
        null = null_score.detach().float().reshape(-1)
        sorted_observed = torch.sort(observed).values
        sorted_null = torch.sort(null).values
        observed_tail = (
            observed.numel()
            - torch.searchsorted(sorted_observed, observed, right=False)
        ).to(observed.dtype) / float(observed.numel())
        null_tail = (
            null.numel()
            - torch.searchsorted(sorted_null, observed, right=False)
        ).to(observed.dtype) / float(null.numel())
        raw = (
            1.0 - null_tail / observed_tail.clamp_min(1.0 / observed.numel())
        ).clamp(0.0, 1.0)

        order = torch.argsort(observed, stable=True)
        fitted = IsotonicRegression(
            increasing=True, y_min=0.0, y_max=1.0, out_of_bounds="clip"
        ).fit_transform(
            observed[order].cpu().numpy().astype(np.float64, copy=False),
            raw[order].cpu().numpy().astype(np.float64, copy=False),
        )
        result = torch.empty_like(observed)
        result[order] = torch.from_numpy(
            np.asarray(fitted, dtype=np.float32)
        ).to(observed.device)
        return result

    @staticmethod
    def _gate_summary(gate: torch.Tensor) -> dict:
        return {
            "mean": gate.mean().detach(),
            "nonzero_fraction": (gate > 1e-12).to(gate.dtype).mean().detach(),
            "full_fraction": (gate >= 1.0 - 1e-7).to(gate.dtype).mean().detach(),
            "ge_0p25_fraction": (gate >= .25).to(gate.dtype).mean().detach(),
            "ge_0p50_fraction": (gate >= .50).to(gate.dtype).mean().detach(),
            "ge_0p75_fraction": (gate >= .75).to(gate.dtype).mean().detach(),
        }

    @staticmethod
    def _slice_codes(
            slice_codes: Optional[torch.Tensor], n_spots: int, device: torch.device
    ) -> torch.Tensor:
        if slice_codes is None:
            return torch.zeros(n_spots, dtype=torch.long, device=device)
        codes = torch.as_tensor(slice_codes, dtype=torch.long, device=device)
        if codes.shape != (n_spots,) or torch.any(codes < 0):
            raise ValueError("slice_codes must be non-negative with shape [num_spot]")
        return codes

    @staticmethod
    def _normalize_within_slices(
            values: torch.Tensor, slice_codes: torch.Tensor
    ) -> torch.Tensor:
        result = torch.empty_like(values)
        for code in torch.unique(slice_codes, sorted=True):
            selected = slice_codes == code
            if values.ndim == 1:
                scale = values[selected].median().clamp_min(1e-8)
            elif values.ndim == 2:
                scale = values[selected].median(dim=0).values.clamp_min(1e-8)
            else:
                raise ValueError("adaptive scores must be one- or two-dimensional")
            result[selected] = values[selected] / scale
        return result

    @classmethod
    def _tail_excess_gate_within_slices(
            cls, observed: torch.Tensor, null: torch.Tensor,
            slice_codes: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.empty_like(observed)
        for code in torch.unique(slice_codes, sorted=True):
            selected = slice_codes == code
            result[selected] = cls._tail_excess_gate(
                observed[selected], null[selected]
            )
        return result

    @classmethod
    def _gate_summary_by_slice(
            cls, gate: torch.Tensor, slice_codes: torch.Tensor
    ) -> dict:
        return {
            int(code.item()): cls._gate_summary(gate[slice_codes == code])
            for code in torch.unique(slice_codes, sorted=True)
        }

    @torch.no_grad()
    def initialize_adaptive_gate(
            self, A_spatial: torch.Tensor, n_permutations: int, seed: int,
            slice_codes: Optional[torch.Tensor] = None,
    ) -> dict:
        """Build an independent within-slice degree null for every slice."""
        if self.spot_peak_degree is None:
            raise RuntimeError("adaptive gate requires full spot-peak degree")
        if n_permutations < 2:
            raise ValueError("adaptive gate requires at least two permutations")

        log_degree = torch.log2(self.spot_peak_degree.float() + 1.0)
        codes = self._slice_codes(slice_codes, self.num_spot, log_degree.device)
        observed = torch.abs(
            log_degree - self._spatial_row_mean(
                A_spatial, log_degree.unsqueeze(1)
            ).squeeze(1)
        )
        observed = self._normalize_within_slices(observed, codes)

        permuted = torch.empty(
            (self.num_spot, int(n_permutations)),
            dtype=log_degree.dtype,
            device=log_degree.device,
        )
        for code in torch.unique(codes, sorted=True):
            selected = codes == code
            base = log_degree[selected].detach().cpu().numpy()
            # Reusing the registered seed independently in each slice matches
            # the preceding single-slice experiments exactly.
            rng = np.random.default_rng(int(seed))
            local = np.column_stack([
                rng.permutation(base) for _ in range(int(n_permutations))
            ]).astype(np.float32, copy=False)
            permuted[selected] = torch.from_numpy(local).to(log_degree.device)
        null = torch.abs(permuted - self._spatial_row_mean(A_spatial, permuted))
        null = self._normalize_within_slices(null, codes)

        self._adaptive_degree_observed = observed
        self._adaptive_degree_null = null
        gate = self._tail_excess_gate_within_slices(observed, null, codes)
        self.cached_degree_gate.copy_(gate.to(self.cached_degree_gate))
        self.cached_first_layer_gate.copy_(gate.to(self.cached_first_layer_gate))
        self._last_gate_summary = self._gate_summary(gate)
        result = dict(self._last_gate_summary)
        result["by_slice"] = self._gate_summary_by_slice(gate, codes)
        return result

    @torch.no_grad()
    def refresh_adaptive_gate(
            self, A_full_spot_peak: torch.Tensor, A_spatial: torch.Tensor,
            slice_codes: Optional[torch.Tensor] = None,
    ) -> dict:
        """Refresh the gate from P1 using independent within-slice calibration."""
        if self._adaptive_degree_observed is None or self._adaptive_degree_null is None:
            raise RuntimeError("initialize_adaptive_gate must run before refresh")
        codes = self._slice_codes(
            slice_codes, self.num_spot, self.embedding_spot.device
        )
        x = torch.cat([self.embedding_spot, self.embedding_peak], dim=0)
        p1 = torch.sparse.mm(A_full_spot_peak, x)[:self.num_spot]
        neighbour = self._spatial_row_mean(A_spatial, p1)
        message = torch.linalg.vector_norm(
            F.normalize(p1, p=2, dim=1, eps=1e-8)
            - F.normalize(neighbour, p=2, dim=1, eps=1e-8), dim=1
        )
        message = self._normalize_within_slices(message, codes)
        observed_score = message * self._adaptive_degree_observed
        null_score = message.unsqueeze(1) * self._adaptive_degree_null
        joint_gate = self._tail_excess_gate_within_slices(
            observed_score, null_score, codes
        )
        gate = joint_gate
        self.cached_first_layer_gate.copy_(gate.to(self.cached_first_layer_gate))
        self._last_gate_summary = self._gate_summary(gate)
        result = dict(self._last_gate_summary)
        result["degree_mean"] = self.cached_degree_gate.mean().detach()
        result["joint_mean"] = joint_gate.mean().detach()
        result["by_slice"] = self._gate_summary_by_slice(gate, codes)
        result["degree_by_slice"] = self._gate_summary_by_slice(
            self.cached_degree_gate.to(gate), codes
        )
        result["joint_by_slice"] = self._gate_summary_by_slice(joint_gate, codes)
        return result

    @staticmethod
    def _spatial_row_mean(A_spatial: torch.Tensor, values: torch.Tensor,
                          source_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if source_weight is None:
            source_weight = torch.ones(
                values.size(0), dtype=values.dtype, device=values.device
            )
        weighted = values * source_weight.unsqueeze(1)
        numerator = torch.sparse.mm(A_spatial, weighted)
        denominator = torch.sparse.mm(A_spatial, source_weight.unsqueeze(1))
        return numerator / denominator.clamp_min(1e-8)

    def _correct_first_layer_peak_message(self, p1, spatial):
        gate = self.cached_first_layer_gate.detach().to(p1)
        clean = self._spatial_row_mean(spatial, p1, 1.0 - gate)
        return p1 + gate.unsqueeze(1) * (clean - p1)

    def propagate(self, A, dna=None, spatial=None, use_spatial=True,
                  r_spatial=0.0, r_dna=0.0):
        A_spot_peak, A_spatial = A, spatial
        users = self.embedding_spot
        items = self.embedding_peak

        x = torch.cat([users, items], dim=0)
        x1 = torch.sparse.mm(A_spot_peak, x)
        u1, i1 = torch.split(x1, [self.num_spot, self.num_peak], dim=0)

        if use_spatial and A_spatial is not None:
            u1 = self._correct_first_layer_peak_message(u1, A_spatial)

        if use_spatial and A_spatial is not None:
            u1 = u1 + r_spatial * torch.sparse.mm(A_spatial, users)

        x1 = torch.cat([u1, i1], dim=0)
        x2 = torch.sparse.mm(A_spot_peak, x1)
        u2, i2 = torch.split(x2, [self.num_spot, self.num_peak], dim=0)

        if use_spatial and A_spatial is not None:
            u2 = u2 + r_spatial * torch.sparse.mm(A_spatial, u1)

        kmers_final = None
        users_final = u1 + u2
        items_final = i1 + i2

        return users_final, items_final, kmers_final

class GateBackbone(_DepthGateLightGCN):
    def __init__(self, n_spots, n_peaks, dim, degree):
        degree = torch.as_tensor(degree, dtype=torch.float32)
        if degree.shape != (n_spots,) or not torch.isfinite(degree).all() or (degree <= 0).any():
            raise ValueError('depth gate requires finite positive full-filtered degrees')
        super().__init__(n_spots, n_peaks, dim, degree)
        self.register_buffer('_correction_enabled', torch.tensor(True))
        self.register_buffer('gate_epoch', torch.tensor(-1, dtype=torch.long))
        self.register_buffer('null_seed', torch.tensor(-1, dtype=torch.long))
        self.register_buffer('null_permutations', torch.tensor(0, dtype=torch.long))
        # Preserve the legacy null in checkpoints for identical future refreshes.
        self._non_persistent_buffers_set.difference_update(
            {'_adaptive_degree_observed', '_adaptive_degree_null'})
        self._enabled = True

    def validate_gate_state(self):
        for key in ['spot_peak_degree', 'cached_first_layer_gate', 'cached_degree_gate',
                    '_adaptive_degree_observed', '_adaptive_degree_null']:
            value = getattr(self, key)
            if value is not None and not torch.isfinite(value).all():
                raise FloatingPointError(f'nonfinite depth gate buffer: {key}')
        for key in ['cached_first_layer_gate', 'cached_degree_gate']:
            gate = getattr(self, key)
            if gate.shape != (self.num_spot,) or (gate < 0).any() or (gate > 1).any():
                raise ValueError(f'invalid depth gate probability: {key}')
        if (self.spot_peak_degree <= 0).any():
            raise ValueError('nonpositive full-filtered spot degree')

    @torch.no_grad()
    def initialize_adaptive_gate(self, spatial, n_permutations, seed, slice_codes=None):
        result = super().initialize_adaptive_gate(spatial, n_permutations, seed, slice_codes)
        self.null_seed.fill_(seed); self.null_permutations.fill_(n_permutations)
        self.gate_epoch.zero_(); self.validate_gate_state()
        return result

    @torch.no_grad()
    def refresh_if_due(self, completed_epoch, full, spatial, codes, every=50):
        if every < 1 or completed_epoch < 0:
            raise ValueError('refresh cadence must be positive and epoch nonnegative')
        if completed_epoch < int(self.gate_epoch):
            raise ValueError('stale backward gate epoch')
        if completed_epoch == int(self.gate_epoch):
            return False
        if completed_epoch == 0:
            raise ValueError('initialize the gate before refreshing')
        if completed_epoch % every:
            return False
        super().refresh_adaptive_gate(full, spatial, codes)
        self.validate_gate_state(); self.gate_epoch.fill_(completed_epoch)
        return True

    def load_state_dict(self, state_dict, strict=True, assign=False):
        for name in ['_adaptive_degree_observed', '_adaptive_degree_null']:
            if name in state_dict:
                self._buffers[name] = torch.empty_like(
                    state_dict[name], device=self.embedding_spot.device)
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._enabled = bool(self._correction_enabled)
        self.validate_gate_state()
        return result

@torch.no_grad()
def infer_depth_gate_cpu(model, full, dna, spatial, use_spatial, rs, rd):
    """Copy parameters and all gate/null buffers to CPU without drawing RNG."""
    memo = {id(p): torch.nn.Parameter(p.detach().to('cpu', copy=True), requires_grad=p.requires_grad)
            for p in model.parameters()}
    memo.update({id(b): b.detach().to('cpu', copy=True) for b in model.buffers()})
    cpu_model = copy.deepcopy(model, memo).eval()
    values = cpu_model.propagate(full.cpu(), None if dna is None else dna.cpu(),
                                None if spatial is None else spatial.cpu(), use_spatial, rs, rd)
    return tuple(None if x is None else x.detach() for x in values)
