"""Budget-aware CUDA residency for immutable sampled shard tensors."""

import math
import time

import torch


GIB = 1024**3


def _tensor_key(tensor):
    return id(tensor), tensor._version


def tensor_bytes(tensor):
    """Count logical payload bytes without treating sparse tensors as dense."""
    if tensor.layout == torch.sparse_coo:
        return (
            tensor._indices().numel() * tensor._indices().element_size()
            + tensor._values().numel() * tensor._values().element_size()
        )
    if tensor.layout == torch.sparse_csr:
        return (
            tensor.crow_indices().numel() * tensor.crow_indices().element_size()
            + tensor.col_indices().numel() * tensor.col_indices().element_size()
            + tensor.values().numel() * tensor.values().element_size()
        )
    return tensor.numel() * tensor.element_size()


def _batch_tensors(batch):
    return (batch.users, batch.positive, batch.negative, batch.adjacency)


def _batch_keys(batch):
    return tuple(_tensor_key(tensor) for tensor in _batch_tensors(batch))


def _unique_tensors(batches):
    result = {}
    for batch in batches:
        for tensor in _batch_tensors(batch):
            if not torch.is_tensor(tensor):
                raise TypeError("resident cache accepts tensor-valued shard fields only")
            if tensor.requires_grad:
                raise ValueError("resident cache cannot own tensors that require gradients")
            result.setdefault(_tensor_key(tensor), tensor)
    return result


class ResidentBatchCache:
    """Keep all shards on CUDA only when the declared budget and headroom allow it."""

    def __init__(self, enabled=True, budget_gib=0.0, reserve_gib=4.0):
        if type(enabled) is not bool:
            raise ValueError("resident cache enabled must be boolean")
        if any(not math.isfinite(value) or value < 0 for value in (budget_gib, reserve_gib)):
            raise ValueError("resident cache budget and reserve must be finite and nonnegative")
        self.enabled = enabled
        self.budget_bytes = math.ceil(budget_gib * GIB)
        self.reserve_bytes = math.ceil(reserve_gib * GIB)
        self.source = None
        self.source_keys = None
        self.device_batches = None
        self.device = None
        self.fallback_source = None
        self.fallback_keys = None
        self.stats = {
            "enabled": enabled,
            "budget_mode": "auto" if budget_gib == 0 else "explicit",
            "budget_bytes": self.budget_bytes,
            "reserve_bytes": self.reserve_bytes,
            "status": "pending" if enabled else "disabled",
            "device": None,
            "payload_bytes": 0,
            "cached_bytes": 0,
            "upload_available_bytes": None,
            "copy_seconds": 0.0,
            "copy_events": 0,
            "resident_epochs": 0,
            "streaming_epochs": 0,
            "fallback_reasons": {},
        }

    def release(self):
        self.source = None
        self.source_keys = None
        self.device_batches = None
        self.stats["cached_bytes"] = 0

    def _fallback(self, reason, source=None, source_keys=None):
        self.release()
        self.fallback_source = source
        self.fallback_keys = source_keys
        self.stats["status"] = reason
        self.stats["streaming_epochs"] += 1
        reasons = self.stats["fallback_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1
        return None

    def prepare(self, batches, device):
        """Return resident device batches, or None for pre-step per-shard streaming."""
        if not isinstance(batches, (list, tuple)) or not batches:
            raise ValueError("resident cache requires a nonempty shard sequence")
        if not self.enabled:
            self.release()
            self.stats["status"] = "disabled"
            self.stats["streaming_epochs"] += 1
            return None

        tensors = _unique_tensors(batches)
        source_keys = tuple(_batch_keys(batch) for batch in batches)
        payload = sum(tensor_bytes(tensor) for tensor in tensors.values())
        self.stats["payload_bytes"] = payload
        device = torch.device(device)
        self.stats["device"] = str(device)
        # This branch intentionally precedes every CUDA query/copy. It is also
        # testable on hosts where a requested CUDA device is not visible.
        if self.budget_bytes and payload > self.budget_bytes:
            return self._fallback("fallback_budget", batches, source_keys)

        if device.type != "cuda":
            return self._fallback("fallback_non_cuda", batches, source_keys)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.stats["device"] = str(device)

        if (
            self.device == device
            and self.source is batches
            and self.source_keys == source_keys
            and self.device_batches is not None
        ):
            self.stats["status"] = "resident"
            self.stats["resident_epochs"] += 1
            return self.device_batches
        if self.fallback_source is batches and self.fallback_keys == source_keys:
            self.stats["streaming_epochs"] += 1
            reason = self.stats["status"]
            reasons = self.stats["fallback_reasons"]
            reasons[reason] = reasons.get(reason, 0) + 1
            return None

        reuse_stable = (
            self.device == device
            and self.source is not None
            and self.device_batches is not None
            and len(batches) == len(self.source) == len(self.device_batches)
            and all(
                current[0] == prior[0]
                and current[1] == prior[1]
                and current[3] == prior[3]
                for current, prior in zip(source_keys, self.source_keys)
            )
        )
        prior_device = self.device_batches if reuse_stable else None
        if not reuse_stable:
            self.release()

        retained_keys = set()
        if reuse_stable:
            for keys in self.source_keys:
                retained_keys.update((keys[0], keys[1], keys[3]))
        needed = sum(
            tensor_bytes(tensor)
            for key, tensor in tensors.items()
            if key not in retained_keys and tensor.device != device
        )
        free, _ = torch.cuda.mem_get_info(device)
        reusable = max(
            0,
            torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device),
        )
        available = max(0, int(free) + int(reusable) - self.reserve_bytes)
        self.stats["upload_available_bytes"] = available
        if needed > available:
            return self._fallback("fallback_headroom", batches, source_keys)

        start = time.perf_counter()
        try:
            device_batches = [
                batch.to(device, stable=cached if reuse_stable else None)
                for batch, cached in zip(batches, prior_device or ())
            ] if reuse_stable else [batch.to(device) for batch in batches]
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError:
            device_batches = None
            result = self._fallback("fallback_upload_oom", batches, source_keys)
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            return result

        if needed:
            self.stats["copy_seconds"] += time.perf_counter() - start
            self.stats["copy_events"] += 1
        self.source = batches
        self.source_keys = source_keys
        self.device_batches = device_batches
        self.device = device
        self.fallback_source = None
        self.fallback_keys = None
        self.stats["cached_bytes"] = payload
        self.stats["status"] = "resident"
        self.stats["resident_epochs"] += 1
        return device_batches

    @staticmethod
    def iter_batches(source, prepared, device):
        if prepared is not None:
            yield from prepared
            return
        for batch in source:
            yield batch.to(device)
