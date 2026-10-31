"""Balanced slice scheduling over the native SEAREG scoring operators.

The caller owns sampler refresh, model initialization, checkpoints and exports.
This module never runs analysis or invokes the historical model implementation.
"""
from __future__ import annotations

from collections import Counter
import copy
import time

import numpy as np
import torch

from ..loss import EgoL2Coefficients, execution_plan
from ..model import validate_training_indices
from ..resident import ResidentBatchCache
from ..sampling import ShardBatch
from .grouped_loss import bpr_only, group_regularization



def selected_shard_indices(
    bundle, epoch: int, update_budget: int,
    strategy: str = "independent_seeded_shuffle", seed: int = 1234,
) -> list[int]:
    """Select independent seeded full shard passes for each slice."""
    n_shards = int(bundle.num_shards)
    if strategy == "independent_seeded_shuffle":
        selected = []
        pass_index = 0
        while len(selected) < int(update_budget):
            rng = np.random.default_rng([
                int(seed), 0x5A17, int(epoch), int(bundle.code), pass_index,
            ])
            permutation = rng.permutation(n_shards).tolist()
            remaining = int(update_budget) - len(selected)
            selected.extend(permutation[:remaining])
            pass_index += 1
        return selected
    raise ValueError('unsupported shard pairing strategy')


def _cpu_batch(bundle, index, data, model, device, args, cache_state):
    original = bundle.batch_all[index]
    if len(original) != 3:
        raise ValueError("a shard must contain users, positive and negative indices")
    source = tuple(original)
    dtype = model.embedding_spot.dtype
    adjacency = bundle.adjacencies[index]
    signature = (tuple((id(value), value._version) for value in source),
                 (id(adjacency), adjacency._version), str(dtype), str(device),
                 float(args.loss_workspace_mb), int(model.embedding_spot.shape[1]))
    shards = cache_state.setdefault("cpu_shards", {})
    key = (int(bundle.code), int(index))
    cached = shards.get(key)
    if cached is not None and cached["signature"] == signature:
        return cached
    if any(value.device.type != "cpu" for value in source):
        raise ValueError("sampler indices must remain on CPU")
    users, offset_positive, offset_negative = source
    negative = offset_negative - int(data.num_spot)
    same_graph = (cached is not None and adjacency is cached['batch'].adjacency
                  and signature[1] == cached['signature'][1])
    stable_positive = same_graph and signature[0][:2] == cached['signature'][0][:2]
    if same_graph and not stable_positive:
        # Detect in-place mutation even when cached users alias the source.
        for field_index in (0, 1):
            prior_key = cached['signature'][0][field_index]
            if source[field_index] is cached['source'][field_index] and source[field_index]._version != prior_key[1]:
                raise ValueError('positive edges changed without rebuilding the shard adjacency')
    positive = cached['positive'] if stable_positive else offset_positive - int(data.num_spot)
    if same_graph and not stable_positive:
        if not torch.equal(users, cached['users']) or not torch.equal(positive, cached['positive']):
            raise ValueError('positive edges changed without rebuilding the shard adjacency')
        stable_positive = True
    if stable_positive:
        users, positive = cached['users'], cached['positive']
    context = (int(data.num_spot), int(data.num_peak), str(dtype), len(users))
    reusable_counts = (stable_positive and cached.get('coefficient_context') == context)
    validate_training_indices(users, positive, negative, int(data.num_spot),
                              int(data.num_peak), check_range=not reusable_counts)
    if reusable_counts and (negative.min() < 0 or negative.max() >= int(data.num_peak)):
        raise ValueError('training index outside local node universe')
    group = int(args.pn_r)
    if len(users) % group:
        raise ValueError("shard contains an incomplete positive-edge group")
    grouped_users, grouped_positive = users.reshape(-1, group), positive.reshape(-1, group)
    if not reusable_counts and (not torch.all(grouped_users == grouped_users[:, :1])
            or not torch.all(grouped_positive == grouped_positive[:, :1])):
        raise ValueError("pn_r triplets must be contiguous repeated user-positive groups")
    if torch.any(positive == negative):
        raise ValueError("negative index equals its positive index")
    scalar_weight = torch.full((1,), 1.0 / len(users), dtype=dtype)[0]
    if reusable_counts:
        spot_coefficients = cached['coefficients'].spot
        positive_coefficients = cached['positive_coefficients']
    else:
        spot_coefficients = torch.bincount(users.long(), minlength=int(data.num_spot)).to(dtype) * scalar_weight
        positive_coefficients = torch.bincount(positive.long(), minlength=int(data.num_peak)).to(dtype) * scalar_weight
    negative_coefficients = torch.bincount(negative.long(), minlength=int(data.num_peak)).to(dtype) * scalar_weight
    coefficients = EgoL2Coefficients(spot=spot_coefficients,
                                   peak=positive_coefficients + negative_coefficients)
    plan = execution_plan(
        len(users), model.embedding_spot.shape[1], model.embedding_spot.element_size(),
        backend="auto", workspace_mb=float(args.loss_workspace_mb),
        group_size=group, chunk_size=0, device=device, dtype=dtype,
        allow_fused=True)
    cached = {"signature": signature, "source": source, "users": users,
              "positive": positive, "negative": negative,
              "coefficients": coefficients, "plan": plan,
              "positive_coefficients": positive_coefficients,
              "coefficient_context": context,
              "batch": ShardBatch(users, positive, negative, adjacency),
              "device_coefficients": None}
    shards[key] = cached
    return cached


def _prepare_epoch(bundles, data, model, device, args, cache_state):
    prepared, indices = [], {}
    for bundle in bundles:
        for index in range(int(bundle.num_shards)):
            indices[int(bundle.code), index] = len(prepared)
            prepared.append(_cpu_batch(bundle, index, data, model, device, args, cache_state))
    source = [entry['batch'] for entry in prepared]
    previous = cache_state.get('source_batches')
    if previous is not None and len(previous) == len(source) and all(
            old is new for old, new in zip(previous, source)):
        source = previous
    cache_state['source_batches'] = source
    resident = cache_state.get('resident')
    if resident is None:
        resident = ResidentBatchCache(enabled=bool(args.resident_cache),
            budget_gib=float(args.resident_cache_budget_gib),
            reserve_gib=float(args.resident_cache_reserve_gib))
        cache_state['resident'] = resident
    device_batches = resident.prepare(source, device)
    return prepared, indices, source, device_batches, resident


def train_epoch(data, model, optimizer, device, args, epoch, cache_state) -> dict:
    """Accumulate equally weighted local losses before each shared Adam step.

    Production uses 20 groups. Smaller budgets support the CPU contract check.
    ``data.maybe_refresh_slice_shards(epoch)`` must be called by the outer loop.
    """
    device = torch.device(device)
    budget = int(args.balanced_updates_per_slice)
    if budget < 1 or epoch < 0:
        raise ValueError("update budget must be positive and epoch nonnegative")
    bundles = list(data.slice_bundles)
    if len(bundles) < 2 or len({int(b.code) for b in bundles}) != len(bundles):
        raise ValueError("multislice requires distinct biological slices")
    if model.num_spot != data.num_spot or model.num_peak != data.num_peak:
        raise ValueError("model and loader global node axes differ")
    if any(int(b.num_shards) < 1 or len(b.batch_all) != int(b.num_shards)
           or len(b.adjacencies) != int(b.num_shards) for b in bundles):
        raise ValueError("slice shard counts, batches and adjacencies differ")
    selected = {int(b.code): selected_shard_indices(
        b, epoch, budget, strategy=str(args.shard_pairing_strategy), seed=int(args.seed))
        for b in bundles}
    # Native residency changes placement only; it never chooses the update order.
    prepared_batches, batch_indices, source, resident_batches, resident = _prepare_epoch(
        bundles, data, model, device, args, cache_state)
    spatial_cache = cache_state.setdefault("spatial", {})
    spatial = {}
    for bundle in bundles:
        code = int(bundle.code)
        use = epoch >= int(args.pre_train_epochs) and bundle.spatial_adjacency is not None
        if use:
            old = spatial_cache.get(code)
            if (old is None or old[0] is not bundle.spatial_adjacency
                    or old[1].device != device):
                old = (bundle.spatial_adjacency, bundle.spatial_adjacency.to(device))
                spatial_cache[code] = old
            spatial[code] = old[1]
        else:
            spatial[code] = None
    totals = {int(b.code): {"name": str(b.name), "loss": 0.0, "bpr": 0.0,
                           "reg": 0.0, "reg_dna": 0.0, "triplet": 0.0,
                           "participations": 0, "triplets": 0,
                           "positive_edges": 0, "time_seconds": 0.0,
                           "backend_counts": Counter()}
              for b in bundles}
    backend_counts = Counter()
    group_grad_norms_device = torch.empty(budget, device=device)
    group_slice_order = []
    loss_names = ('loss', 'bpr', 'reg', 'reg_dna', 'triplet')
    gpu_totals = {int(b.code): torch.zeros(len(loss_names), device=device, dtype=torch.float64)
                  for b in bundles}
    started = time.perf_counter()
    model.train()
    slice_scale = 1.0 / len(bundles)
    for group_index in range(budget):
        optimizer.zero_grad(set_to_none=True)
        group_order = list(data.ordered_slice_bundles(epoch + group_index))
        if {int(b.code) for b in group_order} != set(totals) or len(group_order) != len(bundles):
            raise ValueError("group order must contain every slice exactly once")
        group_slice_order.append([int(b.code) for b in group_order])
        group_prepared = []
        group_coefficients = []
        for bundle in group_order:
            selected_index = selected[int(bundle.code)][group_index]
            prepared = prepared_batches[batch_indices[int(bundle.code), selected_index]]
            coefficients = prepared['device_coefficients']
            if coefficients is None:
                coefficients = prepared['coefficients'].to(device=device)
                if resident_batches is not None:
                    prepared['device_coefficients'] = coefficients
            if resident_batches is None:
                prepared['device_coefficients'] = None
            group_prepared.append(prepared)
            group_coefficients.append(coefficients)
        reg_group, reg_by_slice = group_regularization(model, group_coefficients)
        finite = torch.isfinite(reg_group.detach())
        for position, bundle in enumerate(group_order):
            code = int(bundle.code)
            index = selected[code][group_index]
            batch_index = batch_indices[code, index]
            prepared = group_prepared[position]
            slice_started = time.perf_counter()
            batch = (resident_batches[batch_index] if resident_batches is not None
                     else source[batch_index].to(device))
            users, positive, negative, adjacency = batch.users, batch.positive, batch.negative, batch.adjacency
            use_spatial = spatial[code] is not None
            r_spatial = (float(data.r_by_slice[code]) * float(args.r_spatial)
                         if use_spatial else 0.0)
            bpr = bpr_only(model, users, positive, negative, adjacency,
                           spatial[code], use_spatial, r_spatial, prepared['plan'], int(args.pn_r))
            reg = reg_by_slice[position]
            loss = bpr.detach() + float(args.decay) * reg
            finite.logical_and_(torch.isfinite(loss))
            objective = bpr * slice_scale
            if position == len(group_order) - 1:
                objective = objective + float(args.decay) * reg_group
            objective.backward()
            row = totals[code]
            with torch.no_grad():
                zero = loss.new_zeros(())
                gpu_totals[code].add_(torch.stack((loss, bpr.detach(), reg, zero, zero)).to(torch.float64))
            backend = prepared["plan"]["effective_backend"]
            row["backend_counts"][backend] += 1
            backend_counts[backend] += 1
            row["participations"] += 1
            row["triplets"] += len(users)
            row["positive_edges"] += len(users) // int(args.pn_r)
            row["time_seconds"] += time.perf_counter() - slice_started
            del loss, bpr, objective, users, positive, negative, adjacency, batch
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip)
        finite.logical_and_(torch.isfinite(grad_norm))
        if not bool(finite):
            raise FloatingPointError("nonfinite loss or global gradient norm before Adam step")
        group_grad_norms_device[group_index] = grad_norm.detach()
        optimizer.step()
        del reg_group, reg_by_slice, group_coefficients, group_prepared, coefficients
    group_grad_norms = group_grad_norms_device.cpu().tolist()
    for code, row in totals.items():
        for name, value in zip(loss_names, gpu_totals[code].cpu().tolist()):
            row[name] = value / row['participations']
        row["backend_counts"] = dict(row["backend_counts"])
        row["selected_shards"] = selected[code]
    return {
        "epoch": int(epoch), "completed_epoch": int(epoch) + 1,
        "optimizer_steps": budget, "propagation_steps": budget * len(bundles),
        "loss": float(np.mean([row["loss"] for row in totals.values()])),
        "bpr": float(np.mean([row["bpr"] for row in totals.values()])),
        "reg": float(np.mean([row["reg"] for row in totals.values()])),
        "per_slice": totals, "selected_shards": selected,
        "group_slice_order": group_slice_order,
        "backend_counts": dict(backend_counts),
        "resident_cache": copy.deepcopy(resident.stats),
        "grad_norm_mean_before_clip": float(np.mean(group_grad_norms)),
        "grad_norm_max_before_clip": max(group_grad_norms),
        "elapsed_seconds": time.perf_counter() - started,
    }
