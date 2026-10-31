"""Explicit isolated training, optimization, state-only checkpoints and export."""

from dataclasses import asdict
from .data import TrainingRun
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
import copy
from .utils import write_json, seal_artifact, environment
from .metrics import binary_metrics, recall_at_k
from .checkpoint import seed_all,restore_rng,read_checkpoint,save_checkpoint
from .exports import infer_cpu,export_embeddings
from .artifacts import validated_operation,stage_metadata


def train_SEAREG(network, config, output, *, execution=None, resume_from=None):
    """Train synchronously in an isolated worker and return a TrainingRun.

    Accept TrainingInputs or a compatible prepared network and a TrainConfig.
    ``output`` must be unused; ``resume_from`` selects a compatible full-state
    checkpoint. For an independent tmux job, use ``start_training_job``.
    """
    from ._runtime import run_worker
    from .compat import resolve_training_inputs
    if Path(output).exists():raise FileExistsError(output)
    inputs=resolve_training_inputs(network,config,output)
    execution = dict(execution or {"physical_gpu": None, "torch_threads": 1,
                     "torch_interop_threads": 1, "blas_threads": 1,
                     "strict_algorithms": True, "tf32": False})
    run_worker("train", {"prepared": str(inputs.path), "config": asdict(config),
               "resume_from": str(resume_from) if resume_from is not None else None},
               output, execution)
    return TrainingRun.load(output)






def _validate_training_support(data, config, full, spatial, dna, rs, rd):
    """Check the actual fixed training graph once before optimizer creation.

    Raw empty spots have no defined count-based scale in the retained protocol.
    No node is dropped and no held-out edge is moved into the training graph.
    """
    empty = np.flatnonzero(data.X.getnnz(axis=1) == 0)
    if empty.size:
        raise ValueError(f"empty count spots {[data.spot_ids[i] for i in empty[:10]]}; "
                         "supply counts or explicitly exclude them before preprocessing")

    def support(graph, n):
        result = np.zeros(n, dtype=bool)
        if graph is not None:
            graph = graph.coalesce().cpu()
            result[graph.indices()[0, graph.values() > 0].numpy()] = True
        return result

    ns, npk = data.X.shape
    active = support(full, ns + npk)
    if config.epochs > config.pre_train_epochs and rs > 0:
        active[:ns] |= support(spatial.adjacency, ns)
    motif_active = np.zeros(len(data.dna_ids), dtype=bool)
    if dna is not None:
        dna_active = support(dna, npk + len(data.dna_ids))
        if rd > 0:
            active[ns:] |= dna_active[:npk]
        # Motif messages are computed from peak ego vectors even when rd=0.
        motif_active = dna_active[npk:]
    for kind, ids, mask in (("spot", data.spot_ids, active[:ns]),
                            ("peak", data.peak_ids, active[ns:]),
                            ("motif", data.dna_ids, motif_active)):
        missing = np.flatnonzero(~mask)
        if missing.size:
            raise ValueError(f"{kind} nodes {[ids[i] for i in missing[:10]]} have no effective "
                             "final-epoch propagation support in the fixed training graph")
    return {"protocol": "fixed_training_graph_final_support_v1",
            "spots": ns, "peaks": npk, "motifs": len(data.dna_ids),
            "final_spatial_active": bool(config.epochs > config.pre_train_epochs and rs > 0)}


def _execution_provenance(torch, device, execution=None):
    """Capture and verify the numerical runtime separately from transfer policy."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    defaults = {
        "physical_gpu": None,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "blas_threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
        "strict_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32": False,
    }
    requested = dict(defaults)
    if execution is not None:
        unknown = set(execution) - set(defaults)
        if unknown:
            raise ValueError(f"unknown execution options: {sorted(unknown)}")
        requested.update(execution)
    if requested["physical_gpu"] is not None and (
        type(requested["physical_gpu"]) is not int or requested["physical_gpu"] < 0
    ):
        raise ValueError("physical_gpu must be null or a nonnegative integer")
    for name in ("torch_threads", "torch_interop_threads", "blas_threads"):
        if type(requested[name]) is not int or requested[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("strict_algorithms", "tf32"):
        if type(requested[name]) is not bool:
            raise ValueError(f"{name} must be boolean")

    strict = torch.are_deterministic_algorithms_enabled()
    if strict != requested["strict_algorithms"]:
        raise ValueError("requested strict_algorithms differs from the active runtime")
    if torch.get_num_threads() != requested["torch_threads"]:
        raise ValueError("requested torch_threads differs from the active runtime")
    if torch.get_num_interop_threads() != requested["torch_interop_threads"]:
        raise ValueError("requested torch_interop_threads differs from the active runtime")
    blas_environment = {
        name: os.environ.get(name)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
    }
    if execution is not None and any(
        value != str(requested["blas_threads"])
        for value in blas_environment.values()
    ):
        raise ValueError("requested blas_threads differs from the active runtime")

    gpu_name = None
    gpu_uuid = None
    compute_capability = None
    total_memory = None
    matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    if device.type == "cuda":
        if requested["physical_gpu"] is None:
            raise ValueError("CUDA training requires a recorded physical_gpu")
        if matmul_tf32 != requested["tf32"] or cudnn_tf32 != requested["tf32"]:
            raise ValueError("requested tf32 differs from the active CUDA runtime")
        properties = torch.cuda.get_device_properties(device)
        gpu_name = properties.name
        uuid = getattr(properties, "uuid", None)
        gpu_uuid = None if uuid is None else str(uuid)
        compute_capability = list(torch.cuda.get_device_capability(device))
        total_memory = int(properties.total_memory)
    elif requested["physical_gpu"] is not None:
        raise ValueError("physical_gpu cannot be set for CPU training")

    effective = {
        "logical_device": str(device),
        "physical_gpu": requested["physical_gpu"],
        "cuda_visible_devices": visible,
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "compute_capability": compute_capability,
        "gpu_total_memory_bytes": total_memory,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "blas_environment": blas_environment,
        "strict_algorithms": strict,
        "matmul_allow_tf32": matmul_tf32,
        "cudnn_allow_tf32": cudnn_tf32,
    }
    numerical = {
        key: effective[key]
        for key in (
            "logical_device",
            "physical_gpu",
            "gpu_name",
            "gpu_uuid",
            "compute_capability",
            "torch_threads",
            "torch_interop_threads",
            "blas_environment",
            "strict_algorithms",
            "matmul_allow_tf32",
            "cudnn_allow_tf32",
        )
    }
    return {"requested": requested, "effective": effective, "numerical": numerical}






def _scales(data, sampler, spatial, dna_degree, config):
    import torch
    spot_degree = torch.tensor(data.X.getnnz(axis=1), dtype=torch.float32)
    peak_degree = torch.tensor(data.X.getnnz(axis=0), dtype=torch.float32)
    rs = (
        torch.sqrt(spatial.raw_degree.clamp_min(1e-8) / spot_degree.clamp_min(1e-8))
        .clamp_min(1e-8)
        .mean()
        if spatial.adjacency is not None
        else torch.tensor(0.0, dtype=torch.float32)
    )
    rd = (
        torch.sqrt(dna_degree.clamp_min(1e-8) / peak_degree.clamp_min(1e-8)).mean()
        if dna_degree is not None
        else torch.tensor(0.0, dtype=torch.float32)
    )
    # This condition is part of the legacy graph-scale protocol, not a new ablation rule.
    if spatial.adjacency is not None and config.tr < 1 - 1e-6:
        train_degree = torch.bincount(
            torch.from_numpy(sampler.train_edges[0]), minlength=len(spot_degree)
        ).float()
        mask = spot_degree > 0
        fraction = (train_degree[mask].mean() / spot_degree[mask].mean()).item()
        multiplier = torch.sqrt(torch.tensor(fraction, dtype=torch.float32))
        rs = rs * multiplier
        rd = rd * multiplier
    return float(rs * config.spatial_weight), float(rd * config.dna_weight)


def _evaluate(values, sampler, config):
    import torch
    u, p, _ = values
    test = sampler.test_triplets
    if test.shape[1] == 0:
        empty = torch.empty(0)
        result = binary_metrics(empty, empty)
        result["evaluation_population"] = "fixed_sampled_heldout_triplets"
        result["recall_candidate_protocol"] = config.recall_candidates
        # Validate the embedding matrices and candidate protocol without building
        # known-positive sets from every training edge when there are no queries.
        result["recall_at_k"] = recall_at_k(
            u, p, test[:2], test[:2], candidates=config.recall_candidates
        )
        return result
    positive = []
    negative = []
    for start in range(0, test.shape[1], 65536):
        user, pos, neg = [torch.from_numpy(x) for x in test[:, start : start + 65536]]
        positive.append((u[user] * p[pos]).sum(1))
        negative.append((u[user] * p[neg]).sum(1))
    pos = torch.cat(positive) if positive else torch.empty(0)
    neg = torch.cat(negative) if negative else torch.empty(0)
    result = binary_metrics(pos, neg)
    result["evaluation_population"] = "fixed_sampled_heldout_triplets"
    result["recall_candidate_protocol"] = config.recall_candidates
    result["recall_at_k"] = recall_at_k(
        u, p, test[:2], sampler.train_edges, candidates=config.recall_candidates
    )
    return result


def _grouped_loss_size(batches, config):
    """Validate the sampler's contiguous pn_r blocks before grouped gathers."""
    if not config.grouped_loss or config.pn_r <= 1:
        return 0
    group = config.pn_r
    for index, batch in enumerate(batches):
        if len(batch.users) % group:
            raise ValueError(
                f"grouped loss batch {index} is not a multiple of pn_r={group}"
            )
        users = batch.users.reshape(-1, group)
        positive = batch.positive.reshape(-1, group)
        if not users.equal(users[:, :1].expand_as(users)) or not positive.equal(
            positive[:, :1].expand_as(positive)
        ):
            raise ValueError(
                f"grouped loss batch {index} does not contain contiguous repeated "
                "(user, positive) blocks"
            )
    return group


def _loss_execution_metadata(
    config,
    group_size,
    planned_max_chunk_triplets,
    actual_chunk_histogram,
    dna_weight_precomputed,
    backend_counts=None,
):
    fused_active = bool(backend_counts and backend_counts.get("cuda_fused"))
    if config.weighted_ego_l2 or fused_active:
        coefficient_backend = "uniform_integer_bincount_nodewise_v1"
    else:
        coefficient_backend = "triplet_gather_v1"
    if fused_active:
        implementation = ("cuda_fused_sampled_dot_v1" if len(backend_counts) == 1
                          else "mixed_sampled_scoring_v1")
    elif config.loss_backend == "auto":
        implementation = "torch_auto_eager_or_checkpoint_v1"
    elif config.loss_backend == "torch_streamed":
        implementation = "torch_checkpoint_streamed_v1"
    elif config.loss_chunk_size:
        implementation = "torch_checkpoint_explicit_chunk_v1"
    else:
        implementation = "torch_eager_v1"
    return {
        "requested_backend": config.loss_backend,
        "effective_backend": (
            next(iter(backend_counts)) if backend_counts and len(backend_counts) == 1
            else "mixed" if backend_counts else config.loss_backend
        ),
        "backend_step_counts": dict(backend_counts or {}),
        "implementation": implementation,
        "workspace_mb": config.loss_workspace_mb,
        "planned_max_chunk_triplets_observed": sorted(planned_max_chunk_triplets),
        "actual_chunk_triplet_histogram": {
            str(size): actual_chunk_histogram[size]
            for size in sorted(actual_chunk_histogram)
        },
        "grouped_shared_gather": group_size > 1 and not fused_active,
        "fused_gradient_tile_reduce": fused_active,
        "fused_tile_triplets": 32 if fused_active else None,
        "group_size": group_size,
        "reduction_order_change": (
            "fused_sampled_dot_flat_weighted_sum_v1" if fused_active else
            "grouped_shared_gather_and_chunk_scalar_sum_v1"
            if group_size > 1 and implementation != "torch_eager_v1"
            else "grouped_shared_gather_v1"
            if group_size > 1
            else "chunk_scalar_sum_v1"
            if implementation != "torch_eager_v1"
            else "none"
        ),
        "weighted_ego_l2": config.weighted_ego_l2 or fused_active,
        "ego_regularization_execution": coefficient_backend,
        "dna_weight_precomputed": dna_weight_precomputed,
        "dense_spot_peak_matrix": False,
    }


@validated_operation
def _train(prepared, config, output, resume_from=None, *, execution=None):
    import torch
    from tqdm import tqdm
    from .graph import GraphResult
    from .sampling import EpochSampler
    from .model import LightGCN
    from .model import (compute_loss, clip_motif_rows, prepare_dna_regularization_weight,
                        validate_training_indices, batch_edge_weights)
    from .loss import execution_plan, prepare_ego_l2_coefficients
    from .resident import ResidentBatchCache
    from .compat import resolve_training_inputs
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    inputs=resolve_training_inputs(prepared,config,output)
    network=inputs.network();data=network.load_data()
    data_fingerprint=network.metadata['data_fingerprint']
    device = torch.device(config.device)
    if device.type == "cuda" and (
        not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()
    ):
        raise ValueError(f"requested CUDA device is unavailable: {config.device}")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("only explicitly verified CPU/CUDA devices are supported")
    execution_record = _execution_provenance(torch, device, execution)
    if config.loss_backend in ("auto", "cuda_fused"):
        from .scoring import fused_support
        supported, reason, scoring_runtime = fused_support(device, torch.float32, config.hidden_dim)
        if config.loss_backend == "cuda_fused" and not supported:
            raise ValueError(reason)
        if supported:
            execution_record["effective"]["sampled_scoring"] = scoring_runtime
            execution_record["numerical"]["sampled_scoring"] = {
                name: value for name, value in scoring_runtime.items() if name != "compiler"}
        else:
            execution_record["effective"]["scoring_fallback_reason"] = reason
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def memory_record():
        return {
            "scope": "PyTorch allocator since training setup; excludes non-PyTorch allocations",
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            "process_peak_vram_bytes": None,
            "process_peak_evidence": "not_measured_by_allocator_counters",
        }
    static=network.load_graphs()
    spatial=GraphResult(static['spatial'],static['spatial_degree'],{'enabled':network.metadata['has_spatial']})
    sampler=EpochSampler(data,config,split_state=inputs.load_split())
    dna,peak_degree,dna_degree=static['dna'],static['peak_degree'],static['motif_degree']
    full=inputs.load_full_graph()
    rs, rd = _scales(data, sampler, spatial, peak_degree, config)
    propagation_support = _validate_training_support(data, config, full, spatial, dna, rs, rd)
    checkpoint = (
        read_checkpoint(resume_from,data,config,execution_record,training_inputs=inputs,data_fingerprint=data_fingerprint)
        if resume_from
        else None
    )
    seed_all(config.seed)
    model = LightGCN(*data.X.shape, len(data.dna_ids), config.hidden_dim).to(device)
    dna_reg_weight = (
        prepare_dna_regularization_weight(
            dna_degree,
            config.dna_reg_alpha,
            device=model.embedding_kmer.device,
            dtype=model.embedding_kmer.dtype,
        )
        if model.embedding_kmer is not None and dna_degree is not None
        else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0)
    resident = ResidentBatchCache(
        enabled=config.resident_cache,
        budget_gib=config.resident_cache_budget_gib,
        reserve_gib=config.resident_cache_reserve_gib,
    )
    group_size = 0
    grouping_validated = False
    planned_max_chunk_triplets = set()
    actual_chunk_histogram = {}
    backend_counts = {}
    cached_source_batches = None
    batch_plans = []
    batch_ego_coefficients = []
    history = []
    completed = 0
    steps = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != "step":
                    state[key] = value.to(device)
        completed = checkpoint["completed_epochs"]
        steps = checkpoint["optimizer_steps"]
        history = checkpoint["history"]
        restore_rng(checkpoint["rng_state"])
    output.mkdir(parents=True, exist_ok=False)
    progress = None
    try:
        (output / "checkpoints").mkdir()
        (output / "embeddings").mkdir()
        write_json(output / "config.json", asdict(config))
        write_json(
            output / "data.json",
            {
                "spot_ids": data.spot_ids,
                "peak_ids": data.peak_ids,
                "dna_ids": data.dna_ids,
                "dna_kind": data.dna_kind,
                "slice_ids": data.slice_ids,
                "provenance": data.provenance,
                "data_fingerprint": data_fingerprint,
            },
        )
        if data.coordinates is not None:
            np.save(output / "coordinates.npy", data.coordinates, allow_pickle=False)
        history_path = output / "training_history.tsv"
        history_written = bool(history)
        if history_written:
            pd.DataFrame(history).to_csv(history_path, sep="\t", index=False)
        input_refs=stage_metadata(output,upstreams={'training_inputs':inputs})
        write_json(output/'inputs.json',input_refs)
        lineage = (
            None
            if resume_from is None
            else {
                "checkpoint": str(Path(resume_from).resolve()),
                "sha256": checkpoint['_source_sha256'],
                "completed_epochs": completed,
            }
        )
        write_json(
            output / "status.json",
            {
                "state": "running",
                "pid": os.getpid(),
                "verified_completed_epochs": completed,
                "execution": execution_record,
            },
        )
        dna_device = dna.to(device) if dna is not None else None
        spatial_device = spatial.adjacency.to(device) if spatial.adjacency is not None else None
        progress = tqdm(
            range(completed, config.epochs), total=config.epochs, initial=completed,
        )
        loss_keys = ("loss", "bpr", "reg", "reg_dna")
        for epoch in progress:
            epoch_started = time.perf_counter()
            use_spatial = epoch >= config.pre_train_epochs and spatial_device is not None
            source_batches = sampler.at_epoch(epoch)
            if not grouping_validated:
                group_size = _grouped_loss_size(source_batches, config)
                grouping_validated = True
            if source_batches is not cached_source_batches:
                # The sampler owns immutable CPU batches until a negative refresh
                # or reshard. Validate before upload, not on every cached GPU step.
                for source_batch in source_batches:
                    validate_training_indices(source_batch.users, source_batch.positive,
                        source_batch.negative, model.num_spot, model.num_peak)
                batch_plans = [execution_plan(len(batch.users), config.hidden_dim,
                    model.embedding_spot.element_size(), config.loss_backend,
                    config.loss_workspace_mb, group_size, config.loss_chunk_size,
                    device=device, dtype=model.embedding_spot.dtype,
                    allow_fused=True)
                    for batch in source_batches]
                batch_ego_coefficients = [None] * len(source_batches)
                cached_source_batches = source_batches
            for source_batch, plan in zip(source_batches, batch_plans):
                backend = plan["effective_backend"]
                backend_counts[backend] = backend_counts.get(backend, 0) + 1
                planned_step = plan["chunk_triplets"]
                planned_max_chunk_triplets.add(planned_step)
                full_chunks, tail = divmod(len(source_batch.users), planned_step)
                if full_chunks:
                    actual_chunk_histogram[planned_step] = actual_chunk_histogram.get(planned_step, 0) + full_chunks
                if tail:
                    actual_chunk_histogram[tail] = actual_chunk_histogram.get(tail, 0) + 1
            batches = resident.prepare(source_batches, device)
            totals = torch.zeros(len(loss_keys), dtype=torch.float64, device=device)
            model.train()
            for batch_index,batch in enumerate(resident.iter_batches(source_batches,batches,device)):
                users, pos, neg = batch.users, batch.positive, batch.negative
                optimizer.zero_grad()
                ego_coefficients = batch_ego_coefficients[batch_index]
                if (config.weighted_ego_l2 or batch_plans[batch_index]["effective_backend"] == "cuda_fused") and ego_coefficients is None:
                    weights = batch_edge_weights(users, model.num_spot,
                        dtype=model.embedding_spot.dtype)
                    ego_coefficients = prepare_ego_l2_coefficients(
                        users, pos, neg, weights, n_spots=model.num_spot,
                        n_peaks=model.num_peak, assume_uniform=True)
                    # Streaming must not quietly accumulate all-shard CUDA state
                    # outside the resident budget. Keep only its active pair on GPU.
                    batch_ego_coefficients[batch_index] = (
                        ego_coefficients if batches is not None else ego_coefficients.to("cpu"))
                    del weights
                if ego_coefficients is not None and batches is None:
                    ego_coefficients = ego_coefficients.to(device)
                loss, parts = compute_loss(
                    model,
                    users,
                    pos,
                    neg,
                    batch.adjacency,
                    dna_device,
                    spatial_device,
                    use_spatial,
                    rs,
                    rd,
                    config.decay,
                    config.dna_decay,
                    dna_degree,
                    config.dna_reg_alpha,
                    config.loss_chunk_size,
                    group_size=group_size,
                    loss_backend=config.loss_backend,
                    loss_workspace_mb=config.loss_workspace_mb,
                    weighted_ego_l2=config.weighted_ego_l2,
                    ego_coefficients=ego_coefficients,
                    dna_reg_weight=dna_reg_weight,
                    validate_index_range=False,
                    loss_plan=batch_plans[batch_index],
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                clip_motif_rows(model, config.motif_cap_factor)
                steps += 1
                with torch.no_grad():
                    totals.add_(
                        torch.stack(
                            (loss.detach(), *(parts[key].detach() for key in loss_keys[1:]))
                        ).to(dtype=torch.float64)
                    )
            completed = epoch + 1
            values = totals.cpu().tolist()
            if not np.isfinite(values).all():
                raise FloatingPointError(f"non-finite training values at epoch {completed}")
            row = {
                key: value / len(source_batches)
                for key, value in zip(loss_keys, values)
            }
            row.update(
                epoch=completed,
                optimizer_steps=steps,
                nonempty_shards=len(source_batches),
                epoch_seconds_excluding_export=time.perf_counter() - epoch_started,
            )
            history.append(row)
            pd.DataFrame([row]).to_csv(history_path, sep="\t", index=False,
                                      mode="a", header=not history_written)
            history_written = True
            print(
                f"EPOCH epoch={completed} optimizer_steps={steps} "
                + " ".join(f"{key}={row[key]:.9g}" for key in loss_keys)
                + f" seconds={row['epoch_seconds_excluding_export']:.6f}",
                file=sys.stderr, flush=True,
            )
            write_json(
                output / "status.json",
                {
                    "state": "running",
                    "pid": os.getpid(),
                    "verified_completed_epochs": completed,
                    "optimizer_steps": steps,
                    "execution": execution_record,
                    "loss_execution": _loss_execution_metadata(
                        config,
                        group_size,
                        planned_max_chunk_triplets,
                        actual_chunk_histogram,
                        dna_reg_weight is not None,
                        backend_counts,
                    ),
                    "resident_cache": copy.deepcopy(resident.stats),
                },
            )
            export_due = config.result_export_due(completed)
            if export_due:
                values = infer_cpu(model, full, dna, spatial.adjacency, use_spatial, rs, rd)
                export_embeddings(
                    output / "embeddings" / f"{completed}_epoch",
                    data,
                    model,
                    values,
                    completed,
                    data_fingerprint=data_fingerprint, final=completed == config.epochs,
                )
                write_json(
                    output / f"evaluation_epoch_{completed:04d}.json",
                    _evaluate(values, sampler, config),
                )
            if completed % config.checkpoint_every == 0 or export_due:
                rolling = output / "checkpoints" / "latest.pt"
                save_checkpoint(
                    rolling,
                    model,
                    optimizer,
                    data,
                    config,
                    completed,
                    steps,
                    sampler,
                    history,
                    execution_record,
                    training_inputs=inputs,data_fingerprint=data_fingerprint,
                )
                if export_due:
                    import shutil
                    destination = output / "checkpoints" / f"epoch_{completed:04d}.pt"
                    temporary = destination.with_suffix(".pt.tmp")
                    shutil.copyfile(rolling, temporary)
                    os.replace(temporary, destination)
                print(f"CHECKPOINT epoch={completed} path={rolling}", file=sys.stderr, flush=True)
            if os.environ.get("SEAREG_JOB_PATH"):
                from .jobs import record_progress
                record_progress(output, row, config)
        metadata = {
            "configured_epochs": config.epochs,
            "verified_completed_epochs": completed,
            "final_adopted_output_epoch": completed,
            "optimizer_steps": steps,
            "data_fingerprint": data_fingerprint,
            "training_inputs_fingerprint":inputs.fingerprint,
            "model_network_fingerprint":network.fingerprint,
            "propagation_support": propagation_support,
            "effective_r_spatial": rs,
            "effective_r_dna": rd,
            "graph_scale_protocol": "copy1_conditional_spatial_v1",
            "model_protocol": "copy1_two_layer_sum_v1",
            "loss_protocol": "copy1_sampled_bpr_v1",
            "loss_objective_protocol": "copy1_sampled_bpr_weighted_regularization_v1",
            "loss_chunk_size": config.loss_chunk_size,
            "loss_group_size": group_size,
            "loss_execution": _loss_execution_metadata(
                config,
                group_size,
                planned_max_chunk_triplets,
                actual_chunk_histogram,
                dna_reg_weight is not None,
                backend_counts,
            ),
            "execution": execution_record,
            "batch_transfer": {
                "requested": {
                    "resident_cache": config.resident_cache,
                    "resident_cache_budget_gib": config.resident_cache_budget_gib,
                    "resident_cache_reserve_gib": config.resident_cache_reserve_gib,
                },
                "effective": copy.deepcopy(resident.stats),
            },
            "resident_cache": copy.deepcopy(resident.stats),
            "export_inference_device": "cpu",
            "torch_memory": memory_record(),
            "training_wall_seconds_including_setup_and_exports": time.perf_counter() - started,
            "parent": lineage,
            "config": asdict(config),
            "environment": environment(),
        }
        metadata.update(input_refs)
        write_json(output / "status.json", {"state": "complete", **metadata})
        seal_artifact(output, "training_run", metadata)
        print(f"COMPLETE epochs={completed} optimizer_steps={steps} output={output}", file=sys.stderr, flush=True)
    except BaseException as exc:
        try:
            write_json(
                output / "status.json",
                {
                    "state": "failed",
                    "verified_completed_epochs": completed,
                    "optimizer_steps": steps,
                    "execution": execution_record,
                    "loss_execution": _loss_execution_metadata(
                        config,
                        group_size,
                        planned_max_chunk_triplets,
                        actual_chunk_histogram,
                        dna_reg_weight is not None,
                        backend_counts,
                    ),
                    "batch_transfer": {
                        "requested": {
                            "resident_cache": config.resident_cache,
                            "resident_cache_budget_gib": config.resident_cache_budget_gib,
                            "resident_cache_reserve_gib": config.resident_cache_reserve_gib,
                        },
                        "effective": copy.deepcopy(resident.stats),
                    },
                    "error": str(exc),
                },
            )
        except Exception as status_error:
            # Preserve the scientific/IO failure even if disk permissions or
            # capacity also prevent recording its status. Python 3.11+ can
            # attach the secondary diagnostic without replacing the exception.
            if hasattr(exc, "add_note"):
                exc.add_note(f"Could not record failed training status: {status_error}")
        raise
    finally:
        if progress is not None:
            progress.close()
    return output
