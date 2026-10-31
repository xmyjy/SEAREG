"""State-only checkpoint, numerical identity and random-state preservation."""
from dataclasses import asdict
from pathlib import Path
import copy
import os
import random
import numpy as np
from .utils import environment,write_json,identity,file_sha256


def training_compatibility_id():
    """Keep executable source compatibility separate from exact source provenance."""
    import ast
    from .artifacts import _STAGE_MODULES

    class WithoutDocstrings(ast.NodeTransformer):
        def strip(self, node):
            self.generic_visit(node)
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:]
            return node

        visit_Module = visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = strip

    root = Path(__file__).parent
    return identity({name: ast.dump(WithoutDocstrings().visit(ast.parse((root/name).read_text())),
                                    include_attributes=False)
                     for name in _STAGE_MODULES['training']})

def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    import torch
    name, keys, pos, gauss, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (name, keys.tolist(), pos, gauss, cached),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng(state):
    import torch
    random.setstate(state["python"])
    name, keys, pos, gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG topology changed")
        torch.cuda.set_rng_state_all(state["cuda"])


def trajectory(config):
    values = asdict(config)
    values.pop("epochs")
    for name in (
        "resident_cache",
        "resident_cache_budget_gib",
        "resident_cache_reserve_gib",
        "export_every",
        "export_start_epoch",
        "checkpoint_every",
    ):
        values.pop(name)
    return values
def read_checkpoint(path, data, config, execution_provenance, *, training_inputs=None,data_fingerprint=None):
    import torch
    source_identity=file_sha256(path)
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if file_sha256(path)!=source_identity:raise ValueError('checkpoint source changed while loading')
    if checkpoint.get("schema_version") != 2:
        raise ValueError("unsupported checkpoint schema")
    expected=data.fingerprint if data_fingerprint is None else data_fingerprint
    if checkpoint['data_fingerprint']!=expected and checkpoint.get('caller_data_fingerprint')!=expected:
        raise ValueError("resume input fingerprint mismatch")
    if training_inputs is not None and checkpoint.get('training_inputs_fingerprint')!=training_inputs.fingerprint:
        raise ValueError('resume compiled graph/split identity mismatch')
    saved_trajectory = dict(checkpoint["trajectory_config"])
    saved_trajectory.pop("export_every", None)  # Older checkpoints included this output-only option.
    saved_trajectory.pop("export_start_epoch", None)
    saved_trajectory.pop("checkpoint_every", None)
    if saved_trajectory != trajectory(config):
        raise ValueError("resume trajectory config mismatch")
    if checkpoint.get("execution", {}).get("numerical") != execution_provenance["numerical"]:
        raise ValueError("resume numerical execution changed")
    if config.epochs <= checkpoint["completed_epochs"]:
        raise ValueError("resume epochs must extend completed epochs")
    current = environment()
    if any(checkpoint["environment"][k] != current[k] for k in ("torch", "numpy", "scipy")):
        raise ValueError("resume numerical environment changed")
    for key in ("_runtime",):
        if key in checkpoint["environment"] and checkpoint["environment"][key] != current[key]:
            raise ValueError(f"resume environment changed: {key}")
    from .artifacts import implementation_id
    compatibility = checkpoint.get('training_compatibility')
    if compatibility is not None:
        if compatibility != training_compatibility_id():
            raise ValueError('resume training implementation changed')
    elif checkpoint.get('training_implementation')!=implementation_id('training'):
        # Historical checkpoints have no semantic identity: do not guess that
        # changed executable code is compatible with their optimizer state.
        raise ValueError('resume training implementation changed')
    checkpoint['_source_sha256']=source_identity
    return checkpoint


def save_checkpoint(
    path,
    model,
    optimizer,
    data,
    config,
    completed,
    steps,
    sampler,
    history,
    execution_provenance,
    *, training_inputs=None,data_fingerprint=None,
):
    import torch
    state = {
        "schema_version": 2,
        "training_compatibility": training_compatibility_id(),
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "data_fingerprint": data.fingerprint if data_fingerprint is None else data_fingerprint,
        "caller_data_fingerprint":None if training_inputs is None else training_inputs.metadata.get('caller_data_fingerprint'),
        "training_inputs_fingerprint":None if training_inputs is None else training_inputs.fingerprint,
        "trajectory_config": trajectory(config),
        "execution": copy.deepcopy(execution_provenance),
        "batch_transfer_requested": {
            "resident_cache": config.resident_cache,
            "resident_cache_budget_gib": config.resident_cache_budget_gib,
            "resident_cache_reserve_gib": config.resident_cache_reserve_gib,
        },
        "completed_epochs": completed,
        "optimizer_steps": steps,
        "sampler_state": sampler.state_dict(),
        "rng_state": rng_state(),
        "history": history,
        "environment": environment(),
    }
    from .artifacts import implementation_id
    state['training_implementation']=implementation_id('training')
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)

