"""Atomic full-state checkpointing for deterministic multislice CPU recovery.

Scientific source identity is strict: checkpoints produced by earlier package
implementations or experimental adapters are not native resume files here.
"""
from dataclasses import asdict
from pathlib import Path
import ast,copy
from ..checkpoint import rng_state
from ..utils import identity,environment,file_sha256


def trajectory(config):
    values=asdict(config)
    for key in ('epochs','checkpoint_every','export_every','export_start_epoch','resident_cache',
                'resident_cache_budget_gib','resident_cache_reserve_gib'):
        values.pop(key)
    return values


def implementation_identity():
    root=Path(__file__).parent
    files=[root/name for name in ('config.py','inputs.py','stages.py','sampling.py','model.py','gate.py','geometry.py','protection.py','coherence.py','microgain.py','balanced.py','grouped_loss.py','checkpoint.py','train.py')]
    files += [root.parent/name for name in ('model.py','loss.py','scoring.py','_cuda_scoring.py','resident.py','exports.py')]
    return identity({str(path.relative_to(root.parent)):ast.dump(ast.parse(path.read_text(encoding='utf-8-sig')),include_attributes=False) for path in files})


def cpu_state(value):
    import torch
    if torch.is_tensor(value):return value.detach().to('cpu',copy=True)
    if isinstance(value,dict):return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_state(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_state(v) for v in value)
    return value


def save_checkpoint(path,model,optimizer,inputs,config,completed,steps,sampler,history,execution):
    import torch
    state={'schema_version':1,'trainer':'multislice','model_protocol':'two_layer_sum_cs_matched03_v1','micro_formula_version':1,'training_implementation':implementation_identity(),
           'completed_epochs':completed,'optimizer_steps':steps,'model_state':cpu_state(model.state_dict()),
           'optimizer_state':cpu_state(optimizer.state_dict()),'trajectory_config':trajectory(config),
           'training_inputs_fingerprint':inputs.fingerprint,'data_fingerprint':inputs.metadata['data_fingerprint'],
           'execution':copy.deepcopy(execution),'sampler_state':sampler.state_dict(),'rng_state':rng_state(),
           'next_epoch':completed,'history':copy.deepcopy(history),'environment':environment(),
           'scheduler_state':None,'amp_scaler_state':None,
           'order_protocol':'seed_epoch_slice_pass_shuffle_and_rotating_slice_order_v1'}
    path=Path(path);temporary=path.with_name(path.name+'.tmp');torch.save(state,temporary);temporary.replace(path)


def read_checkpoint(path,inputs,config,execution):
    import torch
    before=file_sha256(path);state=torch.load(path,map_location='cpu',weights_only=True)
    if before!=file_sha256(path):raise ValueError('checkpoint changed while loading')
    if state.get('schema_version')!=1 or state.get('trainer')!='multislice':raise ValueError('not a multislice checkpoint')
    if state.get('model_protocol')!='two_layer_sum_cs_matched03_v1' or state.get('micro_formula_version')!=1:raise ValueError('resume model protocol or formula mismatch')
    if state['training_inputs_fingerprint']!=inputs.fingerprint:raise ValueError('resume input fingerprint mismatch')
    if state['trajectory_config']!=trajectory(config):raise ValueError('resume trajectory config mismatch')
    if state['training_implementation']!=implementation_identity():raise ValueError('resume implementation changed')
    if state['execution']['numerical']!=execution['numerical']:raise ValueError('resume numerical execution changed')
    current=environment()
    if any(state['environment'][k]!=current[k] for k in ('torch','numpy','scipy','scikit-learn','_runtime')):
        raise ValueError('resume numerical environment changed')
    completed=state.get('completed_epochs'); next_epoch=state.get('next_epoch'); steps=state.get('optimizer_steps')
    if type(completed) is not int or completed<0 or type(next_epoch) is not int or next_epoch!=completed:
        raise ValueError('resume completed/next epoch mismatch')
    if type(steps) is not int or steps<0 or steps!=completed*config.updates_per_epoch:
        raise ValueError('resume optimizer steps mismatch')
    model_state=state.get('model_state',{}); expected_gate=50*(completed//50)
    for key in ('gate_epoch','micro_epoch'):
        value=model_state.get(key)
        if not torch.is_tensor(value) or value.ndim!=0 or value.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8) or int(value)!=expected_gate:
            raise ValueError('resume gate/micro epoch phase mismatch')
    if config.epochs<=completed:raise ValueError('resume target must exceed completed epoch')
    state['_source_sha256']=before
    return state
