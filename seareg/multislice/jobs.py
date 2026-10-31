"""Submit the multislice trainer through native durable jobs."""
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import shutil,sys,shlex
from .inputs import MultiSliceInputs, sampling_config, preparation_config
from ..jobs import TrainingJob,_submit
from ..utils import protect_output,write_json


def start_multislice_training_job(inputs,config,output,*,execution=None,name=None):
    """Submit an independent tmux job and return a TrainingJob handle.

    Accept MultiSliceInputs or its directory and a matching MultiSliceConfig.
    ``output`` is the unused training directory; its sibling ``<output-name>_job``
    stores logs, the handoff and recovery script. Call ``job.wait()`` for the
    completed TrainingRun or ``load_training_job(job.path)`` to reconnect.
    """
    if shutil.which('tmux') is None:raise RuntimeError('tmux is required for detached training')
    inputs=MultiSliceInputs.load(inputs)
    if dict(inputs.metadata['sampling_config'])!=sampling_config(config) or dict(inputs.metadata['preparation_config'])!=preparation_config(config):
        raise ValueError('training inputs and multislice config differ')
    output=protect_output(output,[inputs.path]);path=protect_output(output.with_name(output.name+'_job'),[inputs.path])
    if output.exists() or path.exists():raise FileExistsError(output if output.exists() else path)
    runtime=validated_execution(config,execution)
    request={'schema_version':1,'trainer':'multislice','name':str(name or output.parent.name),
        'training_inputs':str(inputs.path),'training_inputs_fingerprint':inputs.fingerprint,'shape':list(inputs.metadata['shape']),
        'config':asdict(config),'execution':runtime,'output':str(output),'resume_from':None,
        'created_at_utc':datetime.now(timezone.utc).isoformat()}
    path.mkdir(parents=True)
    write_json(path/'request.json',request);write_json(path/'method_manifest.json',request)
    source=str(Path(__file__).resolve().parents[2])
    command=shlex.join(['env',f'PYTHONPATH={source}',sys.executable,'-m','seareg.jobs','--request',str(path/'request.json')])
    (path/'resume.sh').write_text('#!/bin/sh\nset -eu\n'+command+' "$@"\n')
    _submit(path,request)
    return TrainingJob(path)


def validated_execution(config,execution=None):
    runtime={'physical_gpu':None,'torch_threads':1,'torch_interop_threads':1,'blas_threads':1,'strict_algorithms':False,'tf32':False}
    if execution is not None:
        if set(execution)-set(runtime):raise ValueError('unknown execution options')
        runtime.update(execution)
    gpu=runtime['physical_gpu']
    if gpu is not None and (type(gpu) is not int or gpu<0):raise ValueError('physical_gpu must be null or nonnegative integer')
    for key in ('torch_threads','torch_interop_threads','blas_threads'):
        if type(runtime[key]) is not int or runtime[key]<1:raise ValueError(f'{key} must be positive integer')
    for key in ('strict_algorithms','tf32'):
        if type(runtime[key]) is not bool:raise ValueError(f'{key} must be boolean')
    if config.device=='cpu' and gpu is not None:raise ValueError('CPU jobs require physical_gpu=null')
    if config.device.startswith('cuda') and gpu is None:raise ValueError('CUDA jobs require physical_gpu')
    return runtime
