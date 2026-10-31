"""Independent tmux training jobs with on-disk progress and exact-state resume.

The notebook is a waiter, never the owner of the training process. Importing
this module does not import PyTorch or query a GPU.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid

from .utils import protect_output, write_json


def _read(path):
    return json.loads(Path(path).read_text())


def _session_exists(name):
    return subprocess.run(['tmux', 'has-session', '-t', name],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def _event(path, event, **details):
    with (path / 'attempt_history.jsonl').open('a') as stream:
        stream.write(json.dumps({'event': event, 'utc': datetime.now(timezone.utc).isoformat(),
                                 **details}) + '\n')


def _owned_process_alive(path, state):
    """Do not infer process death from a missing tmux session alone."""
    for key, marker in [('runner_pid', b'seareg.jobs'), ('worker_pid', b'seareg._runtime')]:
        pid = state.get(key)
        if not pid:
            continue
        try:
            command = Path(f'/proc/{pid}/cmdline').read_bytes()
        except FileNotFoundError:
            continue
        if marker in command and str(path.parent).encode() in command:
            return True
    return False


def _checkpoint_summary(source, trainer='single'):
    if trainer == 'multislice':
        # Verification and recovery run in the notebook/runner too. Inspect
        # state in a CPU-only child so those callers never import Torch.
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        script = ('import json,sys,torch; '
                  's=torch.load(sys.argv[1],map_location="cpu",weights_only=True); '
                  'print(json.dumps({k:s[k] for k in '
                  '("completed_epochs","optimizer_steps") if k in s}))')
        result = subprocess.run([sys.executable, '-c', script, str(source)],
                                env=environment, text=True, capture_output=True, check=True)
        return json.loads(result.stdout)
    import torch
    return torch.load(source, map_location='cpu', weights_only=True)


def _pre_target_checkpoint(source, target, *, trainer='single'):
    """Keep the target unchanged when the last save reached the final epoch."""
    completed = _checkpoint_summary(source, trainer)['completed_epochs']
    if completed < target:
        return source
    candidates = []
    for path in source.parent.glob('epoch_*.pt'):
        epoch = _checkpoint_summary(path, trainer)['completed_epochs']
        if epoch < target:
            candidates.append((epoch, path))
    if not candidates:
        raise ValueError('final checkpoint reached target but output is incomplete; '
                         'no pre-target checkpoint is available for exact recovery. '
                         'Existing files are retained; target epochs were not extended.')
    return max(candidates, key=lambda item: item[0])[1]


def _handoff(path, request, state):
    checkpoint = state.get('latest_checkpoint')
    text = (
        '# TRAINING_HANDOFF\n\n'
        f"Purpose: SEAREG {request.get('trainer', 'single')} training, {request['name']}\n\n"
        f"State: {state['state']}\nRunning: {state['state'] in ('starting', 'running')}\n"
        f"Output root: {state.get('output', request['output'])}\n"
        f"Job directory: {path}\nTraining inputs: {request['training_inputs']}\n"
        f"tmux session: {state.get('session', request.get('session'))}\n"
        f"Runner PID: {state.get('runner_pid')}\nWorker PID: {state.get('worker_pid')}\n"
        f"Physical GPU: {request['execution']['physical_gpu']}\n"
        f"Completed epoch: {state.get('verified_completed_epochs', 0)} / {request['config']['epochs']}\n"
        f"Optimizer steps: {state.get('optimizer_steps', 0)}\n"
        f"Latest checkpoint: {checkpoint}\n"
        f"stdout: {state.get('stdout', str(path / 'stdout.log'))}\n"
        f"stderr: {state.get('stderr', str(path / 'stderr.log'))}\n"
        f"Epoch diagnostics: {Path(state.get('output', request['output'])) / 'training_history.tsv'}\n"
        f"Resume: bash {shlex.quote(str(path / 'resume.sh'))} [checkpoint]\n"
        f"Known exception: {state.get('error', 'none recorded')}\n"
        f"Earlier exceptions and recovery attempts: {path / 'attempt_history.jsonl'}\n"
        f"Exit code: {state.get('exit_code')}\n"
        'Next check: read status.json, the last complete diagnostic row and the checkpoint; '
        'verify PID/logs/GPU live when reporting progress.\n\n'
        'Parameters (save intervals are execution management):\n```json\n'
        + json.dumps({'training': request['config'], 'execution': request['execution']}, indent=2)
        + '\n```\n'
    )
    temporary = path / 'TRAINING_HANDOFF.md.tmp'
    temporary.write_text(text)
    os.replace(temporary, path / 'TRAINING_HANDOFF.md')


def record_progress(output, row, config):
    """Persist one completed epoch; no tensor checks or GPU queries here."""
    path = Path(os.environ['SEAREG_JOB_PATH'])
    request = _read(path / 'request.json')
    state = _read(path / 'status.json')
    latest = Path(output) / 'checkpoints/latest.pt'
    state.update(state='running', output=str(output), worker_pid=os.getpid(),
                 verified_completed_epochs=row['epoch'], optimizer_steps=row['optimizer_steps'],
                 epoch_seconds=row['epoch_seconds_excluding_export'], loss=row['loss'])
    if latest.exists():
        state['latest_checkpoint'] = str(latest)
    write_json(path / 'status.json', state)
    _handoff(path, request, state)


@dataclass(frozen=True)
class TrainingJob:
    path: Path

    def wait(self, *, timeout=None, poll_interval=1., follow=True):
        """Wait for this job; a timeout/disconnected waiter never stops training."""
        if poll_interval <= 0 or (timeout is not None and timeout <= 0):
            raise ValueError('wait intervals must be positive')
        started = time.monotonic()
        offsets = {}
        while True:
            state = _read(self.path / 'status.json')
            if follow:
                for name, stream in [('stdout', sys.stdout), ('stderr', sys.stderr)]:
                    p = Path(state[name])
                    if p.exists():
                        with p.open('rb') as f:
                            f.seek(offsets.get(str(p), 0))
                            text = f.read()
                            offsets[str(p)] = f.tell()
                        if text:
                            stream.write(text.decode('utf-8', errors='replace'))
                            stream.flush()
            if state['state'] == 'completed':
                from .data import TrainingRun
                return TrainingRun.load(state['output'])
            if state['state'] == 'failed':
                raise RuntimeError(f"training job failed: {state.get('error')}; logs: {self.path}")
            if not _session_exists(state['session']):
                # Re-read: completion publication can race the final session exit.
                latest = _read(self.path / 'status.json')
                if latest['state'] in ('completed', 'failed'):
                    continue
                if _owned_process_alive(self.path, latest):
                    if timeout is not None and time.monotonic() - started >= timeout:
                        raise TimeoutError(f'training process continues independently; reload job at {self.path}')
                    time.sleep(poll_interval)
                    continue
                latest.update(state='failed', error='tmux exited without a verified COMPLETE state', exit_code=None)
                write_json(self.path / 'status.json', latest)
                _handoff(self.path, _read(self.path / 'request.json'), latest)
                raise RuntimeError(f"training session stopped without completion; logs: {self.path}")
            if timeout is not None and time.monotonic() - started >= timeout:
                raise TimeoutError(f'training continues independently; reload job at {self.path}')
            time.sleep(poll_interval)

    def resume(self, checkpoint=None):
        """Start a new attempt from a complete state; preserve the failed attempt."""
        if shutil.which('tmux') is None:
            raise RuntimeError('tmux is required for detached training')
        lock = self.path / '.resume.lock'
        with lock.open('x'):
            try:
                state = _read(self.path / 'status.json')
                for _ in range(40):
                    if not _session_exists(state['session']):
                        break
                    if state['state'] not in ('failed', 'completed'):
                        raise RuntimeError('training job is still running')
                    time.sleep(.05)
                else:
                    raise RuntimeError('previous training session has not exited')
                if _owned_process_alive(self.path, state):
                    raise RuntimeError('training process is still running; recovery was not started')
                if state['state'] == 'completed':
                    raise ValueError('completed jobs do not need recovery')
                request = _read(self.path / 'request.json')
                try:
                    run, final = _verify_completed({**request, 'output': state['output']})
                except (OSError, ValueError, KeyError, IndexError):
                    pass
                else:
                    state.update(state='completed', exit_code=0,
                                 verified_completed_epochs=run.final_epoch,
                                 optimizer_steps=run.metadata['optimizer_steps'],
                                 latest_checkpoint=str(final))
                    with Path(state['stdout']).open('a') as stream:
                        stream.write(f'COMPLETE verified existing output {run.path}\n')
                    _event(self.path, 'verified_existing_completion', output=str(run.path))
                    _handoff(self.path, request, state)
                    write_json(self.path / 'status.json', state)
                    return self
                source = Path(checkpoint or state.get('latest_checkpoint') or
                              Path(state['output']) / 'checkpoints/latest.pt').resolve()
                if not source.is_file():
                    raise FileNotFoundError(f'no exact checkpoint available: {source}')
                source = _pre_target_checkpoint(source, request['config']['epochs'], trainer=request.get('trainer', 'single'))
                request['resume_from'] = str(source)
                request['output'] = str(Path(request['output']).with_name(
                    Path(request['output']).name + '_resume_' + uuid.uuid4().hex[:8]))
                _submit(self.path, request, resumed=True)
            finally:
                lock.unlink(missing_ok=True)
        return self


def load_training_job(path):
    path = Path(path).resolve()
    _read(path / 'request.json')
    _read(path / 'status.json')
    return TrainingJob(path)


def _submit(path, request, *, resumed=False):
    attempt = path / 'attempts' / uuid.uuid4().hex[:12]
    attempt.mkdir(parents=True)
    request = dict(request)
    slug = re.sub(r'[^A-Za-z0-9_-]+', '-', request['name'])[:32]
    c = request['config']
    shard_label = c.get('shards', '-'.join(str(v) for k,v in sorted(c.get('slice_shards', {}).items())))
    session = f"seareg-{slug}-e{c['epochs']}-p{c['pn_r']}-s{shard_label}-{uuid.uuid4().hex[:8]}"
    request.update(session=session, job_path=str(path))
    request_path = attempt / 'request.json'
    write_json(request_path, request)
    if resumed:
        write_json(attempt / 'previous_status.json', _read(path / 'status.json'))
    _event(path, 'resume' if resumed else 'submit', session=session,
           output=request['output'], resume_from=request.get('resume_from'))
    stdout = attempt / 'stdout.log' if resumed else path / 'stdout.log'
    stderr = attempt / 'stderr.log' if resumed else path / 'stderr.log'
    stdout.touch(exist_ok=False)
    stderr.touch(exist_ok=False)
    source = str(Path(__file__).resolve().parents[1])
    command = ['env', f"PYTHONPATH={source}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
               sys.executable, '-u', '-m', 'seareg.jobs', '--worker', str(request_path)]
    shell_command = shlex.join(command) + ' >' + shlex.quote(str(stdout)) + ' 2>' + shlex.quote(str(stderr))
    (attempt / 'command.txt').write_text(shell_command + '\n')
    if not resumed:
        (path / 'command.txt').write_text(shell_command + '\n')
    state = dict(state='starting', output=request['output'], session=session,
                 stdout=str(stdout), stderr=str(stderr), worker_pid=None, runner_pid=None,
                 verified_completed_epochs=0, optimizer_steps=0, exit_code=None,
                 latest_checkpoint=request.get('resume_from'))
    write_json(path / 'status.json', state)
    _handoff(path, request, state)
    try:
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', source, shell_command],
                       check=True, capture_output=True, text=True)
    except BaseException as error:
        state.update(state='failed', error=f'{type(error).__name__}: {error}')
        write_json(path / 'status.json', state)
        _event(path, 'failed', error=state['error'], session=session)
        _handoff(path, request, state)
        raise


def start_training_job(training_inputs, train_config, output, *, execution=None, name=None):
    """Launch one independent job. The actual training output stays uncreated."""
    if shutil.which('tmux') is None:
        raise RuntimeError('tmux is required for detached training; install it before selecting train mode')
    from .artifacts import TrainingInputs
    inputs = TrainingInputs.load(training_inputs)
    output = protect_output(output, [inputs.path])
    path = protect_output(output.with_name(output.name + '_job'), [inputs.path])
    if output.exists() or path.exists():
        raise FileExistsError(output if output.exists() else path)
    runtime = {'physical_gpu': None, 'torch_threads': 1,
        'torch_interop_threads': 1, 'blas_threads': 1, 'strict_algorithms': True, 'tf32': False}
    runtime.update(execution or {})
    if train_config.device == 'cpu' and runtime.get('physical_gpu') is not None:
        raise ValueError('CPU jobs require physical_gpu=null')
    request = {'schema_version': 1, 'name': str(name or output.parent.name),
               'training_inputs': str(inputs.path), 'training_inputs_fingerprint': inputs.fingerprint,
               'shape': list(inputs.metadata['shape']), 'config': asdict(train_config),
               'execution': runtime, 'output': str(output), 'resume_from': None,
               'created_at_utc': datetime.now(timezone.utc).isoformat()}
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / 'request.json', request)
    write_json(path / 'method_manifest.json', request)
    source = str(Path(__file__).resolve().parents[1])
    resume_command = shlex.join(['env', f'PYTHONPATH={source}', sys.executable,
                                 '-m', 'seareg.jobs', '--request', str(path / 'request.json')])
    (path / 'resume.sh').write_text('#!/bin/sh\nset -eu\n' + resume_command + ' "$@"\n')
    _submit(path, request)
    return TrainingJob(path)


def _verify_completed(request):
    from .data import TrainingRun
    import pandas as pd
    output = Path(request['output'])
    run = TrainingRun.load(output)
    target = request['config']['epochs']
    history = pd.read_csv(output / 'training_history.tsv', sep='\t')
    checkpoint = output / 'checkpoints' / f'epoch_{target:04d}.pt'
    state = _checkpoint_summary(checkpoint, request.get('trainer', 'single'))
    if (run.final_epoch != target or int(history.iloc[-1]['epoch']) != target
            or state['completed_epochs'] != target
            or state['optimizer_steps'] != run.metadata['optimizer_steps']):
        raise ValueError('final epoch/diagnostics/checkpoint do not agree')
    from .exports import embedding_directory
    directory=embedding_directory(output,target)
    export = _read(directory / 'export.json')
    for entry in export['entries']:
        if not (directory / entry['path']).is_file():
            raise FileNotFoundError(entry['path'])
    return run, checkpoint


def _worker(request_file):
    request = _read(request_file)
    path = Path(request['job_path'])
    os.environ['SEAREG_JOB_PATH'] = str(path)
    state = _read(path / 'status.json')
    state.update(state='running', runner_pid=os.getpid())
    write_json(path / 'status.json', state)
    _handoff(path, request, state)
    print('COMMAND:', (Path(request_file).parent / 'command.txt').read_text().strip(), flush=True)
    print('SHAPE:', request['shape'], 'GPU:', request['execution']['physical_gpu'], flush=True)
    print('TRAINING PROTOCOL:', json.dumps(request['config'], sort_keys=True), flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'training runner received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if request.get('trainer', 'single') == 'multislice':
            from .multislice.config import MultiSliceConfig as TrainConfig
            from .multislice.train import train_multislice as train_SEAREG
        elif request.get('trainer', 'single') == 'single':
            from .config import TrainConfig
            from .train import train_SEAREG
        else:
            raise ValueError('unknown trainer')
        from .config import from_dict
        train_SEAREG(request['training_inputs'], from_dict(TrainConfig, request['config']),
                     request['output'], execution=request['execution'],
                     resume_from=request.get('resume_from'))
        run, checkpoint = _verify_completed(request)
        state = _read(path / 'status.json')
        state.update(state='completed', exit_code=0, output=str(run.path),
                     verified_completed_epochs=run.final_epoch,
                     optimizer_steps=run.metadata['optimizer_steps'], latest_checkpoint=str(checkpoint))
        print('COMPLETE', run.path, flush=True)
        _handoff(path, request, state)
        _event(path, 'completed', output=str(run.path), epoch=run.final_epoch)
        write_json(path / 'status.json', state)
    except BaseException as error:
        state = _read(path / 'status.json')
        state.update(state='failed', exit_code=1, error=f'{type(error).__name__}: {error}')
        _event(path, 'failed', error=state['error'], output=request['output'])
        _handoff(path, request, state)
        write_json(path / 'status.json', state)
        print('FAILED:', state['error'], file=sys.stderr, flush=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    group.add_argument('--request', type=Path, help='existing job request to recover in tmux')
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('checkpoint', nargs='?', type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        _worker(args.worker)
    else:
        job = load_training_job(args.request.parent)
        job.resume(args.resume_from or args.checkpoint)
        print('Recovery submitted:', job.path)


if __name__ == '__main__':
    main()
