"""Isolated execution settings and worker dispatch before scientific imports."""

import os
from pathlib import Path
import tempfile
import json
import subprocess
import sys
import codecs
import selectors
from .utils import protect_output


def configure_runtime():
    """Use writable caches without changing caller-supplied environment choices."""
    if "NUMBA_CACHE_DIR" not in os.environ or "MPLCONFIGDIR" not in os.environ:
        uid = os.getuid() if hasattr(os, "getuid") else "user"
        root = Path(tempfile.gettempdir()) / f"seareg_runtime_{uid}"
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.is_symlink() or (hasattr(os, "getuid") and root.stat().st_uid != os.getuid()):
            raise ValueError("runtime cache directory must belong to the current user")
        for key, name in (("NUMBA_CACHE_DIR", "numba"), ("MPLCONFIGDIR", "matplotlib")):
            if key not in os.environ:
                path = root / name
                path.mkdir(exist_ok=True)
                os.environ[key] = str(path)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("MPLBACKEND", "Agg")


def run_worker(action, request, output, execution):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    protect_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".seareg-worker-", dir=output.parent))
    payload = {**request, "action": action, "output": str(output), "execution": execution}
    request_path = work / "request.json"
    request_path.write_text(json.dumps(payload, indent=2) + "\n")
    env = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    gpu = execution.get("physical_gpu")
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[key] = str(execution.get("blas_threads", 1))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with (work / "stdout.log").open("w", encoding="utf-8") as stdout, (work / "stderr.log").open("w", encoding="utf-8") as stderr:
        with subprocess.Popen(
            [sys.executable, "-u", "-B", "-m", "seareg._runtime", str(request_path)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ) as process:
            try:
                # Drain both pipes without waiting for newlines or child exit.
                with selectors.DefaultSelector() as selector:
                    for pipe, log, stream in ((process.stdout, stdout, sys.stdout),
                                              (process.stderr, stderr, sys.stderr)):
                        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                        selector.register(pipe, selectors.EVENT_READ, (log, stream, decoder))
                    while selector.get_map():
                        for key, _ in selector.select():
                            log, stream, decoder = key.data
                            block = key.fileobj.read1(8192)
                            message = decoder.decode(block, final=not block)
                            if not block:
                                selector.unregister(key.fileobj)
                            if message:
                                log.write(message)
                                log.flush()
                                stream.write(message)
                                stream.flush()
                returncode = process.wait()
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise
    (work / "exit.json").write_text(json.dumps({"returncode": returncode}) + "\n")
    if returncode:
        tail = (work / "stderr.log").read_text()[-3000:]
        raise RuntimeError(f"SEAREG {action} worker failed ({returncode}); logs: {work}\n{tail}")
    return work


def main(path):
    request = json.loads(Path(path).read_text())
    execution = request["execution"]
    configure_runtime()
    import torch
    torch.set_num_threads(execution.get("torch_threads", 1))
    torch.set_num_interop_threads(execution.get("torch_interop_threads", 1))
    torch.use_deterministic_algorithms(execution.get("strict_algorithms", True))
    torch.backends.cuda.matmul.allow_tf32 = execution.get("tf32", False)
    torch.backends.cudnn.allow_tf32 = execution.get("tf32", False)
    if request["action"] == "train":
        if request.get('trainer', 'single') == 'multislice':
            from .multislice.config import MultiSliceConfig as TrainConfig
            from .multislice.train import _train as train
        elif request.get('trainer', 'single') == 'single':
            from .config import TrainConfig
            from .train import _train as train
        else:
            raise ValueError('unknown trainer')
        from .config import from_dict
        config = from_dict(TrainConfig, request["config"])
        train(
            request["prepared"],
            config,
            request["output"],
            request.get("resume_from"),
            execution=execution,
        )
    elif request["action"] == "analyze":
        from .config import AnalysisConfig, from_dict
        from .analysis import analyze
        from .data import TrainingRun
        from .artifacts import validation_scope
        with validation_scope():
            inputs = TrainingRun.load(request["training"]).load_embeddings()
        analyze(inputs,from_dict(AnalysisConfig,request['config']),request['output'],plots=request.get('plots',True))
    else:
        raise ValueError("unknown isolated worker action")


if __name__ == "__main__":
    main(sys.argv[1])
