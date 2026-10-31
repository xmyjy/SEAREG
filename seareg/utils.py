"""Content identities, immutable artifact publication and output protection."""

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass, fields
from collections.abc import Mapping
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid


def _publish_noreplace(source, destination):
    """Linux atomic directory publication, including noncooperating writers.

    Fail explicitly where renameat2 is unavailable instead of falling back to a
    check-then-rename sequence that can replace an existing empty directory.
    """
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        # Older glibc on the target host lacks the wrapper; its Linux kernel
        # still supplies renameat2. Numbers follow the Linux UAPI headers.
        import platform

        number = {"x86_64": 316, "aarch64": 276}.get(platform.machine())
        if platform.system() != "Linux" or number is None:
            raise RuntimeError("atomic artifact publication requires Linux renameat2")
        libc.syscall.restype = ctypes.c_long
        result = libc.syscall(
            ctypes.c_long(number),
            ctypes.c_int(-100),
            ctypes.c_char_p(os.fsencode(source)),
            ctypes.c_int(-100),
            ctypes.c_char_p(os.fsencode(destination)),
            ctypes.c_uint(1),
        )
    else:
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def json_safe(obj):
    if is_dataclass(obj):
        return {field.name:json_safe(getattr(obj,field.name)) for field in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    os.replace(tmp, path)


@contextmanager
def attempt_writer(output, attempts_dir=None):
    """Keep command diagnostics even when an incomplete artifact is rolled back."""
    root = Path(attempts_dir) if attempts_dir is not None else Path(output).parent/'attempts'
    path = root/('motif-'+uuid.uuid4().hex)
    path.mkdir(parents=True, exist_ok=False)
    write_json(path/'status.json', {'state':'running', 'output':str(Path(output).resolve())})
    try:
        yield path
    except BaseException as error:
        write_json(path/'status.json', {'state':'failed', 'error':f'{type(error).__name__}: {error}'})
        raise
    else:
        write_json(path/'status.json', {'state':'complete'})


def identity(value):
    return hashlib.sha256(
        json.dumps(
            json_safe(value), sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment():
    """Record loaded runtime code separately from installed distribution metadata.

    Optional packages are never imported for bookkeeping. A None runtime version
    means that package was not loaded in this process, even if installed.
    """
    import sys
    import platform
    from seareg import __version__

    modules = {
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "torch": "torch",
        "anndata": "anndata",
        "scikit-learn": "sklearn",
        "scanpy": "scanpy",
        "umap-learn": "umap",
        "pynndescent": "pynndescent",
        "numba": "numba",
        "llvmlite": "llvmlite",
        "igraph": "igraph",
        "louvain": "louvain",
        "matplotlib": "matplotlib",
        "h5py": "h5py",
    }
    result = {}
    distributions = {}
    paths = {}
    for package, module_name in modules.items():
        module = sys.modules.get(module_name)
        try:
            distributions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            distributions[package] = None
        if package == "scanpy" and module is not None:
            # Scanpy's deprecated dynamic __version__ only reads this metadata.
            version = vars(module).get("__version__", distributions[package])
        else:
            version = getattr(module, "__version__", None)
        result[package] = str(version) if version is not None else None
        paths[package] = getattr(module, "__file__", None)
    result["_distribution_versions"] = distributions
    result["_module_paths"] = paths
    result["_version_mismatches"] = {
        k: {"loaded": result[k], "distribution": distributions[k]}
        for k in modules
        if result[k] is not None and result[k] != distributions[k]
    }
    torch_module = sys.modules.get("torch")
    result["_runtime"] = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch_threads": torch_module.get_num_threads() if torch_module is not None else None,
        "torch_interop_threads": (
            torch_module.get_num_interop_threads() if torch_module is not None else None
        ),
        "torch_deterministic": (
            torch_module.are_deterministic_algorithms_enabled()
            if torch_module is not None
            else None
        ),
    }
    sources = {
        str(p.relative_to(Path(__file__).parent)): file_sha256(p)
        for p in sorted(Path(__file__).parent.rglob("*.py"))
    }
    result["_implementation"] = {"version": __version__, "source_fingerprint": identity(sources)}
    return result


def seal_artifact(path, kind, metadata):
    root = Path(path)
    files = {
        str(p.relative_to(root)): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p != root / "manifest.json"
    }
    manifest = {"schema_version": 1, "kind": kind, "metadata": json_safe(metadata), "files": files}
    manifest["fingerprint"] = identity(manifest)
    write_json(root / "manifest.json", manifest)
    return manifest


def verify_artifact(path, kind=None):
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or (kind and manifest["kind"] != kind):
        raise ValueError("artifact schema/kind mismatch")
    unsigned = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if identity(unsigned) != manifest.get("fingerprint"):
        raise ValueError("manifest hash mismatch")
    for relative, digest in manifest["files"].items():
        name = Path(relative)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("unsafe artifact path")
        p = root / name
        if not p.is_file() or p.is_symlink() or root not in p.resolve().parents:
            raise ValueError(f"missing/unsafe artifact file: {relative}")
        if file_sha256(p) != digest:
            raise ValueError(f"artifact hash mismatch: {relative}")
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p != root / "manifest.json"
    }
    if actual != set(manifest["files"]):
        raise ValueError("unregistered artifact files")
    return manifest


@contextmanager
def artifact_writer(output, kind, metadata):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / ("." + output.name + ".lock")
    if output.exists():
        raise FileExistsError(output)
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    stage = None
    try:
        if output.exists():
            raise FileExistsError(output)
        stage = Path(tempfile.mkdtemp(prefix="." + output.name + ".stage-", dir=output.parent))
        yield stage
        seal_artifact(stage, kind, metadata)
        verify_artifact(stage, kind)
        if output.exists():
            raise FileExistsError(output)
        _publish_noreplace(stage, output)
        stage = None
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        lock.unlink(missing_ok=True)


def protect_output(path, protected=()):
    destination = Path(path).resolve()
    for item in protected:
        source = Path(item).resolve()
        if destination == source or source in destination.parents:
            raise ValueError(f"output is inside a protected input artifact: {source}")
    for parent in (destination, *destination.parents):
        if (parent / "manifest.json").is_file() or (parent / "FINAL_RECEIPT.json").is_file():
            raise ValueError(f"output is inside a sealed artifact: {parent}")
    return destination
