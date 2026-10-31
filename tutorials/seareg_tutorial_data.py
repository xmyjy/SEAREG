"""Prepare selected local inputs and reserve unique tutorial outputs."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import re
import tempfile


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _dataset_name(dataset):
    if not isinstance(dataset, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', dataset):
        raise ValueError('invalid dataset name')
    return dataset


def _input_path(root, dataset, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or relative.parts[:2] != ('data', dataset):
        raise ValueError('invalid dataset input path')
    path = root / relative
    if not path.resolve().is_relative_to(root / 'data' / dataset):
        raise ValueError('dataset input path escapes its directory')
    return path


def _check_input(path, entry):
    if not path.is_file() or path.stat().st_size != entry['bytes'] or file_sha256(path) != entry['sha256']:
        raise ValueError(f'tutorial input does not match its manifest: {path}')


def prepare_tutorial_file(root, dataset, name, *, required=True):
    """Reuse an exact input or extract its checked gzip parts without overwriting.

    Call for counts before preprocessing. Call for optional motifs only after
    filtering, so an absent annotation can still reach the existing R branch.
    """
    root = Path(root).resolve()
    dataset = _dataset_name(dataset)
    if name not in ('counts.h5ad', 'motifs.h5'):
        raise ValueError('unsupported tutorial input name')
    manifest = json.loads((root / 'data/manifest.json').read_text())
    spec = next(item for item in manifest['datasets'] if item['dataset_id'] == dataset)
    entry = next((item for item in spec['files'] if item['path'] == f'data/{dataset}/{name}'), None)
    target = _input_path(root, dataset, f'data/{dataset}/{name}')
    if entry is None:
        if not required and not target.exists():
            return None
        raise FileNotFoundError(f'no tutorial input entry for {dataset}/{name}')
    if target.exists() or target.is_symlink():
        _check_input(target, entry)
        return target
    archive = entry.get('archive', {})
    parts = [(item, _input_path(root, dataset, item['path'])) for item in archive.get('parts', [])]
    available = [path.is_file() for _, path in parts]
    if not any(available):
        if not required:
            return None
        raise FileNotFoundError(
            f'missing tutorial input: {target}; restore counts.h5ad or its '
            'manifest-listed gzip parts from the repository download. See data/README.md.')
    if not all(available) or archive.get('compression') != 'gzip':
        raise ValueError(f'incomplete tutorial archive: {target}')
    for item, path in parts:
        if path.stat().st_size != item['bytes'] or file_sha256(path) != item['sha256']:
            raise ValueError(f'tutorial archive hash mismatch: {path}')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # Join split gzip streams in a temporary file, without loading them into RAM.
        with tempfile.TemporaryFile() as compressed:
            for _, path in parts:
                with path.open('rb') as part:
                    for block in iter(lambda: part.read(1024 * 1024), b''):
                        compressed.write(block)
            compressed.seek(0)
            with tempfile.NamedTemporaryFile(prefix='.unpack-', dir=target.parent, delete=False) as output:
                temporary = Path(output.name)
                size = 0
                with gzip.GzipFile(fileobj=compressed, mode='rb') as source:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        size += len(block)
                        if size > entry['bytes']:
                            raise ValueError(f'tutorial archive exceeds its declared input size: {target}')
                        output.write(block)
        _check_input(temporary, entry)
        try:
            os.link(temporary, target)  # Atomic creation; never replace an existing file.
        except FileExistsError:
            _check_input(target, entry)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)



def prepare_input_file(path, *, bundle_root=None, required=True) -> Path | None:
    """Use the selected file, unpacking only an exact missing bundled input.

    Existing files belong to the caller and need no example hash or manifest.
    Scientific count/motif validation remains in the configured workflow.
    The bundle manifest is used only to recover a distributed missing file.
    """
    target = Path(path).expanduser().resolve()
    if target.is_file():
        return target
    if target.is_dir():
        raise IsADirectoryError(f'input must be a regular file: {target}')
    if target.exists():
        raise ValueError(f'input must be a regular file: {target}')

    if bundle_root is not None:
        root = Path(bundle_root).expanduser().resolve()
        relative = target.relative_to(root) if target.is_relative_to(root) else None
        if (relative is not None and len(relative.parts) == 3
                and relative.parts[0] == 'data'
                and relative.name in ('counts.h5ad', 'motifs.h5')):
            manifest_path = root / 'data/manifest.json'
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text())
                dataset = relative.parts[1]
                listed = any(
                    spec.get('dataset_id') == dataset
                    and any(entry.get('path') == relative.as_posix()
                            for entry in spec.get('files', []))
                    for spec in manifest['datasets'])
                if listed:
                    return prepare_tutorial_file(root, dataset, relative.name, required=required)

    if not required:
        return None
    raise FileNotFoundError(
        f'missing selected input: {target}; set the input path to an existing file. '
        'Only exact manifest-listed bundled paths can be unpacked automatically.')


def create_run_directory(root, dataset, mode, *, output=None):
    """Create a unique run beside supplied exports; never move existing results.

    An explicit output base may be outside the repository. The caller prepares
    its required inputs before reserving a run.
    """
    import uuid
    from datetime import datetime, timezone
    root = Path(root).expanduser().resolve()
    dataset = _dataset_name(dataset)
    if mode not in ('provided', 'train'):
        raise ValueError('run mode must be provided or train')
    if output is None:
        parent = root / 'result' / dataset / 'runs'
        for candidate in (root / 'result', root / 'result' / dataset, parent):
            if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
                raise ValueError('run directory escapes its repository')
    else:
        base = Path(output).expanduser().resolve()
        parent = base / 'runs'
        if parent.is_symlink() or not parent.resolve().is_relative_to(base):
            raise ValueError('run directory escapes its selected output directory')
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / (mode + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                     + '_' + uuid.uuid4().hex[:8])
    path.mkdir(exist_ok=False)
    return path
