"""Load selected tutorial exports from a native directory or completed run."""
from pathlib import Path
import hashlib
import json
import os


def _sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _native_export_entries(directory):
    """Verify the export-local receipt for the selected propagated matrices."""
    receipt_path = directory / 'export.json'
    if receipt_path.is_symlink():
        raise ValueError(f'unsafe native export metadata path: {receipt_path}')
    receipt_bytes = receipt_path.read_bytes()
    export = json.loads(receipt_bytes)
    if not isinstance(export, dict) or type(export.get('schema_version')) is not int or export['schema_version'] != 1:
        raise ValueError('native export schema must be version 1')
    epoch = export.get('epoch')
    if type(epoch) is not int or epoch < 1:
        raise ValueError('native export epoch must be a positive integer')
    ready = export.get('analysis_ready', {})
    if not isinstance(ready, dict) or ('propagated' in ready and type(ready['propagated']) is not bool):
        raise ValueError('invalid native export analysis_ready metadata')
    if ready.get('propagated') is False:
        raise ValueError(f'epoch {epoch} propagated embedding is not analyzable: zero-direction rows')
    if not isinstance(export.get('entries'), list):
        raise ValueError('native export entries must be a list')
    selected = {}
    paths = {}
    for entry in export['entries']:
        if not isinstance(entry, dict):
            raise ValueError('invalid native export entry')
        if entry.get('representation') == 'ego':
            continue
        if entry.get('representation') != 'propagated':
            raise ValueError('unknown native export representation')
        kind = entry.get('entity_type')
        if kind not in ('spot', 'peak', 'motif', 'kmer'):
            raise ValueError('unsupported native propagated export entity')
        if kind in selected:
            raise ValueError(f'duplicate native propagated {kind} export')
        if type(entry.get('epoch')) is not int or entry['epoch'] != epoch:
            raise ValueError(f'native {kind} export epoch mismatch')
        shape = entry.get('shape')
        if not isinstance(shape, list) or len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError(f'invalid native {kind} export shape')
        analyzable = entry.get('analyzable', True)
        zero_rows = entry.get('zero_direction_rows', 0)
        if type(analyzable) is not bool or type(zero_rows) is not int or zero_rows < 0:
            raise ValueError(f'invalid native {kind} analyzability metadata')
        if not analyzable or zero_rows:
            raise ValueError(f'native propagated {kind} export is not analyzable: zero-direction rows')
        name = entry.get('path')
        if not isinstance(name, str) or not name:
            raise ValueError('invalid native embedding path')
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative == Path('.'):
            raise ValueError('unsafe native embedding path')
        candidate = directory / relative
        if candidate.resolve() != candidate or not candidate.resolve().is_relative_to(directory):
            raise ValueError(f'unsafe native embedding path: {candidate}')
        if candidate in paths.values():
            raise ValueError('duplicate native embedding path')
        if not candidate.is_file():
            raise FileNotFoundError(f'missing native embedding file: {candidate}')
        if _sha(candidate) != entry.get('sha256'):
            raise ValueError(f'native {kind} export hash mismatch: {candidate}')
        selected[kind] = entry
        paths[kind] = candidate
    if not {'spot', 'peak'} <= selected.keys():
        raise ValueError('native export requires propagated spot and peak embeddings')
    if {'motif', 'kmer'} <= selected.keys():
        raise ValueError('native export must select one DNA entity kind, motif or kmer')
    return export, selected, paths, hashlib.sha256(receipt_bytes).hexdigest()


def _validate_run_counts(embeddings, counts_path):
    """Check the selected cohort while keeping the verified run's native axes."""
    import anndata as ad
    from seareg.data import names, peak_keys

    counts = ad.read_h5ad(counts_path, backed='r')
    try:
        spots = names(counts.obs_names, 'selected counts spots')
        if set(spots) != set(embeddings.spot.index):
            raise ValueError('selected counts cohort does not match training spot IDs')
        known = set(peak_keys([p for p in counts.var_names if str(p).startswith('chr')]))
        if embeddings.peak is not None and not set(peak_keys(embeddings.peak.index)) <= known:
            raise ValueError('training peak IDs are absent from selected counts')
    finally:
        counts.file.close()


def load_tutorial_embeddings(source, *, counts_path):
    """Load one verified run or explicit native export against selected counts.

    Both sources return the existing EmbeddingSet contract. Native directories
    are self-described by export.json and do not need a tutorial dataset catalog.
    Loading never publishes, starts training, or switches to another source.
    """
    from seareg.data import TrainingRun, import_legacy, read_embeddings
    from seareg.utils import identity

    counts = Path(counts_path).expanduser().resolve()
    if not counts.is_file():
        raise FileNotFoundError(f'missing selected counts file: {counts}')
    if isinstance(source, TrainingRun):
        from seareg.exports import embedding_directory

        result = source.load_embeddings()
        _validate_run_counts(result, counts)
        directory = embedding_directory(source.path, source.final_epoch).resolve()
        sources = dict(result.provenance.get('sources', {}))
        sources['metadata'] = {'path': str(counts), 'sha256': _sha(counts)}
        result.provenance.update(
            embedding_origin='new_training', source='training_run',
            export_path=str(directory), export_sha256=_sha(directory / 'export.json'),
            training_executed=True, sources=sources,
            representation_and_epoch_evidence='verified_training_run')
        return result
    if not isinstance(source, (str, os.PathLike)):
        raise TypeError('embedding source must be a TrainingRun or native export directory')

    directory = Path(source).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f'native embedding source must be a directory: {directory}')
    export, entries, paths, export_sha256 = _native_export_entries(directory)
    # import_legacy aligns spots to counts order. Verify the native spot receipt
    # before that alignment so a valid reordered counts file remains usable.
    native_spots = read_embeddings(paths['spot'])
    if list(native_spots.shape) != entries['spot']['shape']:
        raise ValueError('native spot export shape mismatch')
    if identity(native_spots.index.tolist()) != entries['spot'].get('row_ids_hash'):
        raise ValueError('native spot export row ID fingerprint mismatch')
    del native_spots
    dna_kind = 'kmer' if 'kmer' in paths else 'motif'
    result = import_legacy(
        paths['spot'], paths['peak'], counts, motif_path=paths.get(dna_kind), dna_kind=dna_kind,
        representation='propagated', epoch=export['epoch'])
    for kind, entry in entries.items():
        frame = getattr(result, 'motif' if kind == 'kmer' else kind)
        if list(frame.shape) != entry['shape']:
            raise ValueError(f'native {kind} export shape mismatch')
        if kind != 'spot' and identity(frame.index.tolist()) != entry.get('row_ids_hash'):
            raise ValueError(f'native {kind} export row ID fingerprint mismatch')
        source_key = 'dna' if kind in ('motif', 'kmer') else kind
        if result.provenance['sources'][source_key]['sha256'] != entry['sha256']:
            raise ValueError(f'native {kind} export changed while loading')
    if _sha(directory / 'export.json') != export_sha256:
        raise ValueError('native export metadata changed while loading')
    result.provenance.update(
        embedding_origin='provided', source='native_export', export_path=str(directory),
        export_sha256=export_sha256,
        files_sha256={entry['path']: entry['sha256'] for entry in entries.values()},
        data_fingerprint=export.get('data_fingerprint'), training_executed=False,
        representation_and_epoch_evidence='verified_native_export_metadata',
        selected_model=export.get('selected_model'),
        model_protocol=export.get('model_protocol'),
        micro_formula_version=export.get('micro_formula_version'),
        reference=export.get('source'))
    return result
