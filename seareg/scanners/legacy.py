"""Explicit, transactional sequence annotation using a caller-selected scanner.

The assembly is an explicit declaration bound to a FASTA SHA-256, not inferred
from species or a filename. This adapter never downloads or substitutes a
reference. Scanner outputs are annotations; selecting a training DNA graph is a
separate preprocessing step."""

from pathlib import Path
from importlib import resources
import json
import math
import os
import signal
import shutil
import subprocess
import sqlite3
import tempfile
import time
from ..data import PreparedData, PreparedNetwork, canonical_csr, names
from ..utils import artifact_writer, attempt_writer, environment, file_sha256, protect_output, write_json
from ..data import peak_keys
from ..compat import _read_dna


def _argv(values, label, *, nonempty=False):
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{label} must be an explicit list or tuple of argument strings")
    result = list(values)
    if nonempty and not result:
        raise ValueError(f"{label} requires an executable prefix")
    if any(not isinstance(v, str) or not v or "\x00" in v for v in result):
        raise ValueError(f"{label} contains an invalid argument")
    return result


def _tail(path, limit=4096):
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode("utf-8", errors="replace")


def _run_scanner(argv, stdout_path, stderr_path, timeout, *, env=None):
    """Keep logs on disk and terminate scanner workers if the deadline expires."""
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            process = subprocess.Popen(
                argv,
                stdout=stdout,
                stderr=stderr,
                shell=False,
                env=env,
                start_new_session=(os.name == "posix"),
            )
        except OSError as error:
            raise RuntimeError(f"scanner could not execute: {error}") from error
        try:
            returncode = process.wait(timeout=timeout)
        except BaseException as error:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
            if isinstance(error, subprocess.TimeoutExpired):
                raise RuntimeError(f"scanner timed out after {timeout} seconds") from error
            raise
    if returncode:
        raise RuntimeError(f"scanner failed with exit code {returncode}: {_tail(stderr_path)}")
    return returncode


def _legacy_cache(destination):
    """Use an owned R cache; seed only an already-readable JASPAR2024 entry.

    The R package itself downloads its database when no local entry is present.
    Never write to the original home cache or copy unrelated cached resources.
    """
    chosen=os.environ.get('BFC_CACHE')
    destination=Path(chosen if chosen else destination).expanduser().resolve()
    destination.mkdir(parents=True,exist_ok=True)
    if chosen:return destination
    source=Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'R/BiocFileCache'
    if source.resolve()==destination:return destination
    import fcntl
    with (destination/'seed.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (destination/'BiocFileCache.sqlite').exists():return destination
        database=source/'BiocFileCache.sqlite'
        if not database.is_file():return destination
        try:
            with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as original:
                columns={row[1] for row in original.execute('pragma table_info(resource)')}
                fields='rname,rpath,rtype' if 'rtype' in columns else 'rname,rpath'
                rows=original.execute('select '+fields+' from resource where rname like ?',('%/JASPAR2024.sqlite',)).fetchall()
                files=[]
                for row in rows:
                    rpath=row[1]
                    path=Path(rpath)
                    if not path.is_absolute():
                        if path.name!=rpath:return destination
                        path=source/path
                    path=path.resolve()
                    if not path.is_relative_to(source.resolve()) or not path.is_file():return destination
                    files.append(path)
                if not files or len({path.name for path in files})!=len(files):return destination
                with tempfile.TemporaryDirectory(prefix='seed-',dir=destination) as temporary:
                    staged=Path(temporary)/'BiocFileCache.sqlite'
                    with sqlite3.connect(staged) as copy:
                        original.backup(copy)
                        copy.execute('delete from resource where rname not like ?',('%/JASPAR2024.sqlite',))
                        for row,path in zip(rows,files):
                            rname,rpath=row[:2]
                            # BiocFileCache joins web resource paths to its cache;
                            # local resource paths are already absolute.
                            copied=path.name if len(row)>2 and row[2]=='web' else str(destination/path.name)
                            copy.execute('update resource set rpath=? where rname=? and rpath=?',
                                         (copied,rname,rpath))
                    for path in files:shutil.copy2(path,destination/path.name)
                    os.replace(staged,destination/'BiocFileCache.sqlite')
        except (OSError,sqlite3.Error):
            # An unreadable/older cache is optional, not a reason to prevent
            # the R package from preparing its own database in the owned cache.
            pass
    return destination


def scan_annotation(
    prepared: PreparedData,
    fasta: Path,
    assembly: str,
    command: list[str],
    output: Path,
    scanner_args: tuple[str, ...] = (),
    timeout: float = 3600,
) -> Path:
    """Scan the final ordered peaks and exclusively publish validated annotations.

    ``command`` is a trusted executable prefix, for example
    ``['Rscript', '/path/scanner.R']``. The scanner must accept ``-i`` (three-column
    BED), ``-g`` (FASTA) and ``-o`` (output directory), and produce at least one of
    ``freq_motif.h5`` or ``freq_kmer.h5`` with named ``mat/row_names/col_names``.
    Peak rows must match the generated BED order after interval normalization.
    Optional arguments cannot replace the generated short input/output flags.
    """
    return _scan_annotation(
        prepared, fasta, assembly, command, output, scanner_args, timeout
    )


def _scan_annotation(
    prepared, fasta, assembly, command, output, scanner_args, timeout,
    *, scanner_metadata=None, input_source=None, filtered=None, scanner_env=None,
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not isinstance(prepared, PreparedData):
        raise ValueError("prepared must be validated PreparedData")
    if not isinstance(assembly, str) or not assembly.strip() or any(ord(c) < 32 for c in assembly):
        raise ValueError("assembly identity must be an explicit nonempty build tag")
    assembly = assembly.strip()
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("scanner timeout must be finite and positive")
    prefix = _argv(command, "scanner command", nonempty=True)
    extra = _argv(scanner_args, "scanner_args")
    if any(arg.startswith(("-i", "-g", "-o")) and not arg.startswith("--") for arg in extra):
        raise ValueError("scanner_args cannot override reserved -i/-g/-o arguments")
    fasta = Path(fasta).resolve()
    if not fasta.is_file():
        raise FileNotFoundError(f"FASTA reference is missing: {fasta}")
    with fasta.open("rb") as stream:
        if stream.read(1) != b">":
            raise ValueError(f"FASTA must be a nonempty, uncompressed reference: {fasta}")
    expected_peaks = peak_keys(prepared.peak_ids)
    prepared_fingerprint = prepared.fingerprint
    fasta_hash = file_sha256(fasta)
    index = Path(str(fasta)+'.fai')
    index_hash = file_sha256(index) if index.is_file() else None
    metadata = {
        "protocol": "explicit_sequence_annotation_v1",
        "data_fingerprint": prepared_fingerprint,
        "assembly": assembly,
        "assembly_verification": "caller_declared_build_bound_to_fasta_sha256",
        "fasta": {"path": str(fasta), "sha256": fasta_hash},
        "fasta_index_sha256": index_hash,
        "command_prefix": prefix,
        "scanner_args": extra,
        "timeout_seconds": timeout,
        "environment": environment(),
    }
    from ..artifacts import stage_metadata
    from ..utils import identity
    executable=shutil.which(prefix[0])
    metadata.update(stage_metadata(output,upstreams={} if filtered is None else {'filtered':filtered},
        peak_ids=prepared.peak_ids,
        peak_axis_hash=identity(prepared.peak_ids),
        scanner_runtime={'sha256':file_sha256(executable) if executable is not None else None}))
    if scanner_metadata is not None:
        metadata["scanner"] = scanner_metadata
    with attempt_writer(output) as attempt, artifact_writer(output, "sequence_annotation", metadata) as stage:
        if scanner_metadata is not None:
            write_json(stage / "scanner.provenance.json", scanner_metadata)
        bed = stage / "peaks.bed"
        with bed.open("w", encoding="utf-8") as stream:
            for peak in expected_peaks:
                chrom, interval = peak.rsplit(":", 1)
                start, end = interval.split("-", 1)
                stream.write(f"{chrom}\t{start}\t{end}\n")
        # bedtools may create a .fai. Keep that side effect in private staging,
        # not beside the caller's potentially read-only reference genome.
        scanner_fasta=stage/'scanner_reference.fa'
        scanner_fasta.symlink_to(fasta)
        scanner_index=Path(str(scanner_fasta)+'.fai')
        if index.is_file():shutil.copy2(index,scanner_index)
        argv = prefix + ["-i", str(bed), "-g", str(scanner_fasta), "-o", str(stage)] + extra
        metadata["command"] = argv
        write_json(attempt/'command.json', {'argv':argv})
        metadata["bed_sha256"] = file_sha256(bed)
        started = time.monotonic()
        try:
            kwargs={} if scanner_env is None else {'env':scanner_env}
            metadata["returncode"] = _run_scanner(
                argv, attempt / "scanner.stdout.log", attempt / "scanner.stderr.log", timeout, **kwargs)
        finally:
            scanner_fasta.unlink(missing_ok=True)
            scanner_index.unlink(missing_ok=True)
        for name in ('scanner.stdout.log', 'scanner.stderr.log'):
            shutil.copy2(attempt/name, stage/name)
        metadata["elapsed_seconds"] = time.monotonic() - started
        outputs = []
        for filename in ("freq_motif.h5", "freq_kmer.h5"):
            path = stage / filename
            if not path.exists():
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"invalid DNA annotation output file: {filename}")
            matrix, row_names, column_names = _read_dna(path)
            if peak_keys(row_names) != expected_peaks:
                raise ValueError(
                    f"DNA annotation peak row order/identity differs from final BED: {filename}"
                )
            dna_ids = names(column_names, "DNA annotation columns")
            if not dna_ids:
                raise ValueError(f"DNA annotation output has no named columns: {filename}")
            x = canonical_csr(matrix)
            if x.shape != (len(expected_peaks), len(dna_ids)):
                raise ValueError(f"DNA annotation output shape mismatch: {filename}")
            outputs.append(
                {
                    "file": filename,
                    "shape": list(x.shape),
                    "nnz": int(x.nnz),
                    "sha256": file_sha256(path),
                }
            )
        if not outputs:
            raise RuntimeError(
                "scanner produced no freq_motif.h5 or freq_kmer.h5 annotation output"
            )
        if file_sha256(fasta) != fasta_hash:
            raise ValueError("FASTA reference changed during sequence annotation")
        if (file_sha256(index) if index.is_file() else None) != index_hash:
            raise ValueError('FASTA index changed during sequence annotation')
        if executable is not None and file_sha256(executable)!=metadata['scanner_runtime']['sha256']:
            raise ValueError('scanner executable changed during sequence annotation')
        if prepared.fingerprint != prepared_fingerprint:
            raise ValueError("prepared input changed during sequence annotation")
        if input_source is not None:
            path, digest = input_source
            if file_sha256(path) != digest:
                raise ValueError("ATAC input changed during sequence annotation")
        metadata["annotation_outputs"] = outputs
    return output


def scan_peak_motifs_legacy(
    atac,
    fasta,
    assembly,
    output,
    *,
    preprocessing=None,
    species_names=("Homo sapiens", "Mus musculus"),
    p_cutoff=1e-5,
    threads=1,
    rscript="Rscript",
    timeout=3600,
    cache_dir=None,
    _filtered=None,
):
    """Generate a peak–motif training graph with the packaged scanner.

    ``atac`` can be a counts H5AD path, validated ``PreparedData``, a
    ``PreparedNetwork``, or its directory. Counts are filtered with the supplied
    ``PreprocessConfig`` before the BED is written; prepared inputs already fix
    the final ordered peak axis and cannot take another preprocessing config.

    The JASPAR2024/motifmatchr scanner uses the packaged recipe.
    Its weighted scores, frequency filters and two Top-K steps produce a pruned
    training graph, rather than a complete motif-site annotation or a set of
    statistically corrected high-confidence relations. R/Bioconductor packages,
    bedtools and an assembly-matched uncompressed FASTA must be available.
    Returns the new directory containing ``freq_motif.h5`` and provenance.
    """
    output = protect_output(output)
    if output.exists():
        raise FileExistsError(output)
    if (
        isinstance(p_cutoff, bool)
        or not isinstance(p_cutoff, (int, float))
        or not math.isfinite(p_cutoff)
        or not 0 < p_cutoff <= 1
    ):
        raise ValueError("p_cutoff must be finite and in (0, 1]")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    species = _argv(species_names, "species_names", nonempty=True)
    if any("," in item or item != item.strip() or any(ord(c) < 32 for c in item)
           for item in species):
        raise ValueError("species_names must contain separate nonempty species names")
    executable = os.fspath(rscript)
    _argv([executable], "Rscript executable", nonempty=True)
    if not isinstance(atac,PreparedData):
        raise TypeError('legacy scanner core requires a frozen PreparedData; use the public compatibility API')
    data=atac;input_source=None
    input_metadata={'kind':'prepared_data','data_fingerprint':data.fingerprint}
    if preprocessing is not None:
        raise ValueError("prepared inputs already fix the peak axis; omit preprocessing")
    root = resources.files("seareg").joinpath("scanners")
    recipe = json.loads(root.joinpath("legacy_motif_recipe.json").read_text(encoding="utf-8"))
    parameters = dict(recipe["parameters"])
    parameters.update(species_names=species, p_cutoff=float(p_cutoff), threads=threads)
    extra = ["--no_kmer", "--species_names", ",".join(species)]
    for key, value in parameters.items():
        if key in ("no_kmer", "species_names"):
            continue
        if isinstance(value, bool):
            if value:
                extra.append("--" + key)
        else:
            extra.extend(("--" + key, str(value)))
    scanner_metadata = {
        **recipe, "parameters": parameters, "input": input_metadata,
    }
    cache=_legacy_cache(cache_dir if cache_dir is not None else output.parent/'resources/BiocFileCache')
    scanner_metadata['cache_dir']=str(cache)
    scanner_env=dict(os.environ,BFC_CACHE=str(cache))
    with resources.as_file(root.joinpath(recipe["script"])) as script:
        scanner_metadata['script_sha256'] = file_sha256(script)
        return _scan_annotation(
            data, fasta, assembly, [executable, "--vanilla", str(script)],
            output, tuple(extra), timeout,
            scanner_metadata=scanner_metadata, input_source=input_source,filtered=_filtered,
            scanner_env=scanner_env,
        )
