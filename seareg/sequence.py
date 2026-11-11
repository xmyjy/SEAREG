"""Sequence artifacts derived only from a frozen filtered peak axis."""
from dataclasses import asdict
from pathlib import Path
import math
import re
import pandas as pd
from .data import peak_keys
from .utils import artifact_writer, file_sha256, identity, protect_output, write_json
from .artifacts import FilteredCounts, PeakSequences, stage_metadata,validated_operation

def _index_fasta(path):
    """Index a regular uncompressed FASTA in bounded memory, without editing it."""
    index = {}
    with Path(path).open('rb') as handle:
        current = None
        short_line = False
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.startswith(b'>'):
                current = line[1:].split()[0].decode('utf-8')
                if not current or current in index:
                    raise ValueError('FASTA chromosome names must be nonempty and unique')
                index[current] = [0, handle.tell(), 0, 0]
                short_line = False
            else:
                sequence = line.rstrip(b'\r\n')
                if current is None or not sequence or re.search(rb'[^ACGTNacgtn]', sequence):
                    raise ValueError('FASTA requires regular A/C/G/T/N sequence lines')
                entry = index[current]
                if not entry[2]:
                    entry[1:] = [offset, len(sequence), len(line)]
                elif short_line or len(sequence) > entry[2] or (len(sequence) == entry[2] and len(line) not in (entry[3], len(sequence))):
                    raise ValueError('FASTA line widths are irregular; normalize the local reference first')
                short_line = len(sequence) < entry[2]
                entry[0] += len(sequence)
    if not index or any(entry[0] == 0 for entry in index.values()):
        raise ValueError('FASTA contains an empty reference')
    return index


def _reference_index(path):
    fai = Path(str(path) + '.fai')
    if not fai.is_file():
        return _index_fasta(path)
    index = {}
    with fai.open() as stream:
        for line in stream:
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 5 or fields[0] in index:
                raise ValueError('invalid FASTA index')
            values = list(map(int, fields[1:5]))
            if min(values) < 0 or not values[0] or not values[2] or values[3] < values[2]:
                raise ValueError('invalid FASTA index coordinates')
            index[fields[0]] = values
    return index


def extract_peak_sequences(peak_ids, fasta):
    """Fetch final ordered zero-based, half-open BED intervals from local FASTA."""
    fasta = Path(fasta).resolve()
    with fasta.open('rb') as handle:
        if handle.read(1) != b'>':
            raise ValueError('FASTA must be uncompressed and begin with a header')
    index = _reference_index(fasta)
    result = []
    with fasta.open('rb') as handle:
        for peak in peak_keys(peak_ids):
            chrom, interval = peak.rsplit(':', 1)
            start, stop = map(int, interval.split('-'))
            if chrom not in index:
                raise ValueError(f'FASTA chromosome is missing: {chrom}')
            length, offset, bases, width = index[chrom]
            if start < 0 or stop > length:
                raise ValueError(f'peak is outside reference bounds: {peak}')
            first = offset + start // bases * width + start % bases
            last = offset + (stop - 1) // bases * width + (stop - 1) % bases
            handle.seek(first)
            sequence = handle.read(last - first + 1).replace(b'\n', b'').replace(b'\r', b'').decode('ascii').upper()
            if len(sequence) != stop - start or not set(sequence) <= set('ACGTN'):
                raise ValueError(f'FASTA index/sequence does not match requested peak: {peak}')
            result.append(sequence)
    return result


@validated_operation
def prepare_sequences(filtered, config, output):
    """Extract final ordered sequences once and freeze their pooled background."""
    from .config import SequenceConfig
    if not isinstance(config,SequenceConfig):raise TypeError('prepare_sequences requires SequenceConfig')
    filtered=FilteredCounts.load(filtered)
    if Path(output).exists():raise FileExistsError(output)
    fasta=Path(config.fasta).resolve();output=protect_output(output,[filtered.path,fasta])
    before=file_sha256(fasta);fai=Path(str(fasta)+'.fai')
    index_before=file_sha256(fai) if fai.is_file() else None
    sequences=extract_peak_sequences(filtered.peak_ids,fasta)
    counts={base:sum(sequence.count(base) for sequence in sequences) for base in 'ACGT'}
    total=sum(counts.values())
    if not total or min(counts['A']+counts['T'],counts['C']+counts['G'])==0:
        raise ValueError('pooled sequence background must contain both AT and GC bases')
    complement={'A':'T','T':'A','C':'G','G':'C'}
    background={base:(counts[base]+counts[complement[base]])/(2*total) for base in 'ACGT'}
    metadata=stage_metadata(output,peak_ids=filtered.peak_ids,
        peak_axis_hash=identity(filtered.peak_ids),sequence_axis_hash=identity([s for s in sequences]),
        source_filtered_fingerprint=filtered.fingerprint,assembly=config.assembly,
        assembly_verification='caller_declared_bound_to_reference_hash',
        reference={'sha256':before,'index_sha256':index_before},background=background,
        valid_bases=total,ambiguous_bases=sum(s.count('N') for s in sequences),
        protocol='pooled_complement_symmetric_v1')
    with artifact_writer(output,'peak_sequences',metadata) as stage:
        rows=[]
        with (stage/'peaks.fa').open('w') as fasta_out,(stage/'peaks.bed').open('w') as bed:
            for i,(peak,canonical,sequence) in enumerate(zip(filtered.peak_ids,peak_keys(filtered.peak_ids),sequences)):
                token=f'peak_{i:08d}'
                fasta_out.write(f'>{token}\n{sequence}\n')
                chrom,interval=canonical.rsplit(':',1);start,end=map(int,interval.split('-'))
                bed.write(f'{chrom}\t{start}\t{end}\n')
                rows.append({'peak_id':peak,'peak_index':i,'sequence_id':token,'chrom':chrom,
                             'start_0based':start,'end_0based':end,'length':len(sequence)})
        pd.DataFrame(rows).to_csv(stage/'peak_sequences.tsv',sep='\t',index=False)
        write_json(stage/'background.json',{'frequencies':background,'valid_bases':total,
                                          'protocol':metadata['protocol']})
        if file_sha256(fasta)!=before or (index_before is not None and file_sha256(fai)!=index_before):
            raise ValueError('reference changed during extraction')
    return PeakSequences.load(output)


# Compatibility adapters are intentionally thin; the algorithms live once.
from .scanners.legacy import _run_scanner, scan_annotation

def scan_peak_motifs_legacy(*args,**kwargs):
    from .compat import scan_legacy
    return scan_legacy(*args,**kwargs)

