"""Portable motif-input resolution and explicit-build reference preparation."""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

from .utils import file_sha256, write_json

_BUILDS = ('mm10', 'mm39', 'hg19', 'hg38')


def _genome_from_counts(path):
    """Read only scalar assembly metadata, never decode the count matrix."""
    import h5py
    if not Path(path).is_file():return None
    with h5py.File(path,'r') as handle:
        candidates=[]
        for key in ('genome','assembly','genome_assembly'):
            node=handle.get('uns/'+key)
            if isinstance(node,h5py.Dataset) and node.shape==():
                value=node[()]
                if isinstance(value,bytes):value=value.decode()
                if isinstance(value,str) and value.strip():candidates.append(value.strip())
    if len(set(candidates))>1:raise ValueError('conflicting genome/assembly metadata; declare the correct genome explicitly')
    return candidates[0] if candidates else None


def normalize_motif_request(raw, base):
    """Choose disabled/existing/automatic/explicit once, with no output writes."""
    raw=copy.deepcopy(raw)
    existing=raw.get('input',{})
    # Direct entry points explicitly choose True by default. Old JSON files
    # without the new switch retain their former counts-only/no-motif meaning.
    previous=bool(existing.get('dna') or raw.get('annotation') is not None or
        existing.get('network') or raw.get('network',{}).get('prepared'))
    option=raw.get('motif',previous)
    if type(option) is bool:options={'enabled':option}
    elif isinstance(option,dict):options=dict(option)
    else:raise ValueError('motif must be true/false or a motif options object')
    allowed={'enabled','auto_build_dna','genome','fasta','rscript','threads','species_names','download_reference','source'}
    if set(options)-allowed:raise ValueError('unknown motif options')
    for name in ('enabled','auto_build_dna','download_reference'):
        if name in options and type(options[name]) is not bool:raise ValueError(f'motif.{name} must be boolean')
    enabled=options.get('enabled',True)
    source_mode=options.get('source',raw.get('motif_source','auto'))
    if source_mode not in ('auto','read','scan'):raise ValueError('motif_source must be auto, read or scan')
    report={'enabled':enabled,'auto_build_dna':options.get('auto_build_dna',True),
            'source':source_mode,'mode':'disabled'}
    inputs=raw.setdefault('input',{})
    def resolve(value):
        if not isinstance(value,str) or not value.strip():raise ValueError('input/reference paths must be nonempty strings')
        return (base/value).resolve()
    if not enabled:
        inputs.pop('dna',None)
        for key in ('annotation','edge_selection','sequence'):raw.pop(key,None)
        raw['motif']=report
        return raw,None
    # Explicit annotation is an advanced opt-in; never substitute FIMO for R.
    if raw.get('annotation') is not None:
        if source_mode!='auto':raise ValueError('motif_source is for the automatic legacy entry; do not combine it with explicit annotation')
        report['mode']='explicit_'+str(raw['annotation'].get('method'))
        raw['motif']=report
        return raw,None
    if 'network' in inputs or raw.get('network',{}).get('prepared'):
        report['mode']='frozen_network';raw['motif']=report
        return raw,None
    source=inputs.get('atac')
    if source is None:
        # A filtered artifact can still be used, but requires explicit sequence
        # metadata if no motif matrix was supplied.
        source=inputs.get('filtered')
    if source is None:raw['motif']=report;return raw,None
    parent=resolve(source).parent
    candidates=([resolve(inputs['dna'])] if 'dna' in inputs else
        [parent/'motifs.h5',parent/'freq_motif.h5',parent/'output_kmers_motifs/freq_motif.h5'])
    found=next((path for path in candidates if path.is_file()),None) if source_mode!='scan' else None
    if found is not None:
        inputs['dna']=str(found);report.update(mode='existing_matrix',matrix=str(found))
        raw['motif']=report
        return raw,None
    if source_mode=='read' or not report['auto_build_dna']:
        reason='motif_source=read' if source_mode=='read' else 'auto_build_dna=False'
        raise FileNotFoundError(f'motif matrix missing with {reason}; supply input.dna or set motif=False')
    sequence=raw.get('sequence',{})
    genome=options.get('genome') or sequence.get('assembly') or _genome_from_counts(resolve(source))
    if not isinstance(genome,str) or not genome.strip():
        # Reading/filtering the counts is useful before the user chooses how
        # to supply motif information. Refuse only at the motif boundary.
        inputs.pop('dna',None)
        report.update(mode='needs_genome',reason='motif=True needs genome/assembly when no matrix exists; pass genome="mm10"/"hg38" (or the correct build), provide uns["genome"], or use motif=False')
        raw['motif']=report
        return raw,None
    genome=genome.strip()
    if any(ord(c)<32 for c in genome):raise ValueError('invalid genome/assembly')
    explicit_fasta=options.get('fasta') or sequence.get('fasta')
    managed=None
    if explicit_fasta:
        fasta=resolve(explicit_fasta)
    else:
        local=[parent/(genome+'.fa'),parent/'ref_genomes'/(genome+'.fa')]
        fasta=next((path for path in local if path.is_file()),None)
        if fasta is None:
            if genome not in _BUILDS:
                raise ValueError(f'genome {genome!r} needs an explicit matching FASTA; automatic builds: {", ".join(_BUILDS)}')
            output=resolve(raw.get('output','seareg_results'))
            fasta=output/'resources'/genome/(genome+'.fa')
            managed={'assembly':genome,'fasta':str(fasta),'download':options.get('download_reference',True)}
    inputs.pop('dna',None)
    output=resolve(raw.get('output','seareg_results'))
    annotation={'method':'legacy','cache_dir':str(output/'resources/BiocFileCache'),
        **{key:options[key] for key in ('rscript','threads','species_names') if key in options}}
    # The R wrapper accepts exactly the same historical scientific recipe.
    if raw.get('schema_version',1)==1:annotation.update(fasta=str(fasta),assembly=genome)
    else:raw['sequence']={'fasta':str(fasta),'assembly':genome}
    raw['annotation']=annotation
    report.update(mode='auto_legacy',genome=genome,fasta=str(fasta),
                  requested_matrix=str(candidates[0]) if candidates else None)
    raw['motif']=report
    return raw,managed


def _fetch(url,destination):
    with urlopen(url,timeout=120) as source,Path(destination).open('wb') as target:
        shutil.copyfileobj(source,target,1024*1024)


def ensure_reference(request):
    """Download only a declared build; checksum, gzip CRC and atomic publish."""
    if request is None:return
    fasta=Path(request['fasta']);assembly=request['assembly'];receipt=fasta.parent/'reference.json'
    if fasta.is_file():
        if not receipt.is_file():raise ValueError('managed reference has no download receipt; supply it as an explicit FASTA instead')
        saved=json.loads(receipt.read_text())
        if saved.get('assembly')!=assembly or saved.get('fasta_sha256')!=file_sha256(fasta):
            raise ValueError('cached reference identity changed')
        return
    if not request['download']:raise FileNotFoundError('reference FASTA missing and download_reference=False')
    if assembly not in _BUILDS:raise ValueError('unsupported automatic reference assembly')
    fasta.parent.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (fasta.parent/'download.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if fasta.is_file():return ensure_reference(request)
        prefix=f'https://hgdownload.soe.ucsc.edu/goldenPath/{assembly}/bigZips/'
        filename=assembly+'.fa.gz'
        with tempfile.TemporaryDirectory(prefix='download-',dir=fasta.parent) as temporary:
            temp=Path(temporary);checksums=temp/'md5sum.txt';archive=temp/filename
            _fetch(prefix+'md5sum.txt',checksums)
            records={parts[1].lstrip('*'):parts[0] for line in checksums.read_text().splitlines() if len(parts:=line.split())==2}
            expected=records.get(filename)
            if expected is None:raise ValueError('official checksum list does not identify the requested FASTA')
            _fetch(prefix+filename,archive)
            digest=hashlib.md5()
            with archive.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
            if digest.hexdigest()!=expected:raise ValueError('downloaded reference checksum mismatch')
            extracted=temp/(assembly+'.fa')
            with gzip.open(archive,'rb') as source,extracted.open('wb') as target:
                shutil.copyfileobj(source,target,1024*1024)
            with extracted.open('rb') as stream:
                if stream.read(1)!=b'>':raise ValueError('download is not a FASTA reference')
            metadata={'assembly':assembly,'url':prefix+filename,'compressed_md5':expected,
                'fasta_sha256':file_sha256(extracted),'bytes':extracted.stat().st_size}
            write_json(receipt,metadata)
            os.replace(extracted,fasta)
