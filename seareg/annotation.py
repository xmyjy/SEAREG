"""Complete, provenance-bound motif sites on an immutable sequence artifact."""
import csv
from dataclasses import dataclass,asdict
import gzip
import math
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile
import numpy as np
import pandas as pd
import scipy.sparse as sp
from .artifacts import StageArtifact,PeakSequences,stage_metadata,validated_operation
from .data import names,peak_keys
from .sequence import extract_peak_sequences
from .utils import artifact_writer,attempt_writer as _attempt,file_sha256,identity,protect_output,write_json,verify_artifact


@dataclass(frozen=True)
class AnnotationResult(StageArtifact):
    @classmethod
    def load(cls,path):
        result=super().load(path)
        if result.manifest['kind'] not in ('peak_motif_annotation','sequence_annotation'):
            raise ValueError('artifact is not a motif annotation')
        return result

    @property
    def motif_ids(self):
        self.verify()
        if self.manifest['kind']=='sequence_annotation':
            from .preprocessing import _read_dna
            return names(_read_dna(self.path/'freq_motif.h5')[2],'motif IDs')
        return tuple(pd.read_csv(self.path/'motif_ids.tsv',sep='\t',dtype={'motif_id':str},
                                  keep_default_na=False).motif_id)

    @property
    def peak_ids(self):
        if 'peak_ids' in self.metadata:return tuple(self.metadata['peak_ids'])
        return tuple(pd.read_csv(self.path/'peak_ids.tsv',sep='\t',dtype={'peak_id':str},
                                 keep_default_na=False).peak_id)

    @property
    def sites_path(self):return self.path/'motif_sites.tsv.gz'

    @property
    def relations_path(self):return self.path/'peak_motif_relations.tsv.gz'

    @validated_operation
    def relations(self):
        self.verify()
        result=pd.read_csv(self.relations_path,sep='\t',dtype={'peak_id':str,'motif_id':str},keep_default_na=False)
        if 'relation_index' not in result:result.insert(0,'relation_index',np.arange(len(result)))
        return result

    @validated_operation
    def load_matches(self):
        self.verify()
        if self.manifest['kind']!='peak_motif_annotation':
            raise ValueError('legacy graph is not a complete site annotation')
        if (self.path/'candidate_matches.npz').is_file():return sp.load_npz(self.path/'candidate_matches.npz')
        pairs=self.relations();pi={p:i for i,p in enumerate(self.peak_ids)};mi={m:i for i,m in enumerate(self.motif_ids)}
        return sp.csr_matrix((np.ones(len(pairs),np.float32),
            ([pi[p] for p in pairs.peak_id],[mi[m] for m in pairs.motif_id])),shape=(len(pi),len(mi)))


MotifAnnotation=AnnotationResult


def _inputs(sequences,config,output):
    from .config import AnnotationConfig
    if Path(output).exists():raise FileExistsError(output)
    if not isinstance(config,AnnotationConfig):raise TypeError('annotation requires AnnotationConfig')
    sequences=PeakSequences.load(sequences)
    motifs=Path(config.motifs).resolve()
    output=protect_output(output,[sequences.path,motifs])
    axis=_motif_axis(motifs)
    metadata=stage_metadata(output,upstreams={'sequences':sequences},peak_ids=sequences.peak_ids,
        peak_axis_hash=sequences.metadata['peak_axis_hash'],motif_axis_hash=identity([a['motif_id'] for a in axis]),
        protocol='fimo_complete_candidates_v2',assembly=sequences.metadata['assembly'],
        background=sequences.metadata['background'],p_cutoff=config.p_cutoff,
        motif_pseudocount=config.motif_pseudocount,both_strands=True,
        motifs={'sha256':file_sha256(motifs)},confidence='sequence candidates; no global confidence rule')
    return sequences,motifs,axis,output,metadata


def _consume_sites(stage,sequences,axis,hits,metadata):
    """Stream sites and use a disk-backed unique index instead of all-hit sets."""
    peak_ids=sequences.peak_ids;seq=sequences.read_sequences()
    motif_ids=tuple(a['motif_id'] for a in axis);mi={m:i for i,m in enumerate(motif_ids)}
    pi={p:i for i,p in enumerate(peak_ids)}
    pi.update({f'peak_{i:08d}':i for i in range(len(peak_ids))})
    starts=[int(p.rsplit(':',1)[1].split('-')[0]) for p in peak_keys(peak_ids)]
    fields=['site_index','peak_id','peak_index','motif_id','motif_index','motif_name','motif_alt_id',
            'start_1based','stop_1based','strand','score','site_p','q_value','matched_sequence',
            'genomic_start_0based','genomic_end_0based']
    rows=0
    with tempfile.TemporaryDirectory(prefix='seareg-sites-',dir=stage.parent) as scratch:
        with sqlite3.connect(str(Path(scratch)/'sites.sqlite')) as db:
            db.execute('CREATE TABLE hits(p INTEGER,m INTEGER,start INTEGER,stop INTEGER,strand TEXT,pval REAL, PRIMARY KEY(p,m,start,stop,strand)) WITHOUT ROWID')
            pending=[]
            with Path(hits).open() as source,gzip.open(stage/'motif_sites.tsv.gz','wt',newline='') as target:
                reader=csv.DictReader((line for line in source if line.strip() and not line.startswith('#')),delimiter='\t')
                required={'motif_id','sequence_name','start','stop','strand','score','p-value','matched_sequence'}
                if not required<=set(reader.fieldnames or []):raise ValueError('FIMO header lacks required site fields')
                writer=csv.DictWriter(target,fieldnames=fields,delimiter='\t');writer.writeheader()
                for row in reader:
                    if row['motif_id'] not in mi or row['sequence_name'] not in pi:
                        raise ValueError('unknown peak or motif in FIMO site')
                    i,j=pi[row['sequence_name']],mi[row['motif_id']]
                    start,stop=int(row['start']),int(row['stop']);p=float(row['p-value']);score=float(row['score'])
                    strand=row['strand']
                    if not 1<=start<=stop<=len(seq[i]) or stop-start+1!=axis[j]['width']:
                        raise ValueError('FIMO site coordinates/width outside reference sequence')
                    if strand not in ('+','-') or not math.isfinite(score) or not math.isfinite(p) or not 0<=p<=metadata['p_cutoff']:
                        raise ValueError('invalid FIMO strand/score/P-value')
                    matched=seq[i][start-1:stop]
                    if strand=='-':matched=matched.translate(str.maketrans('ACGTN','TGCAN'))[::-1]
                    if matched!=row['matched_sequence'].upper():raise ValueError('FIMO matched sequence disagrees with reference')
                    writer.writerow(dict(zip(fields,[rows,peak_ids[i],i,motif_ids[j],j,axis[j]['motif_name'],
                        row.get('motif_alt_id','UNKNOWN'),start,stop,strand,row['score'],row['p-value'],
                        row.get('q-value','UNKNOWN'),matched,starts[i]+start-1,starts[i]+stop])))
                    pending.append((i,j,start,stop,strand,p));rows+=1
                    if len(pending)>=4096:
                        db.executemany('INSERT INTO hits VALUES(?,?,?,?,?,?) ON CONFLICT(p,m,start,stop,strand) DO UPDATE SET pval=MIN(pval,excluded.pval)',pending)
                        db.commit();pending=[]
            if pending:
                db.executemany('INSERT INTO hits VALUES(?,?,?,?,?,?) ON CONFLICT(p,m,start,stop,strand) DO UPDATE SET pval=MIN(pval,excluded.pval)',pending)
            db.commit()
            query='''SELECT h.p,h.m,MIN(h.pval),physical.n,COUNT(*) FROM hits h JOIN
                (SELECT p,m,COUNT(*) n FROM (SELECT DISTINCT p,m,start,stop FROM hits) GROUP BY p,m) physical
                ON h.p=physical.p AND h.m=physical.m GROUP BY h.p,h.m ORDER BY h.p,h.m'''
            peak_counts=np.zeros(len(peak_ids),dtype=np.int64);motif_counts=np.zeros(len(motif_ids),dtype=np.int64)
            count=0;physical_count=0
            with gzip.open(stage/'peak_motif_relations.tsv.gz','wt',newline='') as stream:
                writer=csv.writer(stream,delimiter='\t')
                writer.writerow(['relation_index','peak_index','motif_index','peak_id','motif_id',
                                 'best_site_p','physical_site_count','strand_site_count'])
                for i,j,best,physical,strands in db.execute(query):
                    writer.writerow([count,i,j,peak_ids[i],motif_ids[j],repr(best),physical,strands])
                    peak_counts[i]+=1;motif_counts[j]+=1;physical_count+=physical;count+=1
    pd.DataFrame({'peak_id':peak_ids,'sequence_id':[f'peak_{i:08d}' for i in range(len(peak_ids))],
                  'sequence_length':list(map(len,seq)),'matched_motifs':peak_counts}).to_csv(stage/'peak_ids.tsv',sep='\t',index=False)
    motifs=pd.DataFrame(axis);motifs['matched_peaks']=motif_counts
    motifs.to_csv(stage/'motif_ids.tsv',sep='\t',index=False)
    metadata.update(shape=[len(peak_ids),len(motif_ids)],candidate_pairs=count,strand_hit_rows=rows,
        physical_sites=physical_count,unmatched_peaks=int((peak_counts==0).sum()),
        unmatched_motifs=int((motif_counts==0).sum()),site_storage='single_normalized_table',
        pair_aggregation='disk_backed_unique_site_index')
    write_json(stage/'annotation.provenance.json',metadata)


@validated_operation
def annotate_motifs(sequences,config,output,*,attempts_dir=None):
    from .scanners.legacy import _run_scanner
    sequences,motifs,axis,output,metadata=_inputs(sequences,config,output)
    executable=shutil.which(config.fimo)
    if executable is None:raise FileNotFoundError(f'FIMO executable not found: {config.fimo}')
    before=file_sha256(executable)
    version=subprocess.run([executable,'--version'],capture_output=True,text=True,check=True,timeout=30)
    metadata['scanner']={'sha256':before,'version':(version.stdout+version.stderr).strip()}
    with _attempt(output,attempts_dir) as attempt:
        bg=attempt/'background.txt'
        bg.write_text(''.join(f'{base} {p:.17g}\n' for base,p in metadata['background'].items()))
        hits=attempt/'scanner.stdout.tsv';stderr=attempt/'scanner.stderr.log'
        argv=[executable,'--text','--thresh',str(config.p_cutoff),'--bgfile',str(bg),
              '--motif-pseudo',str(config.motif_pseudocount),'--verbosity','1',str(motifs),str(sequences.path/'peaks.fa')]
        write_json(attempt/'command.json',{'argv':argv});metadata['command_options']=asdict(config)
        _run_scanner(argv,hits,stderr,config.timeout)
        with artifact_writer(output,'peak_motif_annotation',metadata) as stage:
            _consume_sites(stage,sequences,axis,hits,metadata)
            shutil.copy2(stderr,stage/'scanner.stderr.log')
            (stage/'scanner.stdout.log').write_text('FIMO stdout normalized into motif_sites.tsv.gz\nsha256='+file_sha256(hits)+'\n')
            if file_sha256(motifs)!=metadata['motifs']['sha256'] or file_sha256(executable)!=before:
                raise ValueError('motif resource/scanner changed during annotation')
        # The canonical site table contains all reported fields; no second large stdout table.
        hits.unlink()
    return AnnotationResult.load(output)


@validated_operation
def import_motif_sites(sequences,config,hits,output):
    sequences,motifs,axis,output,metadata=_inputs(sequences,config,output)
    hits=Path(hits).resolve();before=file_sha256(hits)
    metadata['imported_hits']={'sha256':before,'exhaustiveness':'caller_declared_unverified'}
    with _attempt(output) as attempt:
        with artifact_writer(output,'peak_motif_annotation',metadata) as stage:
            _consume_sites(stage,sequences,axis,hits,metadata)
            if file_sha256(hits)!=before or file_sha256(motifs)!=metadata['motifs']['sha256']:
                raise ValueError('annotation resource changed during import')
    return AnnotationResult.load(output)


def _motif_axis(path):
    text = Path(path).read_text(encoding='utf-8')
    if not text.startswith('MEME version') or not re.search(r'ALPHABET\s*=\s*ACGT', text):
        raise ValueError('motifs must be a local DNA MEME-format resource')
    motifs = []
    for block in re.split(r'^MOTIF\s+', text, flags=re.MULTILINE)[1:]:
        lines = block.splitlines()
        fields = lines[0].split(maxsplit=1)
        motif_id, label = fields[0], fields[1] if len(fields) == 2 else fields[0]
        match = re.search(r'letter-probability matrix:[^\n]*\bw\s*=\s*(\d+)', block)
        if match is None or int(match[1]) < 1:
            raise ValueError(f'motif has no valid probability matrix width: {motif_id}')
        width = int(match[1])
        rows = block[match.end():].splitlines()[1:width + 1]
        try:
            probabilities = np.array([list(map(float, row.split())) for row in rows])
        except ValueError as exc:
            raise ValueError(f'invalid motif probabilities: {motif_id}') from exc
        if probabilities.shape != (width, 4) or not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-4):
            raise ValueError(f'invalid motif probabilities: {motif_id}')
        motifs.append({'motif_id': motif_id, 'motif_name': label, 'width': width})
    names([item['motif_id'] for item in motifs], 'motif IDs')
    if not motifs:
        raise ValueError('MEME resource contains no motifs')
    return motifs


def annotate_peak_motifs(atac,fasta,assembly,motifs,output,**kwargs):
    from .compat import annotate_legacy_api
    return annotate_legacy_api(atac,fasta,assembly,motifs,output,**kwargs)


def import_fimo_annotation(atac,fasta,assembly,motifs,hits,output,**kwargs):
    from .compat import annotate_legacy_api
    return annotate_legacy_api(atac,fasta,assembly,motifs,output,hits=hits,**kwargs)


def export_motif_graph(annotation,output,*,selection,p_cutoff=None):
    from .compat import export_selected_h5
    return export_selected_h5(annotation,output,selection=selection,p_cutoff=p_cutoff)
