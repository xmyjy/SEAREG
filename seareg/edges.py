"""Explicit candidate edge selection; model-node compilation happens later."""
from dataclasses import asdict,dataclass
from pathlib import Path
from typing import ClassVar
import numpy as np
import pandas as pd
import scipy.sparse as sp
from .artifacts import StageArtifact,FilteredCounts,stage_metadata,validated_operation
from .utils import artifact_writer,identity,write_json,protect_output,file_sha256


@dataclass(frozen=True)
class SelectedMotifEdges(StageArtifact):
    KIND: ClassVar[str]='selected_motif_edges'

    @property
    def motif_ids(self):return tuple(self.metadata['motif_ids'])

    @validated_operation
    def edges(self):
        self.verify()
        if (self.path/'selected_edges.tsv.gz').is_file():
            return pd.read_csv(self.path/'selected_edges.tsv.gz',sep='\t',
                dtype={'peak_id':str,'motif_id':str},keep_default_na=False)
        from .annotation import AnnotationResult
        annotation=self.upstream('annotation',AnnotationResult)
        result=annotation.relations()
        if self.metadata['selection']!='all_candidates':raise ValueError('missing selected edge table')
        result['raw_weight']=np.float32(1)
        return result

    @validated_operation
    def load_matrix(self):
        edges=self.edges();pi={name:i for i,name in enumerate(self.peak_ids)};mi={name:i for i,name in enumerate(self.motif_ids)}
        return sp.csr_matrix((edges.raw_weight.to_numpy(np.float32),
            ([pi[p] for p in edges.peak_id],[mi[m] for m in edges.motif_id])),shape=(len(pi),len(mi)))


@validated_operation
def select_motif_edges(annotation,config,output):
    from .annotation import AnnotationResult
    from .config import EdgeSelectionConfig
    if Path(output).exists():raise FileExistsError(output)
    if not isinstance(config,EdgeSelectionConfig):raise TypeError('selection requires EdgeSelectionConfig')
    result=AnnotationResult.load(annotation)
    if result.manifest['kind']!='peak_motif_annotation':raise ValueError('legacy graph is already selected; sites unavailable')
    if config.selection=='site_p' and config.p_cutoff>result.metadata['p_cutoff']:
        raise ValueError('selection cutoff exceeds scanned cutoff')
    pairs=result.relations()
    chosen=np.ones(len(pairs),bool) if config.selection=='all_candidates' else pairs.best_site_p.to_numpy()<=config.p_cutoff
    selected=pairs.loc[chosen].copy();selected['raw_weight']=np.float32(1)
    output=protect_output(output,[result.path])
    metadata=stage_metadata(output,upstreams={'annotation':result},peak_ids=result.peak_ids,motif_ids=result.motif_ids,
        peak_axis_hash=identity(result.peak_ids),motif_axis_hash=identity(result.motif_ids),
        selection=config.selection,p_cutoff=config.p_cutoff,nnz=int(chosen.sum()),
        state='complete' if chosen.any() else 'complete_empty_selection',
        site_evidence='sequence candidates; not high-confidence inference',
        shape=[len(result.peak_ids),len(result.motif_ids)],
        storage='annotation_relation_reference' if config.selection=='all_candidates' else 'selected_named_edges')
    with artifact_writer(output,'selected_motif_edges',metadata) as stage:
        if config.selection!='all_candidates':selected.to_csv(stage/'selected_edges.tsv.gz',sep='\t',index=False)
        pd.DataFrame({'relation_index':pairs.relation_index,'selected':chosen,
            'reason':np.where(chosen,'selected_by_declared_rule','above_site_p_cutoff')}).to_csv(
                stage/'selection_decisions.tsv.gz',sep='\t',index=False)
        write_json(stage/'selection.json',metadata)
    return SelectedMotifEdges.load(output)


def import_motif_edges(source,output,*,upstream=None):
    """Import an existing named matrix without inventing site evidence."""
    from .preprocessing import _read_dna
    from .data import canonical_csr,names
    if Path(output).exists():raise FileExistsError(output)
    source=Path(source).resolve();before=file_sha256(source)
    matrix,peaks,motifs=_read_dna(source)
    return _save_matrix_edges(matrix,peaks,motifs,protect_output(output,[source]),{'sha256':before},
                              validate=lambda:file_sha256(source)==before,
                              upstreams={} if upstream is None else {'legacy_annotation':upstream})


def _save_matrix_edges(matrix,peaks,motifs,output,source,*,validate=lambda:True,upstreams=None):
    from .data import canonical_csr,names
    matrix=canonical_csr(matrix);peaks=names(peaks,'peaks');motifs=names(motifs,'motifs')
    output=Path(output).resolve();coo=matrix.tocoo()
    metadata=stage_metadata(output,upstreams=upstreams,peak_ids=peaks,motif_ids=motifs,shape=list(matrix.shape),nnz=int(matrix.nnz),
        peak_axis_hash=identity(peaks),motif_axis_hash=identity(motifs),selection='imported_matrix',
        site_evidence='UNKNOWN; input is an already selected matrix',
        state='complete' if matrix.nnz else 'complete_empty_selection',source=source)
    with artifact_writer(output,'selected_motif_edges',metadata) as stage:
        pd.DataFrame({'relation_index':-1,'peak_index':coo.row,'motif_index':coo.col,
            'peak_id':[peaks[i] for i in coo.row],'motif_id':[motifs[j] for j in coo.col],
            'raw_weight':coo.data}).to_csv(stage/'selected_edges.tsv.gz',sep='\t',index=False)
        write_json(stage/'selection.json',metadata)
        if not validate():raise ValueError('imported motif source changed')
    return SelectedMotifEdges.load(output)
