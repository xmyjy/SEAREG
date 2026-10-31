"""One-way adapters for pre-artifact APIs. No filtering/scanning algorithms."""
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.sparse as sp
from .artifacts import FilteredCounts,stage_metadata,implementation_id
from .config import FilterConfig,PreprocessConfig
from .utils import artifact_writer,file_sha256,identity,write_json


def _read_dna(path):
    import anndata as ad
    path = Path(path)
    if path.suffix == ".h5ad":
        a = ad.read_h5ad(path)
        return a.X, tuple(a.obs_names), tuple(a.var_names)
    import h5py

    with h5py.File(path, "r") as h:
        if not all(k in h for k in ("mat", "row_names", "col_names")):
            raise ValueError(
                "DNA HDF5 must explicitly contain mat/row_names/col_names; select a training graph first"
            )
        rows = tuple(h["row_names"][:])
        cols = tuple(h["col_names"][:])
        node = h["mat"]
        if isinstance(node, h5py.Group):
            if node.attrs.get("encoding") != "csr" or not all(k in node for k in ("data", "indices", "indptr")):
                raise ValueError("unsupported DNA sparse matrix encoding")
            x = sp.csr_matrix((node["data"][:], node["indices"][:], node["indptr"][:]),
                              shape=tuple(node.attrs["shape"]))
            x.check_format(full_check=True)
        else:
            x = node[:]
    if x.shape != (len(rows), len(cols)):
        raise ValueError("DNA HDF5 mat must have peak rows and named DNA columns")
    return x, rows, cols


def resolve_filtered(atac,output,preprocessing=None):
    from .data import PreparedData,PreparedNetwork
    from .preprocessing import filter_counts
    if isinstance(atac,FilteredCounts):
        if preprocessing is not None:raise ValueError('filtered inputs fix their peak axis')
        return FilteredCounts.load(atac)
    from .artifacts import ModelNetwork
    if isinstance(atac,ModelNetwork) and atac.manifest['kind']=='model_network':
        if preprocessing is not None:raise ValueError('prepared inputs already fix the peak axis')
        return atac.upstream('filtered',FilteredCounts)
    if isinstance(atac,(str,Path)) and Path(atac).is_dir():
        import json
        kind=json.loads((Path(atac)/'manifest.json').read_text())['kind']
        if kind=='filtered_counts':
            if preprocessing is not None:raise ValueError('filtered inputs fix their peak axis')
            return FilteredCounts.load(atac)
        atac=PreparedNetwork.load(atac)
    if isinstance(atac,PreparedNetwork):
        atac=atac.load_data()
    base=Path(output).resolve().parent/'.seareg-inputs'
    if isinstance(atac,PreparedData):
        if preprocessing is not None:raise ValueError('prepared inputs already fix the peak axis')
        caller_identity=atac.fingerprint
        destination=base/identity({'prepared':caller_identity,'implementation':implementation_id('filtered')})
        if destination.exists():
            result=FilteredCounts.load(destination)
            if result.metadata.get('caller_data_fingerprint')!=caller_identity:
                raise ValueError('prepared input cache identity differs')
            return result
        import anndata as ad
        a=ad.AnnData(atac.X.copy())
        a.obs_names=list(atac.spot_ids);a.var_names=list(atac.peak_ids)
        if atac.coordinates is not None:a.obsm['spatial']=atac.coordinates
        report=atac.provenance.get('filter',{'protocol':'caller_frozen_axis',
            'original_shape':list(atac.X.shape),'filtered_shape':list(atac.X.shape)})
        metadata=stage_metadata(destination,spot_ids=atac.spot_ids,peak_ids=atac.peak_ids,
            spot_axis_hash=identity(atac.spot_ids),peak_axis_hash=identity(atac.peak_ids),
            slice_ids=atac.slice_ids,shape=list(atac.X.shape),nnz=int(atac.X.nnz),filter_report=report,
            filter_config={'filter_protocol':'caller_frozen_axis','slice_key':None},
            counts_semantics='binary_prepared_input; original raw counts unavailable',
            caller_data_fingerprint=caller_identity)
        with artifact_writer(destination,'filtered_counts',metadata) as stage:
            a.write_h5ad(stage/'counts.h5ad');write_json(stage/'filter_report.json',report)
            pd.DataFrame({'raw_index':np.arange(len(atac.peak_ids)),'peak_id':atac.peak_ids,
                'retained':True,'final_index':np.arange(len(atac.peak_ids)),
                'reason':'caller_frozen_axis'}).to_csv(stage/'peak_filter.tsv.gz',sep='\t',index=False)
            if atac.fingerprint!=caller_identity:raise ValueError('prepared input changed during import')
        return FilteredCounts.load(destination)
    pc=preprocessing if preprocessing is not None else PreprocessConfig()
    fc=FilterConfig(filter_protocol=pc.filter_protocol,slice_key=pc.slice_key)
    source=Path(atac).resolve();digest=file_sha256(source)
    destination=base/identity({'source':digest,'filter':asdict(fc),'implementation':implementation_id('filtered')})
    result=FilteredCounts.load(destination) if destination.exists() else filter_counts(source,fc,destination)
    if result.metadata.get('source',{}).get('sha256')!=digest:
        raise ValueError('filtered source identity differs from compatibility cache key')
    return result


def scan_legacy(atac,fasta,assembly,output,*,preprocessing=None,**kwargs):
    from .scanners.legacy import scan_peak_motifs_legacy
    filtered=resolve_filtered(atac,output,preprocessing)
    return scan_peak_motifs_legacy(filtered.load_data(),fasta,assembly,output,_filtered=filtered,**kwargs)


def annotate_legacy_api(atac,fasta,assembly,motifs,output,*,preprocessing=None,hits=None,**kwargs):
    from .sequence import prepare_sequences
    from .config import SequenceConfig,AnnotationConfig
    from .annotation import annotate_motifs,import_motif_sites
    if Path(output).exists():raise FileExistsError(output)
    filtered=resolve_filtered(atac,output,preprocessing)
    reference_identity=file_sha256(fasta)
    index_identity=file_sha256(str(fasta)+'.fai') if Path(str(fasta)+'.fai').is_file() else None
    key=identity({'peak_axis':filtered.metadata['peak_axis_hash'],'fasta':reference_identity,
                  'index':index_identity,'assembly':assembly,'implementation':implementation_id('sequences')})
    destination=Path(output).resolve().parent/'.seareg-sequences'/key
    from .artifacts import PeakSequences
    sequences=(PeakSequences.load(destination) if destination.exists() else
               prepare_sequences(filtered,SequenceConfig(str(fasta),assembly),destination))
    if (sequences.metadata['reference']['sha256']!=reference_identity or
            sequences.metadata['reference']['index_sha256']!=index_identity or
            sequences.metadata['peak_axis_hash']!=filtered.metadata['peak_axis_hash']):
        raise ValueError('sequence source identity differs from compatibility cache key')
    config=AnnotationConfig(str(motifs),**kwargs)
    return (annotate_motifs(sequences,config,output) if hits is None else
            import_motif_sites(sequences,config,hits,output))


def export_selected_h5(annotation,output,*,selection,p_cutoff=None):
    from .annotation import AnnotationResult
    from .edges import select_motif_edges,SelectedMotifEdges
    from .config import EdgeSelectionConfig
    from .artifacts import stage_metadata
    if Path(output).exists():raise FileExistsError(output)
    annotation=AnnotationResult.load(annotation)
    key=identity({'annotation':annotation.fingerprint,'selection':selection,'p_cutoff':p_cutoff,
                  'implementation':implementation_id('selected_edges')})
    target=Path(output).resolve().parent/'.seareg-edges'/key
    selected=(SelectedMotifEdges.load(target) if target.exists() else
              select_motif_edges(annotation,EdgeSelectionConfig(selection,p_cutoff),target))
    if not selected.metadata['nnz']:raise ValueError('selected training graph is empty; selection remains saved')
    metadata=stage_metadata(output,upstreams={'selected_edges':selected},shape=selected.metadata['shape'],
        selection=selection,p_cutoff=p_cutoff,confidence=selected.metadata['site_evidence'])
    with artifact_writer(output,'motif_training_graph',metadata) as stage:
        write_motif_h5(stage/'freq_motif.h5',selected.load_matrix(),selected.peak_ids,selected.motif_ids)
        write_json(stage/'selection.json',metadata)
    return Path(output).resolve()


def write_motif_h5(path,matrix,peaks,motifs):
    import h5py
    matrix=matrix.tocsr()
    with h5py.File(path,'w') as handle:
        handle.create_dataset('row_names',data=np.asarray(peaks,dtype=object),dtype=h5py.string_dtype())
        handle.create_dataset('col_names',data=np.asarray(motifs,dtype=object),dtype=h5py.string_dtype())
        group=handle.create_group('mat');group.attrs['encoding']='csr';group.attrs['shape']=matrix.shape
        for name,values in [('data',matrix.data),('indices',matrix.indices),('indptr',matrix.indptr)]:
            group.create_dataset(name,data=values,compression='gzip')


def load_prepared_v1(path):
    """Read an existing prepared artifact without filtering or building a graph."""
    import json
    import scipy.sparse as sp
    from .data import PreparedData
    from .utils import verify_artifact
    path=Path(path).resolve();manifest=verify_artifact(path,'prepared')
    meta=json.loads((path/'data.json').read_text())
    data=PreparedData(sp.load_npz(path/'atac_binary.npz'),tuple(meta['spot_ids']),tuple(meta['peak_ids']),
        np.load(path/'coordinates.npy',allow_pickle=False) if (path/'coordinates.npy').is_file() else None,
        tuple(meta['slice_ids']),sp.load_npz(path/'dna.npz') if (path/'dna.npz').is_file() else None,
        tuple(meta['dna_ids']),meta['dna_kind'],meta['provenance'])
    if data.fingerprint!=manifest['metadata']['data_fingerprint']:raise ValueError('prepared data fingerprint mismatch')
    return data


def resolve_training_inputs(value,config,output):
    """Normalize old prepared inputs into the same network/split artifact path."""
    import json
    from .artifacts import ModelNetwork,TrainingInputs
    from .data import PreparedData,PreparedNetwork
    from .sampling import prepare_training,sampling_config
    from .config import NetworkConfig
    from .graph import build_network
    from .edges import _save_matrix_edges,SelectedMotifEdges
    if isinstance(value,TrainingInputs):
        result=TrainingInputs.load(value)
        if result.metadata['sampling_config']!=sampling_config(config):raise ValueError('training inputs sampling configuration changed')
        return result
    if isinstance(value,(str,Path)):
        kind=json.loads((Path(value)/'manifest.json').read_text())['kind']
        if kind=='training_inputs':return resolve_training_inputs(TrainingInputs.load(value),config,output)
        value=ModelNetwork.load(value)
    network=normalize_network(value,config,output)
    target=Path(output).resolve().parent/'.seareg-training-inputs'/identity({'network':network.fingerprint,
        'sampling':sampling_config(config),'implementation':implementation_id('training_inputs')})
    return TrainingInputs.load(target) if target.exists() else prepare_training(network,config,target)

def normalize_network(value,config,output,*,network_output=None):
    from .artifacts import ModelNetwork
    from .data import PreparedData,PreparedNetwork
    from .config import NetworkConfig
    from .graph import build_network
    from .edges import _save_matrix_edges,SelectedMotifEdges
    if isinstance(value,ModelNetwork) and value.manifest['kind']=='model_network':network=ModelNetwork.load(value)
    else:
        if isinstance(value,(PreparedNetwork,ModelNetwork)):value=value.load_data()
        if not isinstance(value,PreparedData):raise TypeError('training requires TrainingInputs, ModelNetwork or legacy PreparedData')
        filtered=resolve_filtered(value,output)
        pc=PreprocessConfig(**value.provenance.get('config',{'use_spatial':False}))
        nc=NetworkConfig(**{name:getattr(pc,name) for name in NetworkConfig.__dataclass_fields__})
        selected=None
        if value.dna is not None:
            target=Path(output).resolve().parent/'.seareg-edges'/identity({'data':value.fingerprint,'mode':'prepared_dna'})
            selected=(SelectedMotifEdges.load(target) if target.exists() else
                      _save_matrix_edges(value.dna,value.peak_ids,value.dna_ids,target,
                        {'prepared_data_fingerprint':value.fingerprint}))
        target=Path(network_output) if network_output is not None else Path(output).resolve().parent/'.seareg-networks'/identity({'filtered':filtered.fingerprint,
            'selected':None if selected is None else selected.fingerprint,'network':asdict(nc),
            'implementation':implementation_id('network')})
        network=ModelNetwork.load(target) if target.exists() else build_network(
            filtered,selected,nc,target,_compiled_data=value)
    return network



def prepare_network_legacy(inputs,config,output):
    from .data import InputArtifact
    if isinstance(inputs,FilteredCounts):
        return network_from_paths(inputs,None,config,output)
    record=InputArtifact.load(inputs.path if isinstance(inputs,InputArtifact) else inputs)
    paths=record.paths
    return network_from_paths(paths['atac'],paths.get('dna'),config,output)


def network_from_paths(atac,dna,config,output):
    from .config import NetworkConfig
    from .artifacts import ModelNetwork
    from .edges import import_motif_edges,SelectedMotifEdges
    from .graph import build_network
    if Path(output).exists():raise FileExistsError(output)
    filtered=resolve_filtered(atac,output,None if isinstance(atac,FilteredCounts) else config)
    selected=None
    if dna is not None:
        if isinstance(dna,SelectedMotifEdges):selected=dna
        else:
            source_identity=file_sha256(dna)
            key=identity({'source':source_identity,'implementation':implementation_id('selected_edges')})
            target=Path(output).resolve().parent/'.seareg-edges'/key
            selected=SelectedMotifEdges.load(target) if target.exists() else import_motif_edges(dna,target)
            if selected.metadata.get('source',{}).get('sha256')!=source_identity:
                raise ValueError('motif source identity differs from compatibility cache key')
    nc=NetworkConfig(**{name:getattr(config,name) for name in NetworkConfig.__dataclass_fields__})
    return build_network(filtered,selected,nc,output)


def validate_run_config(run,config):
    """Validate explicit analysis inputs against the workflow without executing it."""
    import json
    import shutil
    from .artifacts import StageArtifact,TrainingInputs,resolve_reference
    from .annotation import AnnotationResult
    from .edges import SelectedMotifEdges
    c=config;manifest=run.manifest
    refs=manifest['metadata'].get('upstreams',{})
    if 'training_inputs' not in refs:
        provenance=json.loads((run.path/'data.json').read_text())['provenance']
        if provenance.get('config')!=asdict(c['preprocessing']):raise ValueError('training preprocessing configuration differs')
        for name,key in (('atac','source'),('dna','dna_source')):
            expected=c['input'].get(name);recorded=provenance.get(key)
            if (expected is None)!=(recorded is None) or (expected is not None and file_sha256(expected)!=recorded['sha256']):
                raise ValueError('training input identity differs from workflow')
        return
    entry=refs['training_inputs']
    inputs=TrainingInputs.load(resolve_reference(run.path,manifest,entry));network=inputs.network()
    if inputs.fingerprint!=entry['fingerprint']:raise ValueError('training input identity changed')
    if c['network_import'] is not None:
        from .artifacts import ModelNetwork
        imported=ModelNetwork.load(c['network_import'])
        if imported.manifest['kind']=='model_network' and imported.fingerprint!=network.fingerprint:
            raise ValueError('training network identity differs')
        if c['network_options_explicit'] and network.metadata['network_config']!=asdict(c['network_config']):
            raise ValueError('training network configuration differs')
        return
    if network.metadata['network_config']!=asdict(c['network_config']):raise ValueError('training network configuration differs')
    filtered=network.upstream('filtered',FilteredCounts)
    if c['filter_options_explicit'] and filtered.metadata['filter_config']!=asdict(c['filtering']):
        raise ValueError('training filtering configuration differs')
    if 'filtered' in c['input'] and FilteredCounts.load(c['input']['filtered']).fingerprint!=filtered.fingerprint:
        raise ValueError('training filtered input identity differs')
    if 'atac' in c['input'] and file_sha256(c['input']['atac'])!=filtered.metadata.get('source',{}).get('sha256'):
        raise ValueError('training original input identity differs')
    expected_motif=c['annotation'] is not None or 'dna' in c['input']
    if expected_motif!=network.metadata['has_motif']:raise ValueError('training motif configuration differs')
    if not expected_motif:return
    selected=network.upstream('selected_edges',SelectedMotifEdges)
    if c['annotation'] is None:
        if selected.metadata['source'].get('sha256')!=file_sha256(c['input']['dna']):
            raise ValueError('training motif input identity differs')
        return
    if c['annotation']['method']=='legacy':
        legacy=selected.upstream('legacy_annotation',AnnotationResult)
        parameters=legacy.metadata.get('scanner',{}).get('parameters',{})
        ac=c['annotation_config']
        if parameters.get('p_cutoff')!=ac.p_cutoff or tuple(parameters.get('species_names',()))!=ac.species_names:
            raise ValueError('training legacy annotation configuration differs')
        reference=Path(c['sequence'].fasta);index=Path(str(reference)+'.fai')
        if (legacy.metadata.get('assembly')!=c['sequence'].assembly or
                legacy.metadata.get('fasta',{}).get('sha256')!=file_sha256(reference) or
                'fasta_index_sha256' not in legacy.metadata or
                legacy.metadata['fasta_index_sha256']!=(file_sha256(index) if index.is_file() else None)):
            raise ValueError('training legacy sequence resource/configuration identity differs')
        executable=shutil.which(ac.rscript)
        scanner=legacy.metadata.get('scanner_runtime',{})
        if not scanner.get('sha256'):
            raise ValueError('training legacy annotation scanner identity is UNKNOWN')
        if executable is None or scanner['sha256']!=file_sha256(executable):
            raise ValueError('training legacy annotation scanner identity differs')
        return
    rule=c['edge_selection']
    if rule is None or selected.metadata['selection']!=rule.selection or selected.metadata['p_cutoff']!=rule.p_cutoff:
        raise ValueError('training edge selection identity differs')
    annotation=selected.upstream('annotation',AnnotationResult);ac=c['annotation_config']
    if annotation.metadata['motifs']['sha256']!=file_sha256(ac.motifs) or annotation.metadata['p_cutoff']!=ac.p_cutoff or annotation.metadata['motif_pseudocount']!=ac.motif_pseudocount:
        raise ValueError('training annotation resource/configuration identity differs')
    sequences=annotation.upstream('sequences')
    if sequences.metadata['assembly']!=c['sequence'].assembly or sequences.metadata['reference']['sha256']!=file_sha256(c['sequence'].fasta):
        raise ValueError('training sequence resource/configuration identity differs')
    executable=shutil.which(ac.fimo)
    scanner=annotation.metadata.get('scanner',{})
    if not scanner.get('sha256'):
        raise ValueError('training annotation scanner identity is UNKNOWN for imported sites')
    if executable is None or scanner['sha256']!=file_sha256(executable):
        raise ValueError('training annotation scanner identity differs')
