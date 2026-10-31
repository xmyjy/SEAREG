"""One explicit artifact DAG for CLI and Python; check/plan never execute it."""
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
from .config import load_config,effective_config
from .utils import file_sha256,identity,write_json,protect_output


def source_identity():
    """Full audit identity, deliberately not every stage's invalidation key."""
    root=Path(__file__).parent
    return identity({str(p.relative_to(root)):file_sha256(p)
                     for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.py','.R','.json')})


class Workflow:
    def __init__(self,config):self.config=load_config(config)

    @classmethod
    def from_counts(cls,atac,*,motif=True,motif_source='auto',output=None,genome=None,fasta=None,dna=None,
                    auto_build_dna=True,rscript='Rscript',download_reference=True,
                    training=None,network=None,execution=None,analysis=None):
        """One counts input; motif=False skips all motif resources and scanning."""
        if type(motif) is not bool:raise ValueError('motif must be True or False')
        source=Path(atac).resolve()
        if not source.is_file():raise FileNotFoundError(source)
        options={'enabled':motif,'source':motif_source,'auto_build_dna':auto_build_dna,'rscript':str(rscript),
                 'download_reference':download_reference}
        if genome is not None:options['genome']=genome
        if fasta is not None:options['fasta']=str(Path(fasta).resolve())
        inputs={'atac':str(source)}
        if dna is not None:inputs['dna']=str(Path(dna).resolve())
        from dataclasses import is_dataclass
        def plain(value):return asdict(value) if is_dataclass(value) else dict(value or {})
        raw={'schema_version':2,'dataset_id':source.stem,'input':inputs,'motif':options,
            'output':str(Path(output).resolve() if output is not None else source.parent/'seareg_results'/source.stem),
            'training':plain(training),'network':plain(network),'execution':plain(execution)}
        if analysis is not None:raw['analysis']=plain(analysis)
        return cls(raw)

    def _dependencies(self,stage):
        c=self.config;annotation=c['annotation']
        if stage=='filtered':return ()
        if stage=='sequences':return ('filtered',)
        if stage=='annotation':
            if annotation is None:raise ValueError('workflow has no annotation configuration')
            return ('sequences',) if annotation['method']=='fimo' else ('filtered',)
        if stage=='selected_edges':
            if annotation is not None:return ('annotation',)
            return ('filtered',) if c['motif']['mode']=='needs_genome' else ()
        if stage=='network':
            if c['network_import'] is not None:return ()
            return ('filtered','selected_edges') if annotation is not None or 'dna' in c['input'] or c['motif']['enabled'] else ('filtered',)
        return {'training_inputs':('network',),'training':('training_inputs',),
                'analysis':('training',),'render':('analysis',)}[stage]

    def _order(self,target):
        result=[]
        def visit(stage):
            if stage in result:return
            for dependency in self._dependencies(stage):visit(dependency)
            result.append(stage)
        visit(target);return result

    def _target(self):
        if self.config['analysis'] is None:return 'training'
        return 'render' if self.config['plots'] else 'analysis'

    def _fingerprint(self,value):
        from .artifacts import validation_scope
        with validation_scope() as context:
            actual=context.verify(value.path)['fingerprint']
            bound=value.manifest['fingerprint'] if hasattr(value,'manifest') else getattr(value,'artifact_fingerprint',None)
            if bound is not None and actual!=bound:raise ValueError('supplied artifact identity replaced')
            return actual

    def _import(self,stage):
        from .artifacts import FilteredCounts,ModelNetwork
        c=self.config
        if stage=='filtered' and 'filtered' in c['input']:
            result=FilteredCounts.load(c['input']['filtered'])
            if c['filter_options_explicit'] and asdict(c['filtering'])!=result.metadata['filter_config']:
                raise ValueError('imported filtered input belongs to different filtering configuration')
            return result
        if stage=='network' and c['network_import'] is not None:
            result=ModelNetwork.load(c['network_import'])
            if not c['motif']['enabled']:
                has_motif=result.metadata.get('has_motif')
                if has_motif is None:has_motif=result.load_data().dna is not None
                if has_motif:raise ValueError('motif=False conflicts with a frozen motif-enabled network; rebuild it from counts without motifs')
            if c['source_config_version']==1:
                provenance=json.loads((result.path/'data.json').read_text())['provenance']
                if provenance.get('config')!=asdict(c['preprocessing']):
                    raise ValueError('imported network belongs to different preprocessing configuration')
                for name,record in (('atac','source'),('dna','dna_source')):
                    expected=c['input'].get(name);saved=provenance.get(record)
                    if (expected is None)!=(saved is None) or (expected is not None and file_sha256(expected)!=saved['sha256']):
                        raise ValueError('imported prepared source identity differs')
            if result.manifest['kind']=='model_network':
                if c['network_options_explicit'] and result.metadata['network_config']!=asdict(c['network_config']):
                    raise ValueError('imported network belongs to different network configuration')
                return result
        return None

    def _description(self,stage,values,overrides):
        """Describe dependencies/options, resolving new-data defaults when needed."""
        from .artifacts import implementation_id
        c=self.config
        dependencies={name:self._fingerprint(value) for name,value in values.items()}
        options={}
        if stage=='filtered':
            if 'atac' not in c['input']:raise ValueError('a network import already fixes its input; use prepare-network')
            dependencies={'atac':file_sha256(c['input']['atac'])};options=asdict(c['filtering'])
        elif stage=='sequences':
            if c['sequence'] is None:raise ValueError('sequence reference/assembly is not configured')
            if c['annotation'] is not None and c['annotation']['method']=='legacy':
                raise ValueError('legacy scanner extracts its own sequences; no duplicate sequence stage')
            fasta=Path(c['sequence'].fasta);fai=Path(str(fasta)+'.fai')
            dependencies={'peak_axis':values['filtered'].metadata['peak_axis_hash'],
                          'reference':file_sha256(fasta),'index':file_sha256(fai) if fai.is_file() else None}
            options={'assembly':c['sequence'].assembly,'background':'pooled_complement_symmetric_v1'}
        elif stage=='annotation':
            ac=c['annotation_config'];options=asdict(ac)
            binary=ac.fimo if c['annotation']['method']=='fimo' else ac.rscript
            executable=shutil.which(binary)
            if executable is None:raise FileNotFoundError(f'annotation executable unavailable: {binary}')
            dependencies['scanner']=file_sha256(executable)
            if c['annotation']['method']=='fimo':
                dependencies['motifs']=file_sha256(ac.motifs)
                options={name:value for name,value in options.items() if name not in ('fimo','motifs')}
            else:
                index=Path(str(c['sequence'].fasta)+'.fai')
                dependencies.update(reference=file_sha256(c['sequence'].fasta),
                                    index=file_sha256(index) if index.is_file() else None,
                                    peak_axis=values['filtered'].metadata['peak_axis_hash'])
                options['assembly']=c['sequence'].assembly
        elif stage=='selected_edges':
            if c['motif']['mode']=='needs_genome':raise ValueError(c['motif']['reason'])
            if c['annotation'] is None:
                if 'dna' not in c['input']:raise ValueError('no annotation or imported motif matrix')
                dependencies={'matrix':file_sha256(c['input']['dna'])};options={'selection':'imported_matrix'}
            elif c['annotation']['method']=='legacy':options={'selection':'legacy_already_selected'}
            else:
                if c['edge_selection'] is None:raise ValueError('FIMO annotation requires explicit graph_selection/edge_selection before training')
                options=asdict(c['edge_selection'])
        elif stage=='network':
            options=asdict(c['network_config'])
            if c['network_import'] is not None:
                imported=self._load('network',c['network_import'])
                dependencies={'legacy_prepared':self._fingerprint(imported),
                              'legacy_data':imported.metadata['data_fingerprint']}
        elif stage=='training_inputs':
            from .config import sampling_config
            automatic=c.get('training_auto_fields',())
            if automatic:
                from .defaults import training_config
                explicit={name:value for name,value in asdict(c['training']).items()
                          if name not in automatic}
                c['training']=training_config(values['network'],**explicit)
                c['training_defaults_resolved']=True
            options=sampling_config(c['training'])
        elif stage=='training':
            options={'training':asdict(c['training']),'execution':c['execution']}
            if overrides.get('resume_from') is not None:
                dependencies['checkpoint']=file_sha256(overrides['resume_from'])
            elif overrides.get('checkpoint_identity') is not None:
                dependencies['checkpoint']=overrides['checkpoint_identity']
        elif stage=='analysis':
            if c['analysis'] is None:raise ValueError('explicit analysis configuration required')
            options={'analysis':asdict(c['analysis']),'execution':c['execution']}
        elif stage=='render':options={'saved_results_only':True}
        implementation=implementation_id('legacy' if stage=='annotation' and c['annotation']['method']=='legacy' else stage)
        key=identity({'stage':stage,'dependencies':dependencies,'options':options,'implementation':implementation})
        return key,{'dependencies':dependencies,'options':options,'implementation':implementation}

    def _validate_stage_identity(self,stage,value,description):
        """Bind consumed content to its key, including an A->B->A source swap.

        Rehashing the current source alone cannot establish what a completed
        operation actually consumed. Check its sealed resource/upstream hashes
        both on creation and reuse instead of trusting the receipt alone.
        """
        from .artifacts import validation_scope
        with validation_scope() as context:
            metadata=context.verify(value.path)['metadata']
        expected=description['dependencies'];actual={}
        refs=metadata.get('upstreams',{})
        for name in expected:
            ref='legacy_annotation' if stage=='selected_edges' and name=='annotation' and 'legacy_annotation' in refs else name
            if ref in refs:actual[name]=refs[ref]['fingerprint']
        if stage=='filtered':actual['atac']=metadata.get('source',{}).get('sha256')
        elif stage=='sequences':
            actual.update(peak_axis=metadata.get('peak_axis_hash'),
                reference=metadata.get('reference',{}).get('sha256'),
                index=metadata.get('reference',{}).get('index_sha256'))
        elif stage=='annotation':
            if self.config['annotation']['method']=='fimo':
                actual.update(scanner=metadata.get('scanner',{}).get('sha256'),
                              motifs=metadata.get('motifs',{}).get('sha256'))
            else:
                actual.update(scanner=metadata.get('scanner_runtime',{}).get('sha256'),
                    reference=metadata.get('fasta',{}).get('sha256'),
                    index=metadata.get('fasta_index_sha256'),
                    peak_axis=metadata.get('peak_axis_hash'))
        elif stage=='selected_edges' and 'matrix' in expected:
            actual['matrix']=metadata.get('source',{}).get('sha256')
        elif stage=='network' and 'legacy_prepared' in expected:
            actual['legacy_data']=metadata.get('caller_data_fingerprint')
            # The imported handle is revalidated before normalization; its
            # frozen numerical fingerprint is also checked independently.
            actual['legacy_prepared']=self._fingerprint(self._load('network',self.config['network_import']))
        elif stage=='training' and 'checkpoint' in expected:
            actual['checkpoint']=(metadata.get('parent') or {}).get('sha256')
        elif stage=='analysis':actual['training']=metadata.get('provenance',{}).get('run_fingerprint')
        missing=set(expected)-set(actual)
        if missing or any(actual.get(name)!=digest for name,digest in expected.items()):
            raise ValueError(f'stage {stage} consumed source/upstream identity differs from its stage key')

    def _destination(self,stage,key):
        root=self.config['output']
        if stage=='training':return root/'runs'/key
        if stage=='analysis':return root/'analyses'/key
        if stage=='render':return root/'figures'/key
        return root/'artifacts'/stage/key

    def _receipt(self,stage,key):return self.config['output']/'stages'/f'{stage}-{key}.json'

    def _load(self,stage,path):
        from .artifacts import FilteredCounts,PeakSequences,ModelNetwork,TrainingInputs,StageArtifact
        from .annotation import AnnotationResult
        from .edges import SelectedMotifEdges
        from .data import TrainingRun
        loaders={'filtered':FilteredCounts.load,'sequences':PeakSequences.load,'annotation':AnnotationResult.load,
            'selected_edges':SelectedMotifEdges.load,'network':ModelNetwork.load,'training_inputs':TrainingInputs.load,
            'training':TrainingRun.load,'render':StageArtifact.load}
        if stage=='analysis':
            from .analysis import load_result
            return load_result(path)
        return loaders[stage](path)

    @contextmanager
    def _lock(self,stage,key):
        c=self.config
        root=protect_output(c['output'],[p for p in (*c['input'].values(),c['network_import']) if p is not None])
        folder=root/'locks';folder.mkdir(parents=True,exist_ok=True)
        with (folder/f'{stage}-{key}.lock').open('a') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error:raise RuntimeError(f'workflow stage already active: {stage}') from error
            try:yield
            finally:fcntl.flock(handle,fcntl.LOCK_UN)

    def _operate(self,stage,values,destination,overrides):
        c=self.config
        if stage=='filtered':
            from .preprocessing import filter_counts
            return filter_counts(c['input']['atac'],c['filtering'],destination)
        if stage=='sequences':
            from .sequence import prepare_sequences
            return prepare_sequences(values['filtered'],c['sequence'],destination)
        if stage=='annotation':
            if c['annotation']['method']=='fimo':
                from .annotation import annotate_motifs
                return annotate_motifs(values['sequences'],c['annotation_config'],destination,attempts_dir=c['output']/'attempts')
            from .compat import scan_legacy
            from .annotation import AnnotationResult
            path=scan_legacy(values['filtered'],c['sequence'].fasta,c['sequence'].assembly,destination,
                             **asdict(c['annotation_config']))
            return AnnotationResult.load(path)
        if stage=='selected_edges':
            from .edges import select_motif_edges,import_motif_edges
            if c['annotation'] is None:return import_motif_edges(c['input']['dna'],destination)
            if c['annotation']['method']=='legacy':
                return import_motif_edges(values['annotation'].path/'freq_motif.h5',destination,
                                          upstream=values['annotation'])
            return select_motif_edges(values['annotation'],c['edge_selection'],destination)
        if stage=='network':
            if c['network_import'] is not None:
                from .compat import normalize_network
                return normalize_network(self._load('network',c['network_import']),c['training'],destination,
                                         network_output=destination)
            from .graph import build_network
            return build_network(values['filtered'],values.get('selected_edges'),c['network_config'],destination)
        if stage=='training_inputs':
            from .sampling import prepare_training
            return prepare_training(values['network'],c['training'],destination)
        if stage=='training':
            from .train import train_SEAREG
            return train_SEAREG(values['training_inputs'],c['training'],destination,execution=c['execution'],
                                resume_from=overrides.get('resume_from'))
        if stage=='analysis':
            from ._runtime import run_worker
            run_worker('analyze',{'training':str(values['training'].path),'config':asdict(c['analysis']),
                                  'plots':False},destination,c['execution'])
            return self._load('analysis',destination)
        if stage=='render':
            from .plotting import render_analysis
            return render_analysis(values['analysis'],destination)
        raise ValueError(f'unknown workflow stage: {stage}')

    def _selected_training(self, overrides):
        selected=None
        workflow=self.config['output']/'workflow.json'
        pointer=(json.loads(workflow.read_text()).get('stages',{}).get('training')
                 if workflow.is_file() else None)
        if pointer is not None:
            selected=self._load('training',self.config['output']/pointer['path'])
            if self._fingerprint(selected)!=pointer.get('artifact_fingerprint'):
                raise ValueError('selected training artifact identity changed')
            parent=selected.metadata.get('parent') or {}
            overrides={**overrides,'checkpoint_identity':parent.get('sha256')}
        return overrides,selected

    def _execute(self,target,*,read_dependencies=False,supplied=None,**overrides):
        from .artifacts import validation_scope
        protect_output(self.config['output'],[p for p in (*self.config['input'].values(),self.config['network_import']) if p is not None])
        known=dict(supplied or {});pointers={}
        with validation_scope():
            def ensure(stage):
                if stage in known:return known[stage]
                values={name:ensure(name) for name in self._dependencies(stage)}
                imported=self._import(stage)
                if imported is not None:
                    known[stage]=imported;pointers[stage]={'path':os.path.relpath(imported.path,self.config['output']),'artifact_fingerprint':self._fingerprint(imported)}
                    return imported
                if stage=='annotation' and self.config['reference_download'] is not None:
                    from .resources import ensure_reference
                    if read_dependencies and stage!=target and not Path(self.config['sequence'].fasta).is_file():
                        raise ValueError('annotation reference/stage is absent; analyze does not prepare resources')
                    ac=self.config['annotation_config']
                    binary=ac.rscript if self.config['annotation']['method']=='legacy' else ac.fimo
                    if shutil.which(binary) is None:raise FileNotFoundError(f'annotation executable unavailable: {binary}')
                    if self.config['annotation']['method']=='legacy' and shutil.which('bedtools') is None:
                        raise FileNotFoundError('bedtools unavailable; install environment-motif.yml before scanning')
                    ensure_reference(self.config['reference_download'])
                stage_overrides,selected=(self._selected_training(overrides)
                    if stage=='training' and read_dependencies else (overrides,None))
                key,description=self._description(stage,values,stage_overrides)
                destination=self._destination(stage,key);receipt=self._receipt(stage,key)
                if selected is not None and selected.path!=destination:
                    raise ValueError('selected training stage belongs to different inputs/configuration/code identity')
                can_create=not read_dependencies or stage==target
                def read():
                    saved=json.loads(receipt.read_text())
                    if saved.get('state')!='complete' or saved.get('stage_key')!=key:
                        raise ValueError(f'stage {stage} belongs to different or incomplete identity')
                    value=self._load(stage,destination)
                    if self._fingerprint(value)!=saved.get('artifact_fingerprint'):
                        raise ValueError(f'stage {stage} artifact identity changed')
                    self._validate_stage_identity(stage,value,description)
                    return value
                if receipt.is_file() and json.loads(receipt.read_text()).get('state')=='complete':value=read()
                elif not can_create:
                    raise ValueError(f'stage {stage} is absent or belongs to different inputs/config/code identity')
                else:
                    with self._lock(stage,key):
                        if receipt.is_file() and json.loads(receipt.read_text()).get('state')=='complete':value=read()
                        else:
                            if destination.exists():raise FileExistsError(f'unregistered stage output: {destination}')
                            receipt.parent.mkdir(parents=True,exist_ok=True)
                            old=json.loads(receipt.read_text()) if receipt.is_file() else {}
                            attempts=list(old.get('attempts',[]))
                            if old.get('state')=='failed':attempts.append({'error':old.get('error')})
                            record={'stage':stage,'stage_key':key,**description,'attempts':attempts}
                            write_json(receipt,{'state':'running',**record})
                            try:
                                value=self._operate(stage,values,destination,overrides)
                                self._validate_stage_identity(stage,value,description)
                                write_json(receipt,{'state':'complete',**record,
                                    'artifact_fingerprint':self._fingerprint(value),
                                    'path':str(destination.relative_to(self.config['output']))})
                            except BaseException as error:
                                write_json(receipt,{'state':'failed',**record,'error':f'{type(error).__name__}: {error}'})
                                raise
                known[stage]=value;pointers[stage]={'path':os.path.relpath(value.path,self.config['output']),'artifact_fingerprint':self._fingerprint(value)}
                return value
            result=ensure(target)
            self._record_workflow(pointers)
            return result

    def _record_workflow(self,pointers):
        root=self.config['output'];root.mkdir(parents=True,exist_ok=True)
        resolved=effective_config(self.config);key=identity(resolved)
        folder=root/'configs';folder.mkdir(exist_ok=True)
        target=folder/(key+'.json')
        if not target.exists():write_json(target,resolved)
        previous=json.loads((root/'workflow.json').read_text()) if (root/'workflow.json').is_file() else {}
        write_json(root/'workflow.json',{'schema_version':2,'dataset_id':self.config['dataset_id'],
            'config':str(target.relative_to(root)),'stages':{**previous.get('stages',{}),**pointers}})

    def plan(self,target='run'):
        """Inspect reuse and pending upstreams without scanning or making output dirs."""
        from .artifacts import validation_scope
        selected_dependencies=target in ('analyze','analysis','render')
        skip_motif=target=='prepare-motif' and not self.config['motif']['enabled']
        target=self._target() if target=='run' else target
        aliases={'filter':'filtered','prepare-inputs':'filtered','prepare-motif':'selected_edges',
                 'prepare-sequences':'sequences','annotate':'annotation','select-edges':'selected_edges',
                 'prepare-network':'network','prepare-training':'training_inputs','train':'training','analyze':'analysis'}
        target=aliases.get(target,target);known={};rows=[]
        with validation_scope():
            for stage in self._order(target):
                if stage=='selected_edges' and skip_motif:
                    rows.append({'stage':stage,'state':'skipped','reason':'motif=False'});continue
                deps=self._dependencies(stage)
                if any(name not in known for name in deps):
                    rows.append({'stage':stage,'state':'pending_upstream','dependencies':list(deps)});continue
                try:
                    imported=self._import(stage)
                    if imported is not None:
                        known[stage]=imported;rows.append({'stage':stage,'state':'import','path':str(imported.path)});continue
                    options,selected=(self._selected_training({}) if stage=='training' and selected_dependencies else ({},None))
                    key,description=self._description(stage,{name:known[name] for name in deps},options)
                    destination=self._destination(stage,key);receipt=self._receipt(stage,key)
                    if selected is not None and selected.path!=destination:
                        raise ValueError('selected training stage belongs to different inputs/configuration/code identity')
                    if receipt.is_file() and json.loads(receipt.read_text()).get('state')=='complete':
                        value=self._load(stage,destination)
                        if self._fingerprint(value)!=json.loads(receipt.read_text())['artifact_fingerprint']:
                            raise ValueError('stage artifact identity changed')
                        self._validate_stage_identity(stage,value,description)
                        known[stage]=value;state='reuse'
                    else:state='create' if not destination.exists() else 'unregistered_output'
                    rows.append({'stage':stage,'state':state,'stage_key':key,'path':str(destination)})
                except (OSError,ValueError) as error:rows.append({'stage':stage,'state':'blocked','reason':str(error)})
        return {'state':'planned','target':target,'stages':rows,'execution_performed':False,
                'motif':dict(self.config['motif'])}

    def check(self):
        from .preprocessing import inspect_inputs
        c=self.config
        protect_output(c['output'],[p for p in (*c['input'].values(),c['network_import']) if p is not None])
        if 'atac' in c['input']:
            headers=inspect_inputs(c['input']['atac'],c['input'].get('dna'),use_spatial=c['network_config'].use_spatial)
        elif 'filtered' in c['input']:
            filtered=self._import('filtered');headers={'filtered':{'path':str(filtered.path),'shape':filtered.metadata['shape']}}
        else:
            network=self._load('network',c['network_import']);headers={'network':{'path':str(network.path),'shape':network.metadata.get('shape')}}
        libraries=['numpy','scipy','pandas','torch','h5py','anndata']
        if c['analysis'] is not None:libraries+=['scanpy','umap','igraph','louvain','matplotlib']
        dependencies={name:importlib.util.find_spec(name) is not None for name in libraries}
        if not all(dependencies.values()):raise ValueError(f'missing dependencies: {[name for name,ok in dependencies.items() if not ok]}')
        annotation=c['annotation']
        report={'state':'checked','dataset_id':c['dataset_id'],'inputs':headers,'dependencies':dependencies,
            'output':str(c['output']),'output_exists':c['output'].exists(),'training_executed':False,'analysis_executed':False,
            'execution':c['execution'],'training':effective_config(c)['training'],
            'source_identity':source_identity(),'config_migration':c['migration_report'],
            'annotation':None if annotation is None else {'method':annotation['method'],'executed':False,
                'assembly':c['sequence'].assembly,'reference_exists':Path(c['sequence'].fasta).is_file(),
                'training_graph_selection_required':annotation['method']=='fimo' and c['edge_selection'] is None}}
        report['motif']=dict(c['motif'])
        report['ready_to_train']=c['motif']['mode']!='needs_genome'
        if annotation is not None:
            ac=c['annotation_config'];method=annotation['method']
            tools={'scanner':shutil.which(ac.rscript if method=='legacy' else ac.fimo) is not None}
            if method=='legacy':tools['bedtools']=shutil.which('bedtools') is not None
            report['annotation']['tools_available']=tools
            report['ready_to_train'] &= all(tools.values())
            reference_ready=report['annotation']['reference_exists'] or bool(
                c['reference_download'] is not None and c['reference_download']['download'])
            report['ready_to_train'] &= reference_ready and not report['annotation']['training_graph_selection_required']
        report['reference_download_required']=c['reference_download'] is not None and not Path(c['sequence'].fasta).is_file()
        return report

    def filter(self):return self._execute('filtered')
    def prepare_inputs(self):return self.filter()
    def prepare_sequences(self):return self._execute('sequences')
    def annotate(self):return self._execute('annotation')
    def prepare_motif(self):
        """Step 2: skip, import an existing motif matrix, or scan then import."""
        if not self.config['motif']['enabled']:return None
        return self.select_edges()
    def select_edges(self):return self._execute('selected_edges')
    def prepare_network(self):return self._execute('network')
    def prepare_training(self):return self._execute('training_inputs')
    def train(self,*,resume_from=None):return self._execute('training',resume_from=resume_from)

    def analyze(self,training=None):
        if training is None:return self._execute('analysis',read_dependencies=True)
        from .data import TrainingRun
        training=TrainingRun.load(training)
        from .compat import validate_run_config
        validate_run_config(training,self.config)
        return self._execute('analysis',supplied={'training':training})

    def render(self,analysis=None):
        return self._execute('render',read_dependencies=True,supplied=None if analysis is None else {'analysis':analysis})

    def run(self,*,resume_from=None):
        target=self._target()
        result=self._execute(target,resume_from=resume_from)
        if target=='render':
            from .analysis import load_result
            upstream=result.upstream('analysis')
            return load_result(upstream.path)
        return result


def run_workflow(atac,*,output=None,motif=True,motif_source='auto',genome=None,
                 fasta=None,dna=None,rscript='Rscript',auto_build_dna=True,
                 download_reference=True,epochs=None,hidden_dim=None,lr=None,
                 seed=None,device=None,shards=None,pn_r=None,tr=None,
                 spatial_weight=None,dna_weight=None,physical_gpu=None,
                 training=None,network=None,execution=None,analysis=None,
                 resume_from=None):
    """Run counts -> optional motifs -> sparse network -> inputs -> training.

    Common hyperparameters override the corresponding ``training`` options.
    Omitted epochs and shards are selected from the compiled network's spot
    and positive-edge counts. Other options keep the TrainConfig defaults;
    dictionaries and dataclasses are accepted without mutating the caller.
    This is a convenience adapter for Workflow.from_counts(...).run(), not a
    second preparation or training implementation. The returned TrainingRun
    exposes its output through ``.path``. Supplying ``analysis`` explicitly
    also runs the existing analysis/render stages and returns AnalysisResult.

    CUDA requires both device='cuda:0' and physical_gpu=<physical card index>.
    Explicit training options override the data-dependent defaults.
    """
    from dataclasses import is_dataclass
    from collections.abc import Mapping
    def options(value,label):
        if value is None:return {}
        if is_dataclass(value) and not isinstance(value,type):return asdict(value)
        if isinstance(value,Mapping):return dict(value)
        raise ValueError(f'{label} must be a configuration object or mapping')
    training_options=options(training,'training')
    overrides={'epochs':epochs,'hidden_dim':hidden_dim,'lr':lr,'seed':seed,
        'device':device,'shards':shards,'pn_r':pn_r,'tr':tr,
        'spatial_weight':spatial_weight,'dna_weight':dna_weight}
    training_options.update({name:value for name,value in overrides.items() if value is not None})
    execution_options=options(execution,'execution')
    if physical_gpu is not None:execution_options['physical_gpu']=physical_gpu
    workflow=Workflow.from_counts(atac,output=output,motif=motif,motif_source=motif_source,
        genome=genome,fasta=fasta,dna=dna,rscript=rscript,auto_build_dna=auto_build_dna,
        download_reference=download_reference,training=training_options,network=network,
        execution=execution_options,analysis=analysis)
    return workflow.run(resume_from=resume_from)
