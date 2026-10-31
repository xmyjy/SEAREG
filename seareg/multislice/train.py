"""Balanced shared-parameter training with native run/export contracts."""
from dataclasses import asdict
from pathlib import Path
import os,shutil,time
import numpy as np
import pandas as pd
from ..data import TrainingRun
from ..utils import protect_output,write_json,seal_artifact,environment
from ..artifacts import stage_metadata
from .inputs import MultiSliceInputs


def train_multislice(inputs,config,output,*,execution=None,resume_from=None):
    """Execute in a fresh subprocess after physical GPU visibility isolation."""
    from .._runtime import run_worker
    inputs=MultiSliceInputs.load(inputs)
    from .jobs import validated_execution
    runtime=validated_execution(config,execution)
    run_worker('train',{'trainer':'multislice','prepared':str(inputs.path),'config':asdict(config),
                       'resume_from':None if resume_from is None else str(resume_from)},output,runtime)
    return TrainingRun.load(output)


def _train(inputs,config,output,resume_from=None,*,execution=None):
    import torch
    from .sampling import MultiSliceSampler
    from .model import MicroGainTailgate,infer_tailgate_cpu,geometry_summary
    from .balanced import train_epoch
    from .checkpoint import save_checkpoint,read_checkpoint
    from ..checkpoint import seed_all,restore_rng
    from ..train import _execution_provenance
    from ..exports import export_embeddings
    inputs=MultiSliceInputs.load(inputs);output=protect_output(output,[inputs.path])
    if output.exists():raise FileExistsError(output)
    device=torch.device(config.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA device unavailable')
    execution_record=_execution_provenance(torch,device,execution)
    sampler=MultiSliceSampler(inputs,config);data=sampler.data
    checkpoint=read_checkpoint(resume_from,inputs,config,execution_record) if resume_from is not None else None
    # Data preparation has its own deterministic streams. Initialization resets
    # the registered model seed exactly as in the selected source.
    seed_all(config.seed)
    model=MicroGainTailgate(sampler.num_spot,sampler.num_peak,config.hidden_dim,
                            sampler.train_spot_peak_degree,data.coordinates,sampler.spot_slice_codes,diagnostic_root=output/'gate_states').to(device)
    model_protocol='two_layer_sum_cs_matched03_v1'
    spatial=sampler.data_spatial_graph.edge_index.to(device);codes=sampler.spot_slice_codes.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.lr,weight_decay=0.)
    output.mkdir(parents=True);(output/'checkpoints').mkdir();(output/'embeddings').mkdir()
    write_json(output/'status.json',{'state':'initializing','pid':os.getpid()})
    completed=0;steps=0;history=[];lineage=None
    try:
        if checkpoint is None:
            model.initialize_adaptive_gate(spatial,config.gate_null_permutations,config.seed,codes)
            full=sampler.Graph.to(device)
            model.refresh_selective_geometry(full,spatial)
            del full
        else:
            model.load_state_dict(checkpoint['model_state']);optimizer.load_state_dict(checkpoint['optimizer_state'])
            completed=checkpoint['completed_epochs'];steps=checkpoint['optimizer_steps'];history=checkpoint['history']
            sampler.restore_slice_shards_for_resume(completed)
            if sampler.state_dict()!=checkpoint['sampler_state']:raise ValueError('sampler reconstruction differs from checkpoint')
            restore_rng(checkpoint['rng_state'])
            lineage={'checkpoint':str(Path(resume_from).resolve()),'sha256':checkpoint['_source_sha256'],'completed_epochs':completed}
    except BaseException as error:
        write_json(output/'status.json',{'state':'failed','verified_completed_epochs':completed,'optimizer_steps':steps,'error':str(error)})
        raise
    write_json(output/'config.json',asdict(config));write_json(output/'method_manifest.json',{'trainer':'multislice','training':asdict(config),'execution':execution_record,'model_protocol':model_protocol,'micro_formula_version':1,'loss_protocol':'uniform_slice_sampled_bpr_explicit_ego_l2_v1'})
    write_json(output/'data.json',{'spot_ids':data.spot_ids,'peak_ids':data.peak_ids,'slice_ids':data.slice_ids,'dna_ids':[],'dna_kind':None})
    np.save(output/'coordinates.npy',data.coordinates,allow_pickle=False)
    started=time.perf_counter();cache={}
    if history:pd.DataFrame(history).to_csv(output/'training_history.tsv',sep='\t',index=False)
    try:
        save_checkpoint(output/'checkpoints/latest.pt',model,optimizer,inputs,config,completed,steps,sampler,history,execution_record)
        for epoch in range(completed,config.epochs):
            epoch_started=time.perf_counter();sampler.maybe_refresh_slice_shards(epoch)
            result=train_epoch(sampler,model,optimizer,device,config,epoch,cache)
            if result['optimizer_steps']!=config.updates_per_epoch or result['propagation_steps']!=config.updates_per_epoch*sampler.num_slices:
                raise RuntimeError('balanced update protocol counters differ')
            full_gate=sampler.Graph.to(device) if (epoch+1)%config.gate_refresh_every==0 else None
            refreshed=(model.refresh_if_due(epoch+1,full_gate,spatial,codes,config.gate_refresh_every) if full_gate is not None else False)
            del full_gate
            completed=epoch+1;steps+=result['optimizer_steps']
            row={**result,'epoch':completed,'optimizer_steps_epoch':result['optimizer_steps'],'optimizer_steps':steps,
                 'gate_refreshed_after_loss':refreshed,'gate_epoch':int(model.gate_epoch),
                 'gate_mean':float(model.cached_first_layer_gate.mean()),
                 'epoch_seconds_excluding_export':time.perf_counter()-epoch_started}
            row.update(geometry_summary(model))
            history.append(row)
            pd.DataFrame(history).to_csv(output/'training_history.tsv',sep='\t',index=False)
            per_slice=[dict(epoch=item['epoch'],slice_code=int(code),slice_name=sampler.slice_names[int(code)],**values) for item in history for code,values in item['per_slice'].items()]
            pd.DataFrame(per_slice).to_csv(output/'per_slice_epoch_diagnostics.tsv',sep='\t',index=False)
            write_json(output/'status.json',{'state':'running','pid':os.getpid(),'verified_completed_epochs':completed,'optimizer_steps':steps})
            print(f'EPOCH {completed}/{config.epochs} loss={row["loss"]:.9f} seconds={row["epoch_seconds_excluding_export"]:.3f} optimizer_steps={steps} propagations={result["propagation_steps"]}',flush=True)
            export_due=config.result_export_due(completed)
            if export_due:
                values=infer_tailgate_cpu(model,sampler.Graph,None,sampler.scaled_full_spatial_adjacency,True,1.,0.)
                export_embeddings(output/'embeddings'/f'{completed}_epoch',data,model,values,completed,
                                  data_fingerprint=inputs.metadata['data_fingerprint'],final=completed==config.epochs)
            if completed%config.checkpoint_every==0 or export_due:
                latest=output/'checkpoints/latest.pt'
                save_checkpoint(latest,model,optimizer,inputs,config,completed,steps,sampler,history,execution_record)
                if export_due:
                    destination=output/'checkpoints'/f'epoch_{completed:04d}.pt'
                    temporary=destination.with_suffix('.pt.tmp');shutil.copyfile(latest,temporary);os.replace(temporary,destination)
                print(f'CHECKPOINT epoch={completed} path={latest}',flush=True)
            if os.environ.get('SEAREG_JOB_PATH'):
                from ..jobs import record_progress
                record_progress(output,row,config)
        metadata=stage_metadata(output,upstreams={'multislice_inputs':inputs},trainer='multislice',
            configured_epochs=config.epochs,verified_completed_epochs=completed,final_adopted_output_epoch=completed,
            optimizer_steps=steps,data_fingerprint=inputs.metadata['data_fingerprint'],training_inputs_fingerprint=inputs.fingerprint,
            config=asdict(config),execution=execution_record,environment=environment(),parent=lineage,
            model_protocol=model_protocol,micro_formula_version=1,loss_protocol='uniform_slice_sampled_bpr_explicit_ego_l2_v1',
            sampler_state=sampler.state_dict(),gate_epoch=int(model.gate_epoch),export_inference_device='cpu',
            geometry=geometry_summary(model),
            training_wall_seconds_including_setup_and_exports=time.perf_counter()-started)
        write_json(output/'status.json',{'state':'complete',**metadata});seal_artifact(output,'training_run',metadata)
        print(f'COMPLETE epochs={completed} optimizer_steps={steps} output={output}',flush=True)
    except BaseException as error:
        write_json(output/'status.json',{'state':'failed','verified_completed_epochs':completed,'optimizer_steps':steps,'error':str(error)})
        raise
    return output
