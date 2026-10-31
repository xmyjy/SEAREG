"""Unified stage CLI. Read-only check does not configure caches or devices."""

import argparse
import json
from pathlib import Path
import sys
from .pipeline import Workflow


def main(argv=None):
    parser = argparse.ArgumentParser(description="SEAREG public workflow")
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config',type=Path)
    source.add_argument('--atac',type=Path,help='spot-by-peak counts H5AD; no JSON required')
    parser.add_argument('--output',type=Path)
    def boolean(value):
        if value.lower() in ('true','1','yes'):return True
        if value.lower() in ('false','0','no'):return False
        raise argparse.ArgumentTypeError('expected true or false')
    parser.add_argument('--motif',type=boolean,default=None,metavar='{true,false}',help='use peak-motif information (default true)')
    parser.add_argument('--motif-source',choices=('auto','read','scan'),default=None,
                        help='auto reuse/build; read existing only; scan with original R even if matrix exists')
    parser.add_argument('--genome',help='declared build; alternatively counts.uns["genome"]')
    parser.add_argument('--fasta',type=Path,help='local matching reference; otherwise download declared build')
    parser.add_argument('--dna',type=Path,help='existing peak-motif matrix; auto-build if missing')
    parser.add_argument('--rscript',default=None)
    parser.add_argument('--no-auto-build-dna',action='store_true')
    parser.add_argument('--no-download-reference',action='store_true')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--hidden-dim',type=int)
    parser.add_argument('--lr',type=float)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--shards',type=int)
    parser.add_argument('--pn-r',type=int)
    parser.add_argument('--tr',type=float,help='training-positive fraction in (0,1]')
    parser.add_argument('--spatial-weight',type=float)
    parser.add_argument('--dna-weight',type=float)
    parser.add_argument('--device',help='cpu or worker cuda:0')
    parser.add_argument('--physical-gpu',type=int)
    parser.add_argument('stage',choices=['check','plan','filter','prepare-inputs','prepare-motif','prepare-sequences',
        'annotate','select-edges','prepare-network','prepare-training','train','analyze','render','run'])
    parser.add_argument('--resume-from',type=Path,help='explicit checkpoint; only train/run')
    parser.add_argument('--training-run',type=Path,help='explicit completed training directory; only analyze')
    args = parser.parse_args(argv)
    try:
        training_names=('epochs','hidden_dim','lr','seed','shards','pn_r','tr',
                        'spatial_weight','dna_weight','device')
        training={name:getattr(args,name) for name in training_names if getattr(args,name) is not None}
        if args.config is not None:
            if training or any(value is not None for value in (args.output,args.motif,args.motif_source,args.genome,args.fasta,args.dna,args.rscript,args.physical_gpu)) or args.no_auto_build_dna or args.no_download_reference:
                raise ValueError('set options inside --config, or use the direct --atac entry')
            flow=Workflow(args.config)
        else:
            execution={} if args.physical_gpu is None else {'physical_gpu':args.physical_gpu}
            flow=Workflow.from_counts(args.atac,motif=True if args.motif is None else args.motif,
                motif_source=args.motif_source or 'auto',
                output=args.output,genome=args.genome,fasta=args.fasta,dna=args.dna,
                rscript=args.rscript or 'Rscript',auto_build_dna=not args.no_auto_build_dna,
                download_reference=not args.no_download_reference,training=training,execution=execution)
        if args.resume_from is not None and args.stage not in ('train','run'):
            raise ValueError('--resume-from is valid only for train/run')
        if args.training_run is not None and args.stage!='analyze':
            raise ValueError('--training-run is valid only for analyze')
        if args.stage == "check":
            result = flow.check()
        elif args.stage=='plan':result=flow.plan()
        else:
            from ._runtime import configure_runtime
            configure_runtime()
            kwargs={'resume_from':args.resume_from} if args.stage in ('train','run') else {}
            if args.stage=='analyze' and args.training_run is not None:
                kwargs['training']=args.training_run
            value = getattr(flow, args.stage.replace("-", "_"))(**kwargs)
            result = ({'state':'skipped','stage':args.stage,'reason':'motif=False'} if value is None else
                      {"state": "complete", "stage": args.stage, "path": str(value.path)})
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0
    except Exception as exc:
        print(f"SEAREG: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
