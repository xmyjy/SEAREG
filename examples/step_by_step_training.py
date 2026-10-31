"""Inspect five stages with motif=True/False and auto/read/scan sources.

The train stage runs synchronously. For long runs, stop at training_inputs
and submit them with start_training_job(); see docs/continuous_pipeline.rst.
"""
import argparse
import json
import seareg as sr


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atac',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--motif',choices=('true','false'),default='true')
    parser.add_argument('--motif-source',choices=('auto','read','scan'),default='auto')
    parser.add_argument('--genome',help='matching assembly if absent from counts.uns["genome"]')
    parser.add_argument('--fasta',help='optional local matching FASTA instead of downloading')
    parser.add_argument('--dna',help='optional existing named peak-motif matrix')
    parser.add_argument('--rscript',default='Rscript')
    parser.add_argument('--epochs',type=int,default=None,help='override the spot-count default')
    parser.add_argument('--shards',type=int,default=None,help='override the filtered-edge default')
    parser.add_argument('--until',choices=('filter','motif','network','training_inputs','train'),default='train')
    args=parser.parse_args()
    training={'device':'cpu','loss_backend':'auto','loss_workspace_mb':1024.0,
              'grouped_loss':True,'weighted_ego_l2':True,'resident_cache':True}
    training.update({name:value for name,value in {'epochs':args.epochs,'shards':args.shards}.items()
                     if value is not None})
    flow=sr.Workflow.from_counts(args.atac,motif=args.motif=='true',motif_source=args.motif_source,
        output=args.output,genome=args.genome,fasta=args.fasta,dna=args.dna,rscript=args.rscript,
        training=training)
    stages=[('filter',flow.filter),('motif',flow.prepare_motif),('network',flow.prepare_network),
            ('training_inputs',flow.prepare_training),('train',flow.train)]
    for name,execute in stages:
        value=execute()
        print(json.dumps({'step':name,'state':'skipped' if value is None else 'complete',
            'path':None if value is None else str(value.path)}),flush=True)
        if name==args.until:break


if __name__=='__main__':main()
