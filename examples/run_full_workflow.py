"""Run all five stages synchronously with configurable inputs and parameters.

For long training runs, use Workflow.prepare_training() followed by
start_training_job(); docs/continuous_pipeline.rst shows the durable job route.
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
    parser.add_argument('--genome',help='actual assembly when scanning and absent from counts metadata')
    parser.add_argument('--fasta',help='optional local matching reference FASTA')
    parser.add_argument('--dna',help='optional existing named peak-motif matrix')
    parser.add_argument('--rscript',default='Rscript')
    parser.add_argument('--epochs',type=int,default=None,help='override the spot-count default')
    parser.add_argument('--hidden-dim',type=int,default=64)
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--seed',type=int,default=1234)
    parser.add_argument('--spatial-weight',type=float,default=.5)
    parser.add_argument('--dna-weight',type=float,default=.1)
    parser.add_argument('--pn-r',type=int,default=10)
    parser.add_argument('--shards',type=int,default=None,help='override the filtered-edge default')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--physical-gpu',type=int)
    args=parser.parse_args()
    sizes={name:value for name,value in {'epochs':args.epochs,'shards':args.shards}.items()
           if value is not None}
    result=sr.run_workflow(args.atac,output=args.output,motif=args.motif=='true',
        motif_source=args.motif_source,genome=args.genome,fasta=args.fasta,dna=args.dna,
        rscript=args.rscript,hidden_dim=args.hidden_dim,lr=args.lr,
        seed=args.seed,spatial_weight=args.spatial_weight,dna_weight=args.dna_weight,
        pn_r=args.pn_r,device=args.device,physical_gpu=args.physical_gpu,**sizes)
    print(json.dumps({'state':'complete','training':str(result.path)},indent=2))


if __name__=='__main__':main()
