"""Run multiple deterministic seeds and combine the measured forecasting metrics.

Example (PowerShell):
  python scripts/run_seed_sweep.py --dataset metr_la --fast --max-rows 1200 --rounds 3 --seeds 17,23,41
This intentionally takes longer than a single quick run.
"""
import argparse
from pathlib import Path
import pandas as pd
from hilofed.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['metr_la','pems_bay'],default='metr_la')
    p.add_argument('--fast',action='store_true')
    p.add_argument('--max-rows',type=int,default=1200)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--seeds',default='17,23,41')
    args=p.parse_args()
    seeds=[int(x.strip()) for x in args.seeds.split(',') if x.strip()]
    all_metrics=[]
    for seed in seeds:
        out=ROOT/'seed_sweeps'/f'seed_{seed}'
        meta,metrics,costs=run_pipeline(args.dataset,args.fast,args.max_rows,args.rounds,seed,ROOT/'data',out)
        metrics['seed']=seed
        all_metrics.append(metrics)
        print(f"seed={seed} runtime={meta['runtime_seconds']:.1f}s metrics_rows={len(metrics)}")
    combined=pd.concat(all_metrics,ignore_index=True)
    target=ROOT/'results'/f'seed_sweep_metrics_{args.dataset}.csv'
    target.parent.mkdir(parents=True,exist_ok=True)
    combined.to_csv(target,index=False)
    summary=combined.groupby(['method','horizon_minutes']).agg(rmse_mean=('rmse_mph','mean'),rmse_std=('rmse_mph','std'),mae_mean=('mae_mph','mean'),mae_std=('mae_mph','std'),seeds=('seed','nunique')).reset_index()
    summary.to_csv(ROOT/'results'/f'seed_sweep_summary_{args.dataset}.csv',index=False)
    print(f'Wrote {target} and seed_sweep_summary_{args.dataset}.csv')
if __name__=='__main__': main()
