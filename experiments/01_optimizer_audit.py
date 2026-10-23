"""Post-hoc longer-run diagnostic for the largest initialization NLL gaps.

No recovery threshold is selected from these results. This checks whether the
fixed-budget experiment stopped before the two starts reached similar fits.
"""
import argparse
import json
from pathlib import Path
import torch
from graphgp.recovery import read_rows, sample_problem, fit
from graphgp.validation import write_csv, environment_metadata


def audit(source, output, steps=1000, cases=3):
    torch.set_num_threads(1)
    config=json.loads((Path(source)/'config.json').read_text())
    Path(output).mkdir(parents=True,exist_ok=True)
    (Path(output)/'config.json').write_text(json.dumps(dict(source_config=config,
        audit_steps=steps,cases=cases,selection='largest A initialization exact-NLL gaps'),indent=2)+'\n')
    (Path(output)/'environment.json').write_text(json.dumps(environment_metadata(),indent=2)+'\n')
    rows=[r for r in read_rows(Path(source)/'fits.csv') if r['method']=='A' and r['status']=='ok']
    lookup={(r['dataset'],r['init_id']):r for r in rows}
    gaps=sorted([(abs(float(r['exact_nll'])-float(lookup[(r['dataset'],'1')]['exact_nll'])),r['dataset'])
                 for r in rows if r['init_id']=='0'],reverse=True)[:cases]
    results,history=[],[]
    for gap,dataset in gaps:
        row=lookup[(dataset,'0')]
        problem=sample_problem(row['family'],n=config['n'],seed=int(row['seed']),heat_time=float(row['true_heat_time']),
            noise=float(row['true_noise']),train_fraction=float(row['train_fraction']))
        for init_id in ('0','1'):
            initial=lookup[(dataset,init_id)]
            for method in ('A','C'):
                result,trajectory,_=fit(problem,method,steps=steps,
                    learn_amplitude=config['learn_amplitude'],
                    initial=(float(initial['initial_heat_time']),float(initial['initial_noise'])))
                meta=dict(dataset=dataset,method=method,init_id=init_id,steps=steps,
                          original_exact_nll=initial['exact_nll'],original_start_gap=gap)
                results.append(meta|result)
                history.extend(meta|point for point in trajectory)
                print(dataset,method,init_id,result['heat_time'],result['noise'],result['exact_nll'],flush=True)
                write_csv(Path(output)/'fits.csv',results)
                write_csv(Path(output)/'trajectories.csv',history)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='results/heat_recovery')
    parser.add_argument('--output',default='results/optimizer_audit')
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--cases',type=int,default=3)
    args=parser.parse_args()
    if args.steps < 1 or args.cases < 1:
        parser.error('steps and cases must be positive')
    audit(args.source,args.output,args.steps,args.cases)
