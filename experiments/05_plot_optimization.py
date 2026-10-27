#!/usr/bin/env python3
"""Regenerate exactly three Stage 05 figures from CSV evidence only."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS=('A_dense_direct','B_dense_iterative','C32_sparse_iterative','C16_sparse_iterative')
COLORS=('#3265a0','#8a65a6','#d17626','#418b67')
LABELS=('A dense/direct','B dense/CG+SLQ','C32 sparse/CG+SLQ','C16 sparse/CG+SLQ')


def read_csv(path):
    with Path(path).open(newline='') as stream:return list(csv.DictReader(stream))


def number(row,key):
    try:value=float(row.get(key,''))
    except (TypeError,ValueError):return None
    return value if math.isfinite(value) else None


def save(fig,output,name):
    fig.savefig(output/name,dpi=170,bbox_inches='tight',metadata={'Software':'Matplotlib Stage05 CSV-only'})
    plt.close(fig)


def line(ax,rows,*,x='n',label,color,style='-',qualified=False):
    points=sorted((number(r,x),number(r,'seconds_median'),r) for r in rows if number(r,'seconds_median') is not None)
    if not points:return
    xs=[p[0] for p in points];ys=[p[1] for p in points]
    lower=[max(0,p[1]-(number(p[2],'seconds_min') or p[1])) for p in points]
    upper=[max(0,(number(p[2],'seconds_max') or p[1])-p[1]) for p in points]
    ax.errorbar(xs,ys,yerr=[lower,upper],color=color,linestyle=style,label=label,capsize=2,lw=1.2)
    for xx,yy,r in points:
        good=not qualified or r.get('qualified_success')==r.get('attempted')
        ax.plot(xx,yy,'o',ms=4,color=color,markerfacecolor=color if good else 'white')
    for r in rows:
        count=(number(r,'failed') or 0)+(number(r,'timeouts') or 0)
        if count:ax.plot(number(r,x),.03,'x',color=color,transform=ax.get_xaxis_transform())


def decomposition(summary,accuracy,output):
    fig,axes=plt.subplots(2,3,figsize=(12,6.6),constrained_layout=True)
    for ax,op,title in zip(axes[0],('mll','mean','marginal'),('MLL + backward','Cold mean only','Cold exact marginals')):
        for arm,color,label in zip(ARMS,COLORS,LABELS):
            rows=[r for r in summary if r['campaign']=='primary' and r['arm']==arm and r['operation']==op and r['case_id'] in ('s05_n016','s05_n049','s05_n100','s05_n400','s05_n1600')]
            line(ax,rows,label=label,color=color,qualified=True)
        ax.set(title=title,xscale='log',yscale='log',xlabel='Graph vertices n (T=8, m=2n)',ylabel='Seconds (process median, min–max)')
        ax.grid(alpha=.2)
    for ax,field,title in zip(axes[1],('nll_abs_error','heat_time_raw_abs_error','mean_max_abs_error'),('NLL error versus exact','Raw tau gradient error versus exact','Cold marginal mean error versus exact')):
        for arm,color,label in zip(ARMS,COLORS,LABELS):
            rows=[r for r in accuracy if r['campaign']=='primary' and r['arm']==arm and r['operation']==('mll' if field!='mean_max_abs_error' else 'marginal') and r['case_id'] in ('s05_n016','s05_n049','s05_n100','s05_n400') and r['repeat']=='0']
            points=sorted((number(r,'n'),max(number(r,field) or 1e-17,1e-17),r) for r in rows if number(r,field) is not None)
            if points:
                ax.plot([p[0] for p in points],[p[1] for p in points],'-o',color=color,lw=1.1,ms=3,label=label)
                for x,y,r in points:
                    if json.loads(r.get('wrong_sign_parameters') or '[]'):ax.plot(x,y,'x',color='black',ms=8)
        ax.set(title=title,xscale='log',yscale='log',xlabel='Graph vertices n',ylabel='Absolute error (floor 1e−17)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Four-arm decomposition; fixed parameters, one signal, probe label 301\nOpen timing markers: unqualified/missing exact screen. Black ×: wrong-sign raw gradient. n16 uses q32; others q64.',fontsize=10)
    save(fig,output,'chebyshev_stage05_decomposition.png')


def operator_plot(summary,candidate,output):
    fig,axes=plt.subplots(1,3,figsize=(12,3.9),constrained_layout=True)
    for ax,width in zip(axes,(1,8,32)):
        for arm,color,label in zip(ARMS[1:],COLORS[1:],LABELS[1:]):
            rows=[r for r in summary if r['campaign']=='operator' and r['operation']=='graph_mvp' and r['arm']==arm and number(r,'rhs_width')==width]
            line(ax,rows,label=label.split('/CG')[0],color=color)
        for campaign,label,color in [('scaled_candidate','C32 scaled CSR candidate','#333333')]:
            rows=[r for r in candidate if r['campaign']==campaign and number(r,'rhs_width')==width]
            line(ax,rows,label=label,color=color,style='--')
        for op,label,style in [('torch_csr','Torch raw CSR',':'),('scipy_csr','SciPy raw CSR (reference)',(0,(4,2)))]:
            rows=[r for r in summary if r['campaign']=='micro' and r['operation']==op and number(r,'rhs_width')==width]
            line(ax,rows,label=label,color='#999999',style=style)
        ax.set(title=f'Graph-only RHS width {width}',xscale='log',yscale='log',xlabel='Graph vertices n',ylabel='Seconds per product');ax.grid(alpha=.2)
    axes[0].legend(fontsize=6)
    fig.suptitle('Prebuilt graph factors: one warmup + five products in each of three fresh workers\nOperator timings exclude construction; dense only through n400. CSR primitives do not compute a heat kernel.',fontsize=10)
    save(fig,output,'chebyshev_stage05_operator.png')


def diagonal_plot(summary,entries,output):
    fig,axes=plt.subplots(1,3,figsize=(12,3.9),constrained_layout=True)
    for strategy,label,color in [('scalar_reference','Original scalar','#8a65a6'),('batched','Batched width ≤16','#d17626')]:
        rows=[r for r in summary if r['campaign']=='entries_'+strategy]
        line(axes[0],rows,x='unique_columns',label=label,color=color)
        for unique in (50,256):
            samples=[json.loads(r['operator'])['operator_matmul_calls']/6 for r in entries if r['operation']=='entries' and r['campaign']=='entries_'+strategy and number(r,'unique_columns')==unique and r['status']=='ok']
            if samples:axes[1].plot(unique,statistics.median(samples),'o',color=color,label=label if unique==50 else None)
        rows=[r for r in summary if r['operation']=='marginal' and r['arm']==ARMS[2] and r['case_id'] in ('s05_q016','s05_n100','s05_q256') and r['campaign']==('entry_scalar_control' if strategy=='scalar_reference' else 'primary')]
        line(axes[2],rows,x='q',label=label,color=color)
    axes[0].set(title='256 selected entries, n400',xlabel='Unique requested columns',ylabel='Seconds per request',yscale='log')
    axes[1].set(title='Calls per selected-entry request',xlabel='Unique requested columns',ylabel='Graph products (warmup excluded)',yscale='log')
    axes[2].set(title='Complete cold exact marginals, n100',xlabel='Held-out targets q',ylabel='Seconds per prediction',yscale='log')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Exact polynomial entry batching; same C32 covariance, CG/SLQ settings and targets\nIsolated entries: three workers; full posterior: two workers per q. Error bars show process min–max.',fontsize=10)
    save(fig,output,'chebyshev_stage05_diagonal.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,default=Path('docs/evidence/chebyshev_stage05'))
    parser.add_argument('--output',type=Path,default=Path('docs/figures'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    summary=read_csv(args.evidence/'summary.csv');accuracy=read_csv(args.evidence/'accuracy.csv');entries=read_csv(args.evidence/'entry_diagonal.csv')
    candidate_path=args.evidence/'scaled_candidate.csv'
    decomposition(summary,accuracy,args.output);operator_plot(summary,read_csv(candidate_path) if candidate_path.exists() else [],args.output)
    diagonal_plot(summary,entries,args.output)


if __name__=='__main__':main()
