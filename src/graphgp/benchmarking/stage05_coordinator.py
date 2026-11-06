"""Fixed Stage 05 request matrix and bounded, sequential worker orchestration."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

from graphgp.benchmarking.coordinator import collect_records,execute,environment,write_json,TIMEOUT_SECONDS
from graphgp.benchmarking.stage05_cases import cases,arms,ARMS,BASELINE_SHA


def request(case,operation,arm,*,campaign='primary',profile='balanced',seed=301,repeat=0,width=0,**extra):
    suffix=''.join(f'-{k}{v}' for k,v in sorted(extra.items()))
    return dict(run_id=f'{campaign}-{case.case_id}-{operation}-{arm}-{profile}-s{seed}-r{repeat}-w{width}{suffix}',
                case=case.to_dict(),operation=operation,arm=arm,backend=arm,profile=profile,
                campaign=campaign,probe_seed=seed,repeat=repeat,rhs_width=width,**extra)


def candidate_requests():
    """Single predeclared scaled-CSR attempt; never pooled with the four arms."""
    output=[]
    for case in cases():
        if case.n not in (100,400,1600,6400) or case.case_id.startswith('s05_q') or case.secondary:continue
        for repeat in range(3):
            for width in ((1,8,32,128) if case.n<=1600 else (1,8,32)):
                for scaled in (False,True):
                    output.append(request(case,'graph_mvp',ARMS[2],campaign='scaled_candidate' if scaled else 'scaled_baseline',
                                          repeat=repeat,width=width,scaled_laplacian=scaled))
    return output


def requests(*,smoke=False,profile_only=False,max_cases=None):
    selected=cases()
    if smoke:
        c=selected[0]
        output=[request(c,'reference','reference',campaign='smoke')]
        for op in ('mll','mean','marginal','solve'):
            for arm in ARMS:output.append(request(c,op,arm,profile='exhaustive_tiny',campaign='smoke'))
        output.append(request(c,'entries',ARMS[2],campaign='smoke',entry_count=256,unique_columns=16))
        return output
    if profile_only:
        c=next(c for c in selected if c.case_id=='s05_n400')
        return [request(c,'profile',ARMS[2],campaign='profile_gate')]
    if max_cases is not None:
        if max_cases<1:raise ValueError('max_cases must be positive')
        selected=selected[:max_cases]
    output=[]
    for case in selected:
        available=arms(case)
        if case.secondary:
            for op in ('full_mvp','mll','mean','marginal'):
                for width in ((1,8,32) if op=='full_mvp' else (0,)):
                    for arm in available:output.append(request(case,op,arm,campaign='t32_secondary',width=width))
            continue
        if not case.mvp_only:
            if case.n<=400:output.append(request(case,'reference','reference',campaign='reference'))
            for repeat in range(2 if case.n<=400 else 1):
                for op in ('build','mll','train3','mean','marginal','mean_warm','marginal_warm','solve'):
                    for arm in available:output.append(request(case,op,arm,repeat=repeat))
                    if case.case_id in ('s05_q016','s05_n100','s05_q256','s05_q256_n400') and op in ('mll','mean','marginal'):
                        output.append(request(case,op,ARMS[2],campaign='entry_scalar_control',repeat=repeat,entry_strategy='scalar_reference'))
        # q variants share the same operator/data/RHS. Measure operator work
        # once per n/T/m and document this canonical mapping in config.
        if case.case_id.startswith('s05_q'):continue
        for repeat in range(3):
            for op in ('graph_mvp','full_mvp'):
                for width in (1,8,32):
                    for arm in available:output.append(request(case,op,arm,campaign='operator',repeat=repeat,width=width))
        if case.n in (100,400,1600,6400):
            for repeat in range(3):
                for width in ((1,8,32,128) if case.n<=1600 else (1,8,32)):
                    for op in ('torch_csr','scipy_csr'):output.append(request(case,op,'primitive',campaign='micro',repeat=repeat,width=width))
                    if case.n<=1600:
                        for arm in ARMS[2:]:
                            output.append(request(case,'graph_backward',arm,campaign='micro',repeat=repeat,width=width))
            if not case.mvp_only:
                for repeat in range(3):
                    for width in (1,8,32):
                        for op in ('gather','scatter'):output.append(request(case,op,'primitive',campaign='micro',repeat=repeat,width=width))
        if case.case_id in ('s05_n016','s05_n049','s05_n100'):
            # Primary r0 is seed301 balanced; the two other labels complete
            # the three-label accuracy set without duplicating timing rows.
            for profile,seeds in [('balanced',(302,303)),('strict',(301,302,303))]:
                for seed in seeds:
                    for op in ('mll','mean','marginal','solve'):
                        for arm in ARMS[1:]:output.append(request(case,op,arm,campaign='accuracy',profile=profile,seed=seed))
    chosen={c.case_id:c for c in selected}
    if 's05_n400' in chosen:
        c=chosen['s05_n400']
        for repeat in range(3):
            for unique in (50,256):
                for strategy in ('scalar_reference','batched'):
                    output.append(request(c,'entries',ARMS[2],campaign='entries_'+strategy,repeat=repeat,
                                          entry_count=256,unique_columns=unique,entry_strategy=strategy))
    return output


def run_worker(spec, output, *, timeout=TIMEOUT_SECONDS, runner=subprocess.run):
    return execute(
        spec, output, timeout=timeout, runner=runner,
        worker_module="graphgp.benchmarking.stage05_worker",
    )


def orchestrate(output,*,smoke=False,profile_only=False,max_cases=None,candidate=False):
    from graphgp.benchmarking.stage05_report import write_report
    output=Path(output)
    if output.exists():raise FileExistsError('Use a fresh output directory; measurements are never overwritten')
    (output/'workers').mkdir(parents=True)
    specs=candidate_requests() if candidate else requests(smoke=smoke,profile_only=profile_only,max_cases=max_cases)
    config=dict(study='scaled_candidate' if candidate else ('smoke' if smoke else ('incomplete_profile_debug' if profile_only else ('incomplete_debug' if max_cases else 'fixed_full'))),
        baseline_main=BASELINE_SHA,expected_rows=len(specs),cases=[c.to_dict() for c in cases()],arms=list(ARMS),
        model=dict(heat_time=1.2,lengthscale=.24,noise_variance=.06,amplitude=1,mean=0,default_degree=32,entry_block_width=16),
        profiles=dict(balanced=dict(trace_samples=16,lanczos_iterations=32,cg_tolerance=1e-4,max_cg_iterations=500),strict=dict(trace_samples=16,lanczos_iterations=32,cg_tolerance=1e-6,max_cg_iterations=500)),
        fixed_settings=dict(eval_cg_tolerance='same as cg_tolerance',max_preconditioner_size=0,terminate_cg_by_size=False,max_cholesky_size=0,max_eager_kernel_size=0,fast_pred_var=False),
        screening=dict(nll_abs=.05,raw_gradient_atol=.02,raw_gradient_rtol=.20,posterior_abs=.01,true_residual=.001,wrong_sign_reference_minimum=.001),
        repeats='2 GP processes n<=400; 1 GP n1600/T32; 3 operator processes; fixed probe label301; labels302/303 accuracy only',
        operator_canonical_cases='q variants reuse n100/n400 operator rows: same base fingerprint and RHS; no duplicate operator timings',
        warmup='one untimed then five products; graph backward one untimed forward/backward; no GP warmup/update; warm predictions labeled separately',
        ordering='case, repeat, operation, width, A/B/C32/C16 interleaved; old scalar controls follow each four-arm block; candidate alternates baseline/scaled within each width',
        timeout_seconds=TIMEOUT_SECONDS,master_seed=202605,probe_labels=[301,302,303],
        data='Stage04 degree32 heat-half @ Z @ chol(KT).T, approximate GP-like signal; independent noise; secondary-only T32 spectral PSD data root',
        memory='whole-worker peak RSS includes imports/setup; resident graph-kernel bytes = CSR crow+col+values + spectral nodes + cosine basis + raw tau; scaled replaces raw graph, no duplicate buffer',
        candidate_gates=dict(profile_graph_fraction_min=.40,median_elapsed_reduction_min=.10,minimum_sizes=2,max_regression_fraction=.10,max_resident_operator_memory_increase=.15),
        projected_lifecycle_counts=[1,20,100],debug_max_cases=max_cases)
    env=environment()
    env['source_hashes'].update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('experiments').glob('05_*.py'))})
    write_json(output/'config.json',config);write_json(output/'environment.json',env);write_json(output/'requests.json',specs)
    start = time.perf_counter()
    records = collect_records(specs, output, worker_module="graphgp.benchmarking.stage05_worker")
    completion=write_report(output,specs,records)
    completion['total_worker_wall_seconds']=sum(r['child_wall_seconds'] for r in records)
    completion['total_coordinator_wall_seconds']=time.perf_counter()-start
    write_json(output/'completion.json',completion)
    if smoke and (completion['status_counts']!={'ok':len(specs)} or any(r['screen_status']=='fail' for r in completion['smoke_screen'])):
        raise RuntimeError('Stage05 smoke failed; all outcomes retained')
    print(json.dumps(completion,indent=2),flush=True)
    return completion
