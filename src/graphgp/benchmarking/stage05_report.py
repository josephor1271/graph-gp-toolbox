"""Stage 05 accounting and accuracy qualification; standard library only."""
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import statistics

from graphgp.benchmarking.report import save_csv,maximum_error,relative_column_errors
from graphgp.benchmarking.stage05_cases import ARMS

GP_OPS=('mll','train3','mean','marginal','mean_warm','marginal_warm','solve')


def validate(specs,records):
    expected={r['run_id']:r for r in specs}
    observed=[r['run_id'] for r in records]
    if len(expected)!=len(specs) or len(set(observed))!=len(observed):raise ValueError('duplicate measurement keys')
    if set(observed)!=set(expected):raise ValueError('missing or unexpected measurement rows')
    def finite(value):
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        if isinstance(value,(tuple,list)):return all(finite(v) for v in value)
        return not isinstance(value,(int,float)) or math.isfinite(value)
    for row in records:
        if any(row.get(k)!=v for k,v in expected[row['run_id']].items()):raise ValueError('worker request mismatch')
        if row['status']=='ok':
            if not finite(row) or row.get('elapsed_seconds',0)<=0:raise ValueError('invalid successful numeric result/time')
            if row.get('rss_peak_mib') is not None and row['rss_peak_mib']<=0:raise ValueError('invalid peak RSS')
    return dict(expected_rows=len(specs),actual_rows=len(records),status_counts=dict(sorted(Counter(r['status'] for r in records).items())),
                cases_attempted=sorted({r['case']['case_id'] for r in records}),missing_rows=[],duplicate_keys=False)


def compact(row):
    keep=('run_id','campaign','operation','arm','profile','probe_seed','effective_probe_seed','repeat','rhs_width','status',
        'elapsed_seconds','child_wall_seconds','import_seconds','common_imports_preloaded','data_setup_seconds','model_build_seconds',
        'operator_prebuild_seconds','rss_peak_mib','rss_before_operation_mib','memory_scope','memory_status','data_fingerprint','base_fingerprint',
        'settings','graph_backend','inference','degree','entry_block_width','operator','solver','direct','actions','warm_operator','warm_solver','cold_preparation_seconds',
        'endpoint_diagnostic_seconds','endpoint_operator','endpoint_solver',
        'resident_graph_csr_bytes','resident_graph_kernel_bytes','resident_graph_nnz','numerical_warning_count','error_type','error','returncode',
        'entry_strategy','entry_count','unique_columns','scaled_laplacian')
    result={k:row.get(k) for k in keep};result.update(row['case'])
    for key,value in row.get('result',{}).items():
        if key not in ('mean','latent_variance','observation_variance','solutions','exact_solutions','product_sample','products','sample_rows','events'):result[key]=value
    result['warnings']=[dict(category=k[0],message=k[1],count=v) for k,v in Counter((x['category'],x['message']) for x in row.get('warnings',[])).items()]
    return result


def operation_accuracy(row,reference,solve_row=None):
    out={k:row.get(k) for k in ('run_id','arm','profile','probe_seed','repeat','operation','rhs_width','campaign','status')}
    out.update(row['case']);out['reference_status']='available' if reference else 'missing'
    fail=[];out['wrong_sign_parameters']=[]
    if row['status']!='ok':fail.append('operation_'+row['status'])
    if reference is None:fail.append('missing_exact_reference')
    if not fail:
        result=row['result'];op=row['operation'];tiny=row['profile']=='exhaustive_tiny'
        if op=='mll':
            out['nll_reference']=reference['nll'];out['nll_actual']=result['nll'];out['nll_abs_error']=abs(result['nll']-reference['nll'])
            if out['nll_abs_error']>(1e-5 if tiny else .05):fail.append('nll')
            for name,exact in reference['raw_gradients'].items():
                actual=result['raw_gradients'][name];error=abs(actual-exact)
                out.update({name+'_raw_reference':exact,name+'_raw_actual':actual,name+'_raw_abs_error':error})
                if error>(2e-6+2e-4*abs(exact) if tiny else .02+.20*abs(exact)):fail.append(name+'_gradient')
                wrong=abs(exact)>1e-3 and actual*exact<0
                out[name+'_wrong_sign']=wrong
                if wrong:out['wrong_sign_parameters'].append(name)
        elif op in ('mean','marginal','mean_warm','marginal_warm'):
            for key in (('mean',) if op.startswith('mean') else ('mean','latent_variance','observation_variance')):
                out[key+'_max_abs_error']=maximum_error(result[key],reference[key])
                if out[key+'_max_abs_error']>(2e-4 if tiny else .01):fail.append(key)
        elif op=='solve':
            errors=relative_column_errors(result['solutions'],reference['exact_solutions'])
            out.update(relative_solution_errors=errors,maximum_relative_residual=max(result['relative_residual']))
            if out['maximum_relative_residual']>.001:fail.append('true_residual')
        elif op in ('graph_mvp','full_mvp'):
            key=op+str(row['rhs_width']);ref=reference['products'].get(key)
            if ref is None:fail.append('missing_product_reference')
            else:
                if result['sample_rows']!=ref['sample_rows']:raise ValueError('product sample mismatch')
                out['sampled_polynomial_product_max_abs_error']=maximum_error(result['product_sample'],ref['product_sample'])
                out['product_check_scope']='32 fixed evenly spaced rows, all RHS; full small-domain gates separately'
                if out['sampled_polynomial_product_max_abs_error']>1e-9:fail.append('polynomial_product')
        else:fail.append('no_fixed_parameter_oracle_for_operation')
    if row['operation'] in GP_OPS and row['operation'] not in ('solve','train3'):
        if solve_row is None or solve_row['status']!='ok':fail.append('missing_successful_fresh_solve')
        else:
            out['fresh_maximum_relative_residual']=max(solve_row['result']['relative_residual'])
            if out['fresh_maximum_relative_residual']>.001:fail.append('true_residual')
    out['screen_failures']=fail;out['screen_status']='fail' if fail else 'pass'
    out['training_screen_status']='pass' if not fail and not out['wrong_sign_parameters'] else 'fail'
    return out


def accuracy(records):
    refs={r['case']['case_id']:r['result'] for r in records if r['operation']=='reference' and r['status']=='ok'}
    lookup={(r['case']['case_id'],r['arm'],r['profile'],r['probe_seed'],r['campaign'],r['repeat']):r for r in records if r['operation']=='solve'}
    output=[]
    for r in records:
        if r['operation'] not in (*GP_OPS,'graph_mvp','full_mvp') or r['campaign'].startswith('scaled_'):continue
        if r['operation']=='train3':continue # Only same-degree/three-update parity has a valid endpoint reference.
        key=(r['case']['case_id'],r['arm'],r['profile'],r['probe_seed'],r['campaign'],r['repeat'])
        solved=lookup.get(key)
        if solved is None and r['campaign']=='entry_scalar_control':
            solved=lookup.get((*key[:4],'primary',key[-1]))
        output.append(operation_accuracy(r,refs.get(r['case']['case_id']),solved))
    return output


def paired_degree(records):
    key=lambda r:(r['case']['case_id'],r['operation'],r['profile'],r['probe_seed'],r['campaign'],r['repeat'],r['rhs_width'])
    high={key(r):r for r in records if r['arm']==ARMS[2]}
    output=[]
    for low in records:
        if low['arm']!=ARMS[3]:continue
        match=high.get(key(low));out={k:low[k] for k in ('run_id','operation','profile','probe_seed','campaign','repeat','rhs_width')};out.update(low['case'])
        fail=[]
        if not match or match['status']!='ok' or low['status']!='ok':fail.append('missing_successful_pair')
        else:
            a,b=match['result'],low['result'];out['seconds_C32_over_C16']=match['elapsed_seconds']/low['elapsed_seconds']
            if 'nll' in a:
                out['nll_abs_difference']=abs(a['nll']-b['nll'])
                out['C32_raw_gradients']=a['raw_gradients'];out['C16_raw_gradients']=b['raw_gradients']
                out['maximum_raw_gradient_difference']=maximum_error(list(a['raw_gradients'].values()),list(b['raw_gradients'].values()))
                for x,y in zip(a['trajectory'],b['trajectory']):
                    if abs(x['nll']-y['nll'])>2e-4:fail.append('nll')
                    if maximum_error(list(x['raw_gradients'].values()),list(y['raw_gradients'].values()))>2e-3:fail.append('raw_gradient')
            for field in ('mean','latent_variance','observation_variance'):
                if field in a:
                    error=maximum_error(a[field],b[field]);out[field+'_max_abs_difference']=error
                    if error>2e-4:fail.append(field)
        out['pair_status']='fail' if fail else 'pass';out['pair_failures']=sorted(set(fail));output.append(out)
    return output


def pairing(records):
    key=lambda r:(r['case']['case_id'],r['operation'],r['profile'],r['probe_seed'],r['campaign'],r['repeat'],r['rhs_width'])
    dense={key(r):r for r in records if r['arm']==ARMS[1]}
    output=[]
    for r in records:
        if r['arm']!=ARMS[2] or r['operation'] not in GP_OPS:continue
        b=dense.get(key(r))
        if not b:continue
        row=dict(run_id=r['run_id'],dense_run_id=b['run_id'],operation=r['operation'],case_id=r['case']['case_id'],profile=r['profile'],probe_seed=r['probe_seed'])
        row['both_successful']=b['status']==r['status']=='ok'
        if row['both_successful']:
            row['same_data']=r['data_fingerprint']==b['data_fingerprint'];row['same_settings']=r['settings']==b['settings']
            row['same_effective_seed']=r['effective_probe_seed']==b['effective_probe_seed']
            row['same_cg_rhs_shapes']=r['solver']['cg_rhs_shapes']==b['solver']['cg_rhs_shapes']
            row['same_probe_counts']=r['solver']['slq_probe_counts']==b['solver']['slq_probe_counts']
            row['paired']=all(row[k] for k in ('same_data','same_settings','same_effective_seed','same_cg_rhs_shapes','same_probe_counts'))
        output.append(row)
    return output


def summaries(records,checks):
    grouped=defaultdict(list);qualified={r['run_id']:r for r in checks}
    for r in records:
        key=tuple(r.get(k) for k in ('campaign','arm','operation','profile','rhs_width','entry_strategy','unique_columns','scaled_laplacian'))+(r['case']['case_id'],)
        grouped[key].append(r)
    output=[]
    for group in grouped.values():
        first=group[0];row={k:first.get(k) for k in ('campaign','arm','operation','profile','rhs_width','entry_strategy','unique_columns','scaled_laplacian')};row.update(first['case'])
        ok=[r for r in group if r['status']=='ok']
        row.update(attempted=len(group),successful=len(ok),timeouts=sum(r['status']=='timeout' for r in group),failed=sum(r['status'] not in ('ok','timeout') for r in group),
            qualified_success=sum(qualified.get(r['run_id'],{}).get('screen_status')=='pass' and qualified.get(r['run_id'],{}).get('training_screen_status')=='pass' for r in ok))
        for field,prefix in [('elapsed_seconds','seconds'),('rss_peak_mib','rss_peak_mib'),('model_build_seconds','build_seconds'),('resident_graph_kernel_bytes','resident_graph_kernel_bytes')]:
            values=[r[field] for r in ok if r.get(field) is not None]
            if values:row.update({prefix+'_median':statistics.median(values),prefix+'_min':min(values),prefix+'_max':max(values)})
        output.append(row)
    primary={(r['case_id'],r['operation'],r['arm'],r['rhs_width']):r for r in output if r['campaign']=='primary'}
    for r in output:
        r['A_seconds_over_method']=None;r['B_seconds_over_method']=None;r['accuracy_qualified_GP_crossover']=False
        if r['campaign']!='primary':continue
        for arm,label in zip(ARMS[:2],('A','B')):
            control=primary.get((r['case_id'],r['operation'],arm,r['rhs_width']))
            if control and r['qualified_success']==r['attempted'] and control['qualified_success']==control['attempted']:
                r[label+'_seconds_over_method']=control['seconds_median']/r['seconds_median']
        if r['arm'].startswith('C') and r['operation'] in GP_OPS:
            r['accuracy_qualified_GP_crossover']=all(r.get(label+'_seconds_over_method') is not None and r[label+'_seconds_over_method']>1 for label in ('A','B'))
        if r['operation']=='mll' and r.get('seconds_median'):
            build=primary.get((r['case_id'],'build',r['arm'],0))
            if build and build.get('seconds_median'):
                for count in (1,20,100):r[f'projected_build_plus_{count}_mll_seconds']=build['seconds_median']+count*r['seconds_median']
                r['lifecycle_scope']='arithmetic projection of medians; train3 is separate measured compatibility test'
    return output


def candidate_decision(summary):
    baseline={(r['n'],r['rhs_width']):r for r in summary if r['campaign']=='scaled_baseline'}
    pairs=[]
    for r in summary:
        if r['campaign']!='scaled_candidate':continue
        b=baseline.get((r['n'],r['rhs_width']));row=dict(n=r['n'],rhs_width=r['rhs_width'])
        if not b or r['successful']!=r['attempted'] or b['successful']!=b['attempted']:row['complete']=False
        else:
            row.update(complete=True,speedup=b['seconds_median']/r['seconds_median'],
                elapsed_ratio=r['seconds_median']/b['seconds_median'],resident_memory_ratio=r['resident_graph_kernel_bytes_median']/b['resident_graph_kernel_bytes_median'])
        pairs.append(row)
    large=[r for r in pairs if r['n'] in (400,1600,6400)]
    passed=bool(large) and all(r.get('complete') and r['elapsed_ratio']<=1.10 and r['resident_memory_ratio']<=1.15 for r in large)
    widths={r['rhs_width'] for r in large}
    same_width=any(sum(r.get('elapsed_ratio',float('inf'))<=.90 for r in large if r['rhs_width']==width)>=2 for width in widths)
    return dict(passed=passed and same_width,performance_and_memory_gates=passed,same_width_two_size_speed_gate=same_width,pairs=pairs,
                memory_scope='persistent graph-kernel tensor bytes including CSR indices/values, nodes, cosine basis and raw tau; graph replaced, no extra raw CSR; whole-process RSS reported separately')


def write_report(output,specs,records):
    output=Path(output);completion=validate(specs,records);checks=accuracy(records);summary=summaries(records,checks)
    rows=[compact(r) for r in records];byid={r['run_id']:r for r in rows}
    tables=dict(inference=[r for r in rows if r['operation'] in (*GP_OPS,'build')],
                operator_micro=[r for r in rows if r['operation'] in ('graph_mvp','full_mvp','graph_backward','torch_csr','scipy_csr','gather','scatter')],
                accuracy=checks,degree=paired_degree(records),summary=summary,pairing=pairing(records),
                entry_diagonal=[r for r in rows if r['operation']=='entries' or r['operation']=='marginal' and (r['campaign'] in ('primary','entry_scalar_control'))],
                solver=[{**byid[r['run_id']],**({ 'relative_residuals':r['result']['relative_residual']} if r['status']=='ok' and r['operation']=='solve' else {}),
                         'actual_cg_iterations':None} for r in records if r['operation'] in GP_OPS],
                t32_supplement=[r for r in rows if r['num_times']==32])
    for name,table in tables.items():save_csv(output/(name+'.csv'),table)
    completion.update(numerical_warning_records=sum(r.get('numerical_warning_count',0)>0 for r in records),
        numerical_warning_count=sum(r.get('numerical_warning_count',0) for r in records),
        wrong_sign_fixed_mll_rows=sum(bool(r['wrong_sign_parameters']) for r in checks if r['operation']=='mll'),
        accuracy_qualified_GP_crossovers=[{k:r[k] for k in ('case_id','arm','operation','A_seconds_over_method','B_seconds_over_method')} for r in summary if r['accuracy_qualified_GP_crossover']],
        failures=[dict(run_id=r['run_id'],status=r['status'],error=r.get('error')) for r in records if r['status']!='ok'],
        candidate=candidate_decision(summary),
        smoke_screen=[r for r in checks if r['campaign']=='smoke' and r['operation'] in GP_OPS])
    (output/'records.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    return completion
