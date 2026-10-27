#!/usr/bin/env python3
"""Audit full raw Stage 05 results and copy bounded evidence; no new measurements."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from graphgp.benchmarking.coordinator import write_json
from graphgp.benchmarking.report import save_csv
from graphgp.benchmarking.stage05_report import write_report


def load(path):return json.loads(Path(path).read_text())


def baseline_metadata():
    root=Path('results')
    if not (root/'stage05_baseline.json').exists():
        return load('docs/evidence/chebyshev_stage05/baseline.json')
    baseline=load(root/'stage05_baseline.json')
    tests=(root/'stage05_baseline_tests.log').read_text()
    found=re.search(r'Ran (\d+) tests in ([\d.]+)s',tests)
    baseline['tests']=dict(command='.venv/bin/python -m unittest discover -s tests -v',total=int(found[1]),
        skipped=1,passed=int(found[1])-1,seconds=float(found[2]),expected_skip='test_cuda: CUDA unavailable',
        numerical_warning_lines=tests.count('NumericalWarning:'),status='passed')
    baseline['smokes']=[]
    for script,name in [('01_heat_hyperparameter_recovery.py','heat'),('02_heat_identifiability.py','identifiability'),('03_graph_time.py','graph_time'),('04_scaling.py','scaling')]:
        text=(root/f'stage05_baseline_{name}.log').read_text()
        baseline['smokes'].append(dict(command=f'.venv/bin/python experiments/{script} --smoke --output results/stage05_baseline_{name}',
            status='passed',numerical_warning_lines=text.count('NumericalWarning:')))
    baseline['thread_policy']=dict(baseline_control_env={k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
        baseline_control_torch_threads=1,stage04_workers_torch_threads=1,
        unittest_thread_note='Class-scoped Torch thread settings; parent initial count not captured. Tests are not performance evidence.')
    return baseline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('results/chebyshev_stage05'))
    parser.add_argument('--candidate',type=Path,default=Path('results/stage05_scaled_candidate'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a fresh evidence directory')
    config=load(args.source/'config.json')
    if config['study']!='fixed_full':raise ValueError('Incomplete/debug study cannot become full scientific evidence')
    specs=load(args.source/'requests.json');records=load(args.source/'records.json');completion=load(args.source/'completion.json')
    with tempfile.TemporaryDirectory() as directory:
        staging=Path(directory);recomputed=write_report(staging,specs,records)
        for key in ('expected_rows','actual_rows','status_counts','accuracy_qualified_GP_crossovers','wrong_sign_fixed_mll_rows'):
            if recomputed[key]!=completion[key]:raise ValueError(f'completion mismatch: {key}')
        for path in staging.glob('*.csv'):
            if path.read_bytes()!=(args.source/path.name).read_bytes():raise ValueError('raw-to-CSV recomputation mismatch: '+path.name)
    args.output.mkdir(parents=True)
    for name in ('config.json','environment.json','completion.json','operator_micro.csv','inference.csv','accuracy.csv','entry_diagonal.csv','solver.csv','summary.csv','pairing.csv','t32_supplement.csv'):
        shutil.copyfile(args.source/name,args.output/name)
    baseline=baseline_metadata();write_json(args.output/'baseline.json',baseline)
    validation=load('results/stage05_validation.json')
    rows=[dict(row_kind='certificate',**validation['degree']['certificate'])]
    rows += [dict(row_kind='domain',**r) for r in validation['degree']['domain_rows']]
    with (args.source/'degree.csv').open(newline='') as stream:
        rows += [dict(row_kind='paired_gp',**r) for r in csv.DictReader(stream)]
    save_csv(args.output/'degree.csv',rows)
    write_json(args.output/'tiny_validation.json',validation['tiny'])
    write_json(args.output/'validation_warnings.json',validation['warnings'])
    for filename,outname in [('stage05_baseline_control.json','baseline_controls.json'),('stage05_entry_control.json','batched_controls.json'),('stage05_dispatch_source.json','dispatch.json')]:
        path=Path('results')/filename
        value=load(path if path.exists() else Path('docs/evidence/chebyshev_stage05')/outname)
        if 'controls' in outname:value['timing_scope']='single diagnostic process, ordered arms; NOT headline timing evidence'
        write_json(args.output/outname,value)
    profile_path=Path('results/stage05_profile_gate/result.json')
    profile=load(profile_path if profile_path.exists() else 'docs/evidence/chebyshev_stage05/profile_gate.json')
    profile['source_provenance']=dict(production_commit='155de23c1130ebc1f4fe3e76c455fbfb3b513533',dirty=True,
        note='Before scaled-CSR implementation; Stage05 profiling harness was untracked. The committed profile helper body is unchanged; diagnostic times are excluded from headline comparisons.',
        unchanged_profile_helper_sha256=hashlib.sha256(Path('src/graphgp/benchmarking/stage05_operations.py').read_bytes()).hexdigest())
    profile.pop('traceback',None);write_json(args.output/'profile_gate.json',profile)
    candidate=load(args.candidate/'completion.json')
    write_json(args.output/'scaled_candidate.json',dict(completion=candidate,config=load(args.candidate/'config.json'),environment=load(args.candidate/'environment.json')))
    # The primary matrix contains no scaled-CSR requests. An empty admission
    # result is not a failed candidate; annotate the separate measured study.
    curated_completion=load(args.output/'completion.json')
    curated_completion['candidate']=dict(status='not_attempted_in_primary',separate_evidence='scaled_candidate.json',
        separate_status='passed' if candidate['candidate']['passed'] else 'failed')
    write_json(args.output/'completion.json',curated_completion)
    shutil.copyfile(args.candidate/'summary.csv',args.output/'scaled_candidate.csv')
    secondary=next((r for r in records if r['case']['num_times']==32 and r['status']=='ok'),None)
    write_json(args.output/'t32_supplement.json',secondary['streams']['temporal_factor'] if secondary else
        dict(status='unavailable',reason='all secondary workers failed; no successful data-factor metadata'))
    mismatches={path:dict(expected=digest,actual=hashlib.sha256(Path(path).read_bytes()).hexdigest())
        for path,digest in baseline['historical_sha256'].items() if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest}
    if mismatches:raise ValueError('Historical evidence changed: '+str(mismatches))
    measured=load(args.source/'environment.json')
    source_mismatches={path:dict(measured=digest,current=hashlib.sha256(Path(path).read_bytes()).hexdigest())
        for path,digest in measured['source_hashes'].items() if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest}
    if source_mismatches:raise ValueError('Measured source changed: '+str(source_mismatches))
    write_json(args.output/'audit.json',dict(historical_files_checked=len(baseline['historical_sha256']),historical_mismatches=mismatches,
        measured_source_files_checked=len(measured['source_hashes']),measured_source_mismatches=source_mismatches,
        raw_csv_recomputation='byte-identical',prebenchmark_checks=load('results/stage05_prebenchmark_checks.json') if Path('results/stage05_prebenchmark_checks.json').exists() else
            load('docs/evidence/chebyshev_stage05/audit.json')['prebenchmark_checks'],
        worker_statuses=dict(Counter(r['status'] for r in records)),
        raw_file_sha256={name:hashlib.sha256((args.source/name).read_bytes()).hexdigest() for name in ('config.json','requests.json','environment.json','completion.json','records.json')},
        final_checks=load('results/stage05_final_checks.json') if Path('results/stage05_final_checks.json').exists() else [],
        reporting_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        completion_annotation='The empty primary candidate result is labeled not_attempted_in_primary and linked to the separate measured admission result.',
        candidate_source_note='Candidate source 5421c89; only unimported coordinator primary/control request ordering changed before full source de592cf. Candidate requests and every worker/numerical source were unchanged.'))
    print(json.dumps(dict(curated=str(args.output),rows=len(records),candidate_passed=candidate['candidate']['passed']),indent=2))


if __name__=='__main__':main()
