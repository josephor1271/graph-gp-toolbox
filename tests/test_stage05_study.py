"""Paired cases, mandatory accounting, failures and per-operation qualification."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from graphgp.benchmarking.stage05_cases import cases,make_problem,probe_seed,Stage05Case,ARMS
from graphgp.benchmarking.stage05_coordinator import request,requests,run_worker,candidate_requests
from graphgp.benchmarking.stage05_report import validate,operation_accuracy,candidate_decision


class Stage05Study(unittest.TestCase):
    def test_archival_provenance_checks_historical_bytes_and_fails_closed(self):
        spec = importlib.util.spec_from_file_location('curator', 'experiments/05_curate_evidence.py')
        curator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(curator)
        content = b'original measured source\n'
        environment = dict(
            source_commit='a' * 40,
            source_hashes={'worker.py': hashlib.sha256(content).hexdigest()},
        )
        revision = environment['source_commit'] + ':worker.py'
        with patch.object(curator.subprocess, 'check_output', return_value=content) as git_show:
            curator._verify_measured_sources(environment)
            git_show.assert_called_once_with(['git', 'show', revision], stderr=subprocess.PIPE)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                curator._verify_measured_sources({**environment, 'source_hashes': {'worker.py': '0' * 64}})
        failures = (
            OSError('Git unavailable'),
            subprocess.CalledProcessError(128, ['git', 'show'], stderr=b'unknown commit'),
            subprocess.CalledProcessError(128, ['git', 'show'], stderr=b'path missing'),
        )
        for error in failures:
            with patch.object(curator.subprocess, 'check_output', side_effect=error):
                with self.assertRaisesRegex(RuntimeError, 'historical measured source'):
                    curator._verify_measured_sources(environment)

    def test_fixed_cases_streams_fingerprints_and_nested_disjoint_targets(self):
        chosen={c.case_id:c for c in cases()}
        self.assertEqual(len(chosen),10)
        for base,variants in [('s05_n100',['s05_q016','s05_q256']),('s05_n400',['s05_q256_n400'])]:
            p=make_problem(chosen[base])
            for name in variants:
                q=make_problem(chosen[name]);self.assertEqual(p['base_fingerprint'],q['base_fingerprint'])
                np.testing.assert_array_equal(p['train_ids'],q['train_ids'])
                np.testing.assert_array_equal(p['target_ids'][:min(len(p['target_ids']),len(q['target_ids']))],q['target_ids'][:min(len(p['target_ids']),len(q['target_ids']))])
                self.assertEqual(len(set(q['target_ids'])&set(q['train_ids'])),0)
                np.testing.assert_array_equal(p['solve_rhs'],q['solve_rhs'])
        self.assertEqual(len({probe_seed(chosen['s05_n100'],label) for label in (301,302,303)}),3)
        with self.assertRaises(ValueError):Stage05Case('s05_n100',100,8,199,64)

    def test_secondary_psd_root_is_declared_data_only(self):
        primary=make_problem(cases()[2]);secondary=make_problem(cases()[-1]);meta=secondary['stream_metadata']['temporal_factor']
        self.assertEqual(primary['stream_metadata']['temporal_factor']['method'],'plain temporal Cholesky')
        self.assertIn('secondary-only',meta['method']);self.assertLess(meta['maximum_covariance_correction'],1e-12)
        self.assertGreaterEqual(meta['minimum_eigenvalue'],-meta['eigenvalue_tolerance'])
        self.assertTrue(np.isfinite(secondary['y']).all())
        self.assertLess(meta['effective_rank'],32)

    def test_predeclared_matrix_interleaves_and_accounts_all_operations(self):
        specs=requests();self.assertEqual(len(specs),len({s['run_id'] for s in specs}))
        small=[s for s in specs if s['case']['case_id']=='s05_n400' and s['campaign']=='primary' and s['operation']=='mll']
        self.assertEqual([s['arm'] for s in small],list(ARMS)*2)
        self.assertEqual({s['operation'] for s in specs if s['campaign']=='primary'}, {'build','mll','train3','mean','marginal','mean_warm','marginal_warm','solve'})
        large=[s for s in specs if s['case']['n']==6400]
        self.assertTrue(all(s['operation'] in ('graph_mvp','full_mvp','torch_csr','scipy_csr') for s in large))
        self.assertTrue(all(s['arm'].startswith('C') for s in specs if s['case']['num_times']==32))
        self.assertEqual({s['probe_seed'] for s in specs if s['campaign']=='accuracy'}, {301,302,303})
        self.assertEqual(len(candidate_requests()),90)

    def test_timeout_nonzero_signal_oom_invalid_output_and_unsupported_rss(self):
        c=cases()[0];spec=request(c,'mll',ARMS[0])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'workers').mkdir()
            def timeout(command,**kw):
                self.assertEqual(kw['timeout'], .001)
                self.assertEqual(command[:3], [sys.executable, '-m', 'graphgp.benchmarking.stage05_worker'])
                raise subprocess.TimeoutExpired(command,kw['timeout'])
            self.assertEqual(run_worker(spec,root,timeout=.001,runner=timeout)['status'],'timeout')
            for code,status in [(2,'nonzero_exit'),(-9,'signal')]:
                r=run_worker(spec,root,runner=lambda *a,**k:subprocess.CompletedProcess(a[0],code,'','out of memory'))
                self.assertEqual(r['status'],status);self.assertTrue(r['oom_indicator'])
            missing = run_worker(
                spec, root, runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, '', '')
            )
            self.assertEqual(missing['status'], 'missing_output')
            def cannot_spawn(command, **kwargs):
                raise OSError('injected spawn failure')
            self.assertEqual(run_worker(spec, root, runner=cannot_spawn)['status'], 'launch_error')
            def invalid(command,**kw):
                Path(command[-1]).write_text('{}');return subprocess.CompletedProcess(command,0,'','')
            self.assertEqual(run_worker(spec,root,runner=invalid)['status'],'invalid_output')
        row={**spec,'status':'ok','elapsed_seconds':.1,'rss_peak_mib':None}
        self.assertEqual(validate([spec],[row])['status_counts'],{'ok':1})
        for bad in ([],[row,row]):
            with self.assertRaises(ValueError):validate([spec],bad)
        with self.assertRaises(ValueError):validate([spec],[{**row,'elapsed_seconds':0}])
        with self.assertRaises(ValueError):validate([spec],[{**row,'result':{'bad':float('nan')}}])

    def test_wrong_sign_and_residual_disqualify_the_specific_operation(self):
        c=cases()[0];spec=request(c,'mll',ARMS[2]);reference=dict(nll=.9,raw_gradients=dict(heat_time=.01,lengthscale=-.1))
        row={**spec,'status':'ok','result':dict(nll=.9,raw_gradients=dict(heat_time=-.001,lengthscale=-.1))}
        solved=dict(status='ok',result=dict(relative_residual=[1e-5]*3))
        check=operation_accuracy(row,reference,solved)
        self.assertEqual(check['screen_status'],'pass');self.assertEqual(check['training_screen_status'],'fail')
        self.assertEqual(check['wrong_sign_parameters'],['heat_time'])
        solved['result']['relative_residual'][1]=.1
        self.assertIn('true_residual',operation_accuracy(row,reference,solved)['screen_failures'])
        self.assertEqual(operation_accuracy(row,None,solved)['reference_status'],'missing')

    def test_candidate_gates_include_all_widths_and_operator_memory(self):
        rows=[]
        for n in (400,1600,6400):
            for width in (1,8,32):
                for name,seconds in [('scaled_baseline',1.),('scaled_candidate',.8)]:
                    rows.append(dict(campaign=name,n=n,rhs_width=width,attempted=3,successful=3,seconds_median=seconds,resident_graph_kernel_bytes_median=100))
        self.assertTrue(candidate_decision(rows)['passed'])
        rows[-1]['resident_graph_kernel_bytes_median']=116
        self.assertFalse(candidate_decision(rows)['passed'])
        rows[-1]['resident_graph_kernel_bytes_median']=100;rows[-1]['seconds_median']=1.11
        self.assertFalse(candidate_decision(rows)['passed'])


if __name__=='__main__':unittest.main()
