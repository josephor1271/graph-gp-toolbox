"""One isolated Stage 05 operation, with graph representation/inference separated."""
import argparse
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import time
import traceback
import warnings

from graphgp.benchmarking.worker import THREAD_VARIABLES, peak_rss_mib, current_rss_mib


def run(spec):
    # Set thread limits before importing the matched numerical/diagnostic stack.
    before = {k: os.environ.get(k) for k in THREAD_VARIABLES}
    for key in THREAD_VARIABLES:
        os.environ[key] = '1'
    import_start = time.perf_counter()
    import gpytorch
    import numpy as np
    import torch
    import scipy.linalg
    import graphgp.validation_graph_time
    from linear_operator.operators import DiagLinearOperator
    from graphgp.benchmarking.stage05_cases import (
        Stage05Case, ARM_CONFIG, make_problem, profile, probe_seed,
    )
    from graphgp.benchmarking.reference import selected_dense_reference
    from graphgp.benchmarking.stage05_operations import (
        product_measurement, entry_measurement, mll_measurement, profile_mll,
        gather_scatter_measurement, inference_context, warm_prediction, action_diagnostics,
    )
    from graphgp.benchmarking.solver import solver_diagnostics, relative_residual
    from graphgp.benchmarking.worker import guarded_sparse
    from graphgp.inference.instrumentation import instrument_operator
    from graphgp.inference.predict import predict_marginals
    from graphgp.kernels.graph_time import GraphTimeHeatRBFKernel
    from graphgp.models.graph_time import make_graph_time_gp
    from graphgp.operators.chebyshev import scipy_csr_to_torch_csr
    torch.set_num_threads(1)
    case = Stage05Case(**spec['case'])
    mode = ARM_CONFIG.get(spec['arm'])
    row = {
        **spec, 'status': 'ok', 'import_seconds': time.perf_counter() - import_start,
        'common_imports_preloaded': True, 'threads_before': before,
        'threads_effective': {k: '1' for k in THREAD_VARIABLES}, 'torch_threads': 1,
        'actual_cg_iterations': None,
        'memory_scope': 'whole isolated worker peak including setup/imports',
    }

    # Data and model construction stay outside the requested operation's timer.
    start = time.perf_counter()
    problem = make_problem(case)
    row.update(
        data_setup_seconds=time.perf_counter() - start, data_fingerprint=problem['fingerprint'],
        base_fingerprint=problem['base_fingerprint'], streams=problem['stream_metadata'],
        nnz=problem['L'].nnz, graph_scale=problem['graph_scale'],
    )
    operation = spec['operation']
    width = spec['rhs_width']
    seed = probe_seed(case, spec['probe_seed'])
    if operation == 'reference':
        start = time.perf_counter()
        row['result'] = selected_dense_reference(problem, case)
        row.update(
            elapsed_seconds=time.perf_counter() - start, model_build_seconds=0.0,
            rss_peak_mib=peak_rss_mib(),
        )
        return row
    if operation in ('torch_csr', 'scipy_csr', 'gather', 'scatter'):
        L = problem['L']
        start = time.perf_counter()
        matrix = scipy_csr_to_torch_csr(L, dtype=torch.float64) if operation == 'torch_csr' else L
        row['model_build_seconds'] = time.perf_counter() - start
        row['rss_before_operation_mib'] = current_rss_mib()
        if operation in ('gather', 'scatter'):
            result = gather_scatter_measurement(problem, width, operation)
        elif operation == 'torch_csr':
            result = product_measurement(
                lambda rhs: torch.sparse.mm(matrix, rhs),
                torch.from_numpy(problem['graph_rhs'][:, :width]),
            )
        else:
            result = product_measurement(matrix.dot, problem['graph_rhs'][:, :width])
        row.update(result=result, elapsed_seconds=result['elapsed_seconds'], rss_peak_mib=peak_rss_mib())
        return row
    if not mode:
        raise ValueError('unknown Stage 05 arm')
    if mode['backend'] == 'dense_exact' and (case.n > 400 or case.secondary):
        raise ValueError('dense arm prohibited in sparse-only case')
    if spec['profile'] == 'strict' and case.n > 100:
        raise ValueError('strict profile restricted to small cases')
    if spec['profile'] == 'exhaustive_tiny' and case.n != 16:
        raise ValueError('exhaustive control restricted to n16')
    config = profile(spec['profile'], case.m)
    row.update(
        settings=(config.effective_settings() if mode['inference'] == 'iterative'
                  else {'max_cholesky_size': 10**9, 'fast_computations': False}),
        effective_probe_seed=seed, graph_backend=mode['backend'], inference=mode['inference'],
        degree=mode['degree'] if mode['backend'] == 'sparse_chebyshev' else None,
        entry_block_width=16,
    )
    train = torch.from_numpy(problem['train_pairs'])
    values = torch.from_numpy(problem['train_y'])
    targets = torch.from_numpy(problem['target_pairs'])
    with ExitStack() as setup:
        if mode['backend'] == 'sparse_chebyshev':
            setup.enter_context(guarded_sparse())
        start = time.perf_counter()
        if case.mvp_only:
            model = None
            kernel = GraphTimeHeatRBFKernel(
                problem['L'], num_times=case.num_times, backend=mode['backend'],
                heat_time=1.2, lengthscale=.24, degree=mode['degree'],
                scaled_laplacian=spec.get('scaled_laplacian', False),
            )
        else:
            model = make_graph_time_gp(
                problem['L'], train, values, num_times=case.num_times, backend=mode['backend'],
                heat_time=1.2, lengthscale=.24, noise_variance=.06, degree=mode['degree'],
                scaled_laplacian=spec.get('scaled_laplacian', False),
            )
            model.train()
            model.likelihood.train()
            kernel = model.covar_module
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 2
        row['model_build_seconds'] = time.perf_counter() - start
        start = time.perf_counter()
        base = (
            kernel._full_grid_operator()
            if operation in ('graph_mvp', 'full_mvp', 'graph_backward', 'entries') else None
        )
        row['operator_prebuild_seconds'] = time.perf_counter() - start
        if mode['backend'] == 'sparse_chebyshev':
            graph = kernel.graph_kernel
            L = graph.L
            csr_bytes = sum(
                x.numel() * x.element_size()
                for x in (L.crow_indices(), L.col_indices(), L.values())
            )
            row['resident_graph_csr_bytes'] = csr_bytes
            row['resident_graph_kernel_bytes'] = csr_bytes + sum(
                x.numel() * x.element_size()
                for x in (graph.spectral_nodes, graph.cosine_basis, graph.raw_heat_time)
            )
            row['resident_graph_nnz'] = L.values().numel()
    row['rss_before_operation_mib'] = current_rss_mib()

    with (
        torch.random.fork_rng(devices=[]),
        inference_context(
            spec['arm'], config, entry_strategy=spec.get('entry_strategy', 'batched')
        ) as direct,
        solver_diagnostics() as solver,
        action_diagnostics(case.n, case.num_times) as actions,
        instrument_operator(record_widths=True) as stats,
    ):
        torch.manual_seed(seed)
        if operation == 'build':
            result = {'elapsed_seconds': row['model_build_seconds']}
        elif operation in ('graph_mvp', 'full_mvp'):
            operator = base.linear_ops[0] if operation == 'graph_mvp' else base
            rhs = torch.from_numpy(
                problem['graph_rhs'][:, :width] if operation == 'graph_mvp'
                else problem['full_rhs'][:, :width]
            )
            result = product_measurement(lambda x: operator @ x, rhs)
        elif operation == 'graph_backward':
            rhs = torch.from_numpy(problem['graph_rhs'][:, :width])
            ((kernel.graph_kernel.full_graph_operator() @ rhs) * rhs).sum().backward()
            kernel.zero_grad()
            tick = time.perf_counter()
            out = base.linear_ops[0] @ rhs
            loss = (out * rhs).sum()
            loss.backward()
            result = dict(
                elapsed_seconds=time.perf_counter() - tick,
                raw_heat_gradient=kernel.graph_kernel.raw_heat_time.grad.item(), warmup_count=1,
            )
        elif operation == 'entries':
            result = entry_measurement(
                base.linear_ops[0], spec.get('entry_count', 256), spec.get('unique_columns', 50)
            )
        elif operation in ('mll', 'train3'):
            result = mll_measurement(model, train, values, seed, updates=3 if operation == 'train3' else 0)
            if operation == 'train3' and case.n <= 400:
                training_stats = copy.deepcopy(stats)
                training_solver = copy.deepcopy(solver)
                with (
                    instrument_operator(record_widths=True) as post_stats,
                    solver_diagnostics() as post_solver,
                ):
                    tick = time.perf_counter()
                    moments = predict_marginals(model, targets, chunk_size=64)
                    row['endpoint_diagnostic_seconds'] = time.perf_counter() - tick
                result.update(
                    mean=moments.mean.tolist(), latent_variance=moments.latent_variance.tolist(),
                    observation_variance=moments.observation_variance.tolist(),
                )
                row.update(endpoint_operator=post_stats, endpoint_solver=post_solver)
        elif operation == 'profile':
            result = profile_mll(model, train, values, seed)
        elif operation in ('mean', 'marginal', 'mean_warm', 'marginal_warm'):
            mean_only = operation.startswith('mean')
            if operation.endswith('_warm'):
                tick = time.perf_counter()
                cold = predict_marginals(model, targets, mean_only=mean_only, chunk_size=64)
                row['cold_preparation_seconds'] = time.perf_counter() - tick
                # Warm counters describe only the cached operation, separately.
                row['cold_preparation_operator'] = dict(stats)
                row['cold_preparation_solver'] = dict(solver)
                with (
                    instrument_operator(record_widths=True) as warm_stats,
                    solver_diagnostics() as warm_solver,
                ):
                    tick = time.perf_counter()
                    moments = warm_prediction(model, targets, mean_only=mean_only)
                    elapsed = time.perf_counter() - tick
                row.update(warm_operator=warm_stats, warm_solver=warm_solver)
            else:
                tick = time.perf_counter()
                moments = predict_marginals(model, targets, mean_only=mean_only, chunk_size=64)
                elapsed = time.perf_counter() - tick
            result = dict(
                elapsed_seconds=elapsed, mean=moments.mean.tolist(), variance_returned=not mean_only
            )
            if mean_only:
                assert moments.latent_variance is None and moments.observation_variance is None
            else:
                result.update(
                    latent_variance=moments.latent_variance.tolist(),
                    observation_variance=moments.observation_variance.tolist(),
                    minimum_latent_variance=moments.latent_variance.min().item(),
                    minimum_observation_variance=moments.observation_variance.min().item(),
                )
        elif operation == 'solve':
            tick = time.perf_counter()
            with torch.no_grad():
                A = kernel(train) + DiagLinearOperator(model.likelihood.noise.expand(case.m))
                rhs = torch.from_numpy(problem['solve_rhs'])
                x = A.solve(rhs)
                fresh = kernel(train) + DiagLinearOperator(model.likelihood.noise.expand(case.m))
                residual = relative_residual(fresh, x, rhs)
            result = dict(
                elapsed_seconds=time.perf_counter() - tick,
                relative_residual=residual.tolist(), solutions=x.tolist(),
            )
        else:
            raise ValueError('unknown operation')
        row.update(
            result=result, elapsed_seconds=result['elapsed_seconds'], operator=stats,
            solver=solver, direct=direct, actions=actions,
        )
        if operation == 'train3' and case.n <= 400:
            row.update(operator=training_stats, solver=training_solver)

    # Path assertions and accounting do not extend the operation timing interval.
    if operation in ('mll', 'train3'):
        if mode['inference'] == 'iterative' and not (solver['cg_calls'] and solver['slq_calls']):
            raise AssertionError('iterative MLL did not use CG/SLQ')
        if mode['inference'] == 'direct' and (solver['cg_calls'] or solver['slq_calls']):
            raise AssertionError('A used iterative inference')
    if (
        mode['backend'] == 'sparse_chebyshev' and operation != 'build'
        and stats['operator_matmul_calls'] == 0
    ):
        raise AssertionError('sparse operation made no Chebyshev call')
    row['rss_peak_mib'] = peak_rss_mib()
    row['memory_status'] = 'measured' if row['rss_peak_mib'] is not None else 'unsupported'
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            result = run(spec)
        except Exception as error:
            result = {
                **spec, 'status': 'error', 'error_type': type(error).__name__, 'error': str(error),
                'traceback': traceback.format_exc(), 'rss_peak_mib': peak_rss_mib(),
            }
    result['warnings'] = [dict(category=x.category.__name__, message=str(x.message)) for x in caught]
    result['numerical_warning_count'] = sum(
        x['category'] == 'NumericalWarning' for x in result['warnings']
    )
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
