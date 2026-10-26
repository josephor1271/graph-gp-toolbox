"""Scoped measurement of existing Chebyshev methods; no numerical changes.

Counts describe operator calls and polynomial recurrence applications, not
scalar sparse matrix-vector products: each call may carry multiple RHS columns.
The optional histogram retains the original final-dimension width convention
(and does not multiply independent batch sizes). Method timings overlap caller
wall time and are not a complete end-to-end cost breakdown. Wrappers are active
only inside the context and restore original methods even after exceptions.
"""
from contextlib import contextmanager
import time
from unittest.mock import patch

import torch

from graphgp.operators.chebyshev import ChebyshevFunctionLinearOperator


@contextmanager
def instrument_operator(*, record_widths=False):
    stats = dict(operator_matmul_calls=0, estimated_sparse_l_mvps=0,
                 get_indices_calls=0, get_indices_entries=0,
                 matmul_seconds=0.0, get_indices_seconds=0.0)
    if record_widths:
        stats['rhs_width_counts'] = {}
    original_mm = ChebyshevFunctionLinearOperator._matmul
    original_get = ChebyshevFunctionLinearOperator._get_indices
    def mm(op, rhs):
        start = time.perf_counter()
        stats["operator_matmul_calls"] += 1
        if record_widths:
            width = str(rhs.shape[-1] if rhs.ndim > 1 else 1)
            stats['rhs_width_counts'][width] = stats['rhs_width_counts'].get(width, 0) + 1
        stats["estimated_sparse_l_mvps"] += op.cheb_coeffs.numel() - 1
        try:
            return original_mm(op, rhs)
        finally:
            stats["matmul_seconds"] += time.perf_counter() - start
    def get(op, row, col, *batch):
        start = time.perf_counter()
        stats["get_indices_calls"] += 1
        stats["get_indices_entries"] += torch.broadcast_tensors(row, col)[0].numel()
        try:
            return original_get(op, row, col, *batch)
        finally:
            stats["get_indices_seconds"] += time.perf_counter() - start
    with patch.object(ChebyshevFunctionLinearOperator, "_matmul", mm), \
         patch.object(ChebyshevFunctionLinearOperator, "_get_indices", get):
        yield stats


