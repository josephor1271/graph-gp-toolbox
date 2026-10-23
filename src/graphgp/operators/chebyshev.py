"""Matrix-free Chebyshev polynomials of fixed, symmetric sparse matrices."""

import math

import torch
from linear_operator.operators import LinearOperator
import scipy as sp

## convert scipy csr to torch csr
def scipy_csr_to_torch_csr(A_scipy, *, dtype=None, device=None, copy=False):
    """Convert a SciPy sparse matrix/array to a PyTorch sparse CSR tensor.

    Parameters
    ----------
    A_scipy:
        Any SciPy sparse matrix or sparse array. Non-CSR inputs are converted
        with `.tocsr()` before building the PyTorch tensor.
    dtype:
        Optional PyTorch dtype for the sparse values. If omitted, the dtype is
        inferred from `A_scipy.data`.
    device:
        Optional PyTorch device for the returned sparse CSR tensor.
    copy:
        Whether to force a SciPy-side copy before tensor construction. The
        function still copies if canonicalization is needed.
    """

    if not sp.sparse.issparse(A_scipy):
        raise TypeError("A_scipy must be a SciPy sparse matrix or sparse array")

    A_csr = A_scipy.tocsr(copy=copy)

    if not A_csr.has_canonical_format:
        A_csr = A_csr.copy()
        A_csr.sum_duplicates()
        A_csr.sort_indices()

    crow_indices = torch.tensor(A_csr.indptr, dtype=torch.int64, device=device)
    col_indices = torch.tensor(A_csr.indices, dtype=torch.int64, device=device)
    values = torch.tensor(A_csr.data, device=device)

    # NumPy empty arrays can carry zero strides; Torch 2.11's checked CSR
    # constructor requires stride-one index/value storage even for nnz=0.
    if col_indices.numel() == 0:
        col_indices = torch.empty(0, dtype=torch.int64, device=device)
        values = torch.empty(0, dtype=values.dtype, device=device)

    if dtype is not None:
        values = values.to(dtype=dtype)

    return torch.sparse_csr_tensor(
        crow_indices,
        col_indices,
        values,
        size=A_csr.shape,
        device=device,
        check_invariants=True,
    )


class ChebyshevFunctionLinearOperator(LinearOperator):
    """Represent ``c[0]/2 I + sum(c[k] T_k((L-center I)/radius))``.

    ``L`` must be a fixed, real, symmetric sparse matrix. Symmetry is a caller
    contract: no dense symmetry check is performed. The raw matrix is stored
    in CSR format; affine spectral scaling happens inside sparse products.
    Coefficients determine dtype/device and retain their autograd connection.

    Only the graph is unbatched. RHS tensors may have shape ``(n,)``, ``(n,m)``
    or ``(...,n,m)``. Forward recurrence workspace is O(n*m) per RHS batch.
    The inherited autograd derivative saves recurrence states during backward.
    Degenerate spectral intervals are rejected, including an all-zero graph
    with automatically inferred bounds (supply a positive upper bound instead).
    """

    def __init__(self, L, cheb_coeffs, *, lambda_min, lambda_max):
        cheb_coeffs = torch.as_tensor(cheb_coeffs)
        if cheb_coeffs.is_complex():
            raise TypeError("cheb_coeffs must be real")
        if not cheb_coeffs.is_floating_point():
            cheb_coeffs = cheb_coeffs.to(torch.get_default_dtype())
        if cheb_coeffs.ndim != 1 or cheb_coeffs.numel() == 0:
            raise ValueError("cheb_coeffs must be a nonempty 1D tensor")

        if sp.sparse.issparse(L):
            if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] == 0:
                raise ValueError("L must be a nonempty square matrix")
            if L.dtype.kind == "c":
                raise TypeError("L must be real")
            L = scipy_csr_to_torch_csr(L, dtype=cheb_coeffs.dtype, device=cheb_coeffs.device)
        elif not torch.is_tensor(L) or L.layout != torch.sparse_csr:
            raise TypeError("L must be a sparse CSR tensor or SciPy sparse matrix")
        if L.is_complex():
            raise TypeError("L must be real")
        if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] == 0:
            raise ValueError("L must be a nonempty square matrix")
        if L.requires_grad:
            raise ValueError("L must be fixed (requires_grad=False)")
        L = L.to(dtype=cheb_coeffs.dtype, device=cheb_coeffs.device)
        lambda_min, lambda_max = float(lambda_min), float(lambda_max)
        if not (math.isfinite(lambda_min) and math.isfinite(lambda_max)
                and lambda_min < lambda_max):
            raise ValueError("spectral bounds must be finite with lambda_min < lambda_max")

        # The representation tree must reconstruct exactly the tensors used in
        # _matmul, including any dtype/device canonicalization.
        super().__init__(L, cheb_coeffs, lambda_min=lambda_min, lambda_max=lambda_max)
        self.L = L
        self.cheb_coeffs = cheb_coeffs
        self.n = L.shape[0]
        self.center = lambda_min / 2 + lambda_max / 2
        self.radius = lambda_max / 2 - lambda_min / 2

    def _scaled_laplacian_matmul(self, rhs):
        return (torch.sparse.mm(self.L, rhs) - self.center * rhs) / self.radius

    def _matmul(self, rhs):
        vector = rhs.ndim == 1
        if vector:
            rhs = rhs.unsqueeze(-1)
        if rhs.ndim < 2 or rhs.shape[-2] != self.n:
            raise ValueError(f"rhs must have shape (..., {self.n}, m) or ({self.n},)")
        if rhs.dtype != self.L.dtype or rhs.device != self.L.device:
            raise ValueError("rhs must share the operator's dtype and device")

        # Merge independent RHS batches/columns for the 2-D sparse mm API.
        original_shape = rhs.shape
        flat = rhs.movedim(-2, 0).reshape(self.n, -1)
        coeffs = self.cheb_coeffs
        previous = flat
        out = 0.5 * coeffs[0] * previous
        if coeffs.numel() > 1:
            current = self._scaled_laplacian_matmul(previous)
            out = out + coeffs[1] * current
            for k in range(2, coeffs.numel()):
                following = 2 * self._scaled_laplacian_matmul(current) - previous
                out = out + coeffs[k] * following
                previous, current = current, following
        result = out.reshape(self.n, *original_shape[:-2], original_shape[-1]).movedim(0, -2)
        return result.squeeze(-1) if vector else result

    def _size(self):
        return torch.Size((self.n, self.n))

    def _transpose_nonbatch(self):
        return self

    def _get_indices(self, row_index, col_index, *batch_indices):
        """Exact selected entries with one basis-vector MVP per entry.

        linear_operator 0.6.1's default attempts to batch/unsqueeze the CSR
        representation, which PyTorch does not support. This correctness
        fallback avoids both that path and an n-by-n identity allocation.
        Diagonal extraction remains O(n * degree * nnz(L)); it is unoptimized.
        """
        if batch_indices:
            raise ValueError("batched graph indexing is unsupported")
        rows, cols = torch.broadcast_tensors(row_index, col_index)
        entries = []
        for row, col in zip(rows.reshape(-1), cols.reshape(-1)):
            basis_vector = self.cheb_coeffs.new_zeros(self.n, 1)
            basis_vector[col, 0] = 1
            entries.append((self @ basis_vector)[row, 0])
        if not entries:
            return self.cheb_coeffs.new_empty(rows.shape)
        return torch.stack(entries).reshape(rows.shape)
