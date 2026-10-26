"""Model-independent bounds and Laplacian rules for the fixed study domain."""
import numpy as np
import scipy.sparse

HEAT_RANGE = (0.05, 4.0)
NOISE_RANGE = (0.005, 0.5)
AMPLITUDE_RANGE = (0.1, 4.0)
LENGTHSCALE_RANGE = (0.08, 0.8)


def validate_study_laplacian(L) -> scipy.sparse.csr_matrix:
    """Validate a fixed SciPy Laplacian with maximum weighted degree <= 1.

    Return float64 CSR, retaining the degree-scaled combinatorial convention
    and its analytically enclosing spectral interval [0, 2].
    """
    if not scipy.sparse.issparse(L):
        raise TypeError("expected a SciPy sparse Laplacian")
    if L.ndim != 2 or not L.shape[0] or L.shape[0] != L.shape[1]:
        raise ValueError("Laplacian must be nonempty and square")
    if np.iscomplexobj(L.data):
        raise ValueError("Laplacian must be real")
    L = L.astype(np.float64).tocsr()
    off = L - scipy.sparse.diags(L.diagonal())
    asymmetry = L-L.T
    if (not np.isfinite(L.data).all() or
        (asymmetry.nnz and np.max(np.abs(asymmetry.data)) > 1e-12) or
        np.max(np.abs(np.asarray(L.sum(axis=1)))) > 1e-12 or
        np.any(off.data > 1e-12) or np.any(L.diagonal() < 0) or
        np.max(L.diagonal()) > 1+1e-12):
        raise ValueError("expected symmetric Laplacian with maximum weighted degree <= 1")
    return L
