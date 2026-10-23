"""A conservative real-arithmetic interpolation certificate for heat kernels."""
import math


def heat_interpolation_certificate(degree=32, max_heat_time=4.0, lambda_max=2.0):
    """Bound the error when degree+1 Chebyshev nodes interpolate exp(-tau*x).

    On [-1,1], |f^(m)| <= z^m, z=tau*lambda_max/2, m=degree+1.
    The monic Chebyshev nodal product has norm 2^(1-m), so the Lagrange
    remainder is <= 2*(z/2)^m/m!. This covers the ENTIRE declared parameter
    and spectral intervals in exact arithmetic. It is not an interval-arithmetic
    certificate of the floating-point implementation. A separate conservative
    roundoff allowance and numerical tests provide a practical margin.
    """
    if not isinstance(degree, int) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    if not all(math.isfinite(x) and x > 0 for x in (max_heat_time, lambda_max)):
        raise ValueError("bounds must be finite and positive")
    m = degree + 1
    z = max_heat_time * lambda_max / 2
    error = math.exp(math.log(2) + m * math.log(z / 2) - math.lgamma(m + 1))
    floor = math.exp(-max_heat_time * lambda_max)
    allowance = 100 * m * m * 2.220446049250313e-16
    if error + allowance >= floor:
        raise ValueError("degree/range has insufficient positivity margin")
    return dict(degree=degree, num_nodes=m, lambda_min=0.0, lambda_max=lambda_max,
                max_heat_time=max_heat_time, exact_spectral_floor=floor,
                interpolation_error_bound=error, roundoff_allowance=allowance,
                positivity_margin=floor-error-allowance)
