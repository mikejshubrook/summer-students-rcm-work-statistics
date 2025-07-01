import numpy as np

def symmetrize_function(x, f):
    """
    Extend function values f(x) and corresponding x values symmetrically about x=0
    so that f(-x) = f(x) (even symmetry, no complex conjugation).

    Used so that we only need to calculate half of the work/photon characteristic functions.

    Parameters
    ----------
    x : array-like
        1D array of x values, assumed to be >= 0 and evenly spaced starting at 0.
    f : array-like
        1D array of f(x) values corresponding to the input x.

    Returns
    -------
    x_full : ndarray
        Array of x values extended to negative domain in symmetric fashion.
    f_full : ndarray
        Array of function values f(x) extended such that f(-x) = f(x).

    Raises
    ------
    ValueError
        If x and f have different lengths or are not 1D.
    """
    x = np.asarray(x)
    f = np.asarray(f)

    if x.ndim != 1 or f.ndim != 1:
        raise ValueError("Both x and f must be 1D arrays.")
    if len(x) != len(f):
        raise ValueError("x and f must have the same length.")
    if x[0] != 0:
        raise ValueError("x must start at 0 for symmetric extension.")

    x_neg = -x[1:][::-1]   # negative part, excluding x=0
    f_neg = f[1:][::-1]    # mirror values (no conjugate)

    x_full = np.concatenate([x_neg, x])
    f_full = np.concatenate([f_neg, f])

    return x_full, f_full
