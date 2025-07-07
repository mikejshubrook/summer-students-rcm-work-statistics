import numpy as np

def symmetrize_function(x, f):
    """
    Extend domain to negative values, then symmetrize the function such that f(-x)^* = f(x).

    Parameters
    ----------
    x : array-like
        1D array of x real values, assumed to be >= 0 and evenly spaced starting at 0.
    f : array-like
        1D array of f(x) complex values corresponding to the input x.

    Returns
    -------
    x_full : ndarray
        Array of x real values extended to negative domain in symmetric fashion.
    f_full : ndarray
        Array of function complex values f(x) extended such that f(-x)^* = f(x).

    Raises
    ------
    ValueError
        If x and f have different lengths or are not 1D.
    """

    # Convert inputs to numpy arrays
    x = np.asarray(x) # chi values
    f = np.asarray(f) # characterstic function values

    # Validate inputs
    if x.ndim != 1 or f.ndim != 1:
        raise ValueError("Both x and f must be 1D arrays.")
    if len(x) != len(f):
        raise ValueError("x and f must have the same length.")
    if x[0] != 0:
        raise ValueError("x must start at 0 for symmetric extension.")
    if not np.all(np.isreal(x)):
        raise ValueError("x must contain only real values.")

    # conversion to full domain
    x_neg = -x[1:][::-1]   # mirror values (excluding 0), without complex conjugation
    f_neg = np.conj(f[1:][::-1])    # mirror values (exclusing f(0)) with complex conjugation

    x_full = np.concatenate([x_neg, x]) # concatenate
    f_full = np.concatenate([f_neg, f]) # concatenate

    return x_full, f_full
