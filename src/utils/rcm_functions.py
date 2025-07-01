import numpy as np

def J_UD(w, alpha, Gamma, w0):
    """
    Compute the underdamped spectral density J(w) of a bosonic environment.

    Parameters
    ----------
    w : float or array-like
        Frequency or array of frequencies at which to evaluate the spectral density.
    alpha : float
        TLS-full bath coupling strength. Must be real and non-negative.
    Gamma : float
        Spectral density width. Must be real and non-negative.
    w0 : float
        Peak frequency of the spectral density. Must be real and non-negative.

    Returns
    -------
    J : float or ndarray
        The spectral density evaluated at frequency(ies) w.

    Raises
    ------
    ValueError
        If any of `alpha`, `Gamma`, or `w0` is negative or complex.
    """
    # Convert to float and validate
    try:
        alpha = float(alpha)
        Gamma = float(Gamma)
        w0 = float(w0)
    except (ValueError, TypeError):
        raise ValueError("All inputs must be real numbers.")

    for name, val in zip(["alpha", "Gamma", "w0"], [alpha, Gamma, w0]):
        if not np.isreal(val) or val < 0:
            raise ValueError(f"{name} must be a real, non-negative number. Got: {val}")

    # Convert frequency input to ndarray for vectorized operations
    w = np.asarray(w, dtype=float)

    # Compute numerator and denominator
    numerator = alpha * Gamma * (w0**2) * w
    denominator = (w0**2 - w**2)**2 + (Gamma * w)**2

    return numerator / denominator


def perform_reaction_coordinate_mapping(alpha, Gamma, w0):
    """
    Compute the parameters after performing the reaction coordinate mapping (RCM), 
    given the unmapped parameters for a Drude-Lorentz underdamped spectral density
    of the full bath.

    Parameters
    ----------
    alpha : float
        Coupling strength between TLS and full bath (must be real and non-negative).
    Gamma : float
        Spectral density width (must be real and non-negative).
    w0 : float
        Peak frequency of the spectral density (must be real and non-negative).

    Returns
    -------
    Omega : float
        Energy spacing of the RC (equal to w0).
    llambda : float
        Coupling strength between the TLS and RC.
    gam : float
        Coupling strength between RC and residual bath.

    Raises
    ------
    ValueError
        If any input is negative, complex, or not a real number.
    """
    # Convert to float and validate
    try:
        alpha = float(alpha)
        Gamma = float(Gamma)
        w0 = float(w0)
    except (ValueError, TypeError):
        raise ValueError("All inputs must be real numbers.")

    for name, val in zip(["alpha", "Gamma", "w0"], [alpha, Gamma, w0]):
        if not np.isreal(val) or val < 0:
            raise ValueError(f"{name} must be a real, non-negative number. Got: {val}")

    Omega = w0 # Energy spacing of the RC is equal to w0
    llambda = np.sqrt(np.pi * alpha * w0 / 2) # Coupling strength between TLS and RC
    gam = Gamma / (2 * np.pi * w0) # Coupling strength between RC and residual bath

    return Omega, llambda, gam



def N_ph(w, beta_ph):
    """
    Bose-Einstein thermal occupation number at frequency w and inverse temperature beta_ph.

    Parameters
    ----------
    w : float or array-like
        Frequency (or frequencies) of the bosonic mode(s). Must be real and non-negative.
    beta_ph : float
        Inverse temperature (1/kT). Must be real and non-negative.

    Returns
    -------
    n : float or ndarray
        Thermal occupation number(s) at each frequency.

    Raises
    ------
    ValueError
        If any input is complex or negative.
    """
    # Convert inputs to float or array
    w = np.asarray(w, dtype=float)

    try:
        beta_ph = float(beta_ph)
    except (ValueError, TypeError):
        raise ValueError("beta_ph must be a real number.")

    if beta_ph < 0 or not np.isreal(beta_ph):
        raise ValueError(f"beta_ph must be real and non-negative. Got: {beta_ph}")
    if np.any(w < 0) or not np.all(np.isreal(w)):
        raise ValueError("w must be real and non-negative.")

    with np.errstate(divide='ignore', invalid='ignore'):
        result = 1 / (np.exp(beta_ph * w) - 1)
        result = np.where(w == 0, 0.0, result)  # Handle w=0 gracefully

    return result

def J_ph(w, gam):
    """
    Ohmic spectral density without cutoff: J(w) = gamma * w

    Parameters
    ----------
    w : float or array-like
        Frequency or frequencies at which to evaluate J(w). Must be real and non-negative.
    gam : float
        Ohmic damping coefficient. Must be real and non-negative.

    Returns
    -------
    J : float or ndarray
        Spectral density values at each frequency.

    Raises
    ------
    ValueError
        If any input is complex or negative.
    """
    w = np.asarray(w, dtype=float)

    try:
        gam = float(gam)
    except (ValueError, TypeError):
        raise ValueError("gam must be a real number.")

    if gam < 0 or not np.isreal(gam):
        raise ValueError(f"gam must be real and non-negative. Got: {gam}")
    if np.any(w < 0) or not np.all(np.isreal(w)):
        raise ValueError("w must be real and non-negative.")

    return gam * w
