import numpy as np
import qutip as qt

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



def N_residual_bath(w, beta_ph):
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

def J_residual_bath(w, gam):
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



def Lio_residual_bath(vals, vecs, H_ES, A, gamma, beta_ph):
    """ Calculate the Liouville superoperator for the reaction coordinate master equation 
    
    Parameters
    ----------
    vals : list or array-like
        Eigenvalues of the system Hamiltonian.
    vecs : list or array-like
        Eigenvectors of the system Hamiltonian.
    H_ES : qutip.Qobj
        System Hamiltonian.
    A : qutip.Qobj
        Coupling operator between the system and the residual bath.
    gamma : float
        Coupling strength between the reaction coordinate and the residual bath.
    beta_ph : float
        Inverse temperature of the residual bath (1/kT).
        
    Returns
    -------
    L_total : qutip.Qobj
        The total Liouville superoperator for the reaction coordinate master equation.
    """
    # TODO validate inputs and raise ValueError if not correct

    # initialize rate operators
    R1 = 0
    R2 = 0

    #----- loop over all combinations of eigenvalues/eigenvectors -----#
    for m in range(len(vals)):
        for n in range(len(vals)):
            
            # calculate difference in eigenvalues
            lambda_mn = vals[m] - vals[n]

            # calculate the outer product of eigenvectors
            proj_mn = vecs[m] * vecs[n].dag()

            # calculate matrix element of coupling operator A
            A_mn = vecs[m].dag() * A * vecs[n]

            #----- add to the rate operators depending on the value of lambda_mn -----#

            if lambda_mn > 0:
                # call functions to calculate J and N
                J = J_residual_bath(lambda_mn, gamma)
                N = N_residual_bath(lambda_mn, beta_ph)

                R1 += np.pi * A_mn * J + N * proj_mn
                R2 += np.pi * A_mn * J * (1 + N) * proj_mn

            elif lambda_mn < 0:
                # call functions to calculate J and N (with negative lambda_mn)
                # Note: J and N are not defined for negative frequencies so we convert negative lambda_mn to positive
                J = J_residual_bath(-lambda_mn, gamma)
                N = N_residual_bath(-lambda_mn, beta_ph)

                R1 += np.pi * A_mn * J * (1 + N) * proj_mn
                R2 += np.pi * A_mn * J * N *  proj_mn

            elif lambda_mn == 0:
                # TODO possible a factor of 1/2 missing from here
                val = np.pi * A_mn * (gamma / beta_ph) * proj_mn
                R1 += val
                R2 += val

            else:
                raise ValueError(f"Unexpected value for lambda_mn: {lambda_mn}")
            
    #----- calculate the Liouville superoperator -----# 
    
    # coherent term
    L_coherent = -1j*(qt.spre(H_ES) - qt.spost(H_ES))
    
    # dissipative term
    L_dissipative = qt.spre(A*R1) - qt.sprepost(R1, A) - qt.sprepost(A, R2) + qt.spost(R2*A)

    # total Liouville superoperator
    L_total = L_coherent - L_dissipative

    return L_total