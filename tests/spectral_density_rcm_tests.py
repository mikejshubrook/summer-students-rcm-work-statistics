import sys
import os

import numpy as np
from math import isclose

# Add the src folder to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils.rcm_functions import J_UD, perform_reaction_coordinate_mapping, N_ph, J_ph

def test_J_UD():
    # Valid input
    w = np.linspace(0, 10, 100)
    result = J_UD(w, alpha=0.5, Gamma=1.0, w0=2.0)
    assert np.all(result >= 0)

    # Scalar input
    assert J_UD(1.0, 0.5, 1.0, 2.0) > 0

    # Invalid input
    try:
        J_UD(1.0, -0.5, 1.0, 2.0)
        raise AssertionError("Expected ValueError for negative alpha")
    except ValueError:
        pass

def test_perform_reaction_coordinate_mapping():
    Omega, llambda, gam = perform_reaction_coordinate_mapping(0.5, 1.0, 2.0)
    assert isclose(Omega, 2.0)
    assert llambda > 0
    assert gam > 0

    try:
        perform_reaction_coordinate_mapping(-0.5, 1.0, 2.0)
        raise AssertionError("Expected ValueError for negative alpha")
    except ValueError:
        pass

def test_N_ph():
    # Scalar
    assert N_ph(1.0, 2.0) > 0

    # Vector input
    n = N_ph([0.0, 0.5, 1.0], 1.0)
    assert np.all(n >= 0)
    assert np.isclose(n[0], 0.0)

    # Error handling
    try:
        N_ph(-1.0, 1.0)
        raise AssertionError("Expected ValueError for negative w")
    except ValueError:
        pass

    try:
        N_ph(1.0, -2.0)
        raise AssertionError("Expected ValueError for negative beta_ph")
    except ValueError:
        pass

def test_J_ph():
    assert np.isclose(J_ph(1.0, 0.5), 0.5)

    J = J_ph([0.0, 1.0, 2.0], 2.0)
    assert np.allclose(J, [0.0, 2.0, 4.0])

    try:
        J_ph(-1.0, 1.0)
        raise AssertionError("Expected ValueError for negative frequency")
    except ValueError:
        pass

    try:
        J_ph(1.0, -0.1)
        raise AssertionError("Expected ValueError for negative gam")
    except ValueError:
        pass


# ===== MAIN TEST RUNNER =====
if __name__ == "__main__":
    try:
        test_J_UD()
        test_perform_reaction_coordinate_mapping()
        test_N_ph()
        test_J_ph()
        print("✅ All tests passed successfully.")
    except AssertionError as e:
        print("❌ Test failed:", e)
    except Exception as e:
        print("❌ An unexpected error occurred during testing:", e)
