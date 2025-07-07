# Standard library imports
import os
import sys

# Third-party imports
import numpy as np
import pandas as pd

# Get the directory of the current script (workflow)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (src)
src_dir = os.path.dirname(current_dir)
# Add src to sys.path
sys.path.append(src_dir)
# Local imports
from utils.rcm_functions import *

# Path handling for reliable file saving/loading
CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_FILE, "../../../"))  # Adjust based on your layout
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "dynamics-files")

#----- Import parameters from BASH -----#
T0  = float(os.environ['T0'])           
T1 = float(os.environ['T1'])
DT = float(os.environ['DT'])

UNITS = os.environ['UNITS']  # Units for energy

TEMPERATURE = float(os.environ['TEMPERATURE'])  # Temperature
GAMMA = float(os.environ['GAMMA'])  # Spectral density width
W0 = float(os.environ['W0'])  # Spectral density peak location
ALPHA = float(os.environ['ALPHA'])  # Coupling strength

EPSILON = float(os.environ['EPSILON'])  # Energy splitting of the TLS
DELTA = float(os.environ['DELTA'])  # Driving amplitude of TLS  
OMEGA = float(os.environ['OMEGA'])  # Driving frequency of TLS
CONV = float(os.environ['CONV'])  # Conversion factor

M = int(os.environ['M'])  # Dimension of RC Hilbert space

#----- Convert parameters (if needed) -----#
delta = DELTA * CONV  # Convert driving amplitude to eV
epsilon = EPSILON * CONV  # Convert energy splitting to eV
omega = OMEGA * CONV  # Convert driving frequency to eV
nu = epsilon-omega

alpha = ALPHA * CONV  # Convert coupling strength to eV
w0 = W0 * CONV  # Convert spectral density peak location to eV

beta_residual_bath = 7.638/TEMPERATURE
tlist = np.arange(T0, T1, DT)  # List of times

#----- Perform reaction coordinate mapping -----#
Omega, llambda, gamma = perform_reaction_coordinate_mapping(alpha, GAMMA, w0)


#----- Create operators and build Hamiltonian(s) -----#
# system operators 
kete = qt.basis(2,0)
ketg = qt.basis(2,1)
ee = kete*kete.dag()
gg = ketg*ketg.dag()
eg = kete*ketg.dag() 
ge = ketg*kete.dag()

# RC ladder operators
a = qt.destroy(M)

# ES interaction term with RB
A = qt.tensor(qt.qeye(2), a.dag() + a)

H_rc = Omega*a.dag()*a
H_RC = qt.tensor(qt.qeye(2), H_rc)
H_I = llambda * qt.tensor(qt.sigmaz(), a.dag()+a)
H_S = qt.tensor(epsilon * qt.sigmaz() + delta * qt.sigmax(), qt.qeye(M))  # TLS Hamiltonian (on ES Hilbert space)
H_ES = H_S + H_RC + H_I


#----- Diagonalize the Hamiltonian -----#
vals_ES, vecs_ES = H_ES.eigenstates()  # for RB and nadd


#----- Create the Liouvillian -----#
L_total = Lio_residual_bath(vals_ES, vecs_ES, 
                            H_ES, A, 
                            gamma, beta_residual_bath)


#----- Define initial state -----#
rho_s_0 = ee
Z = ((-(beta_residual_bath)*(H_rc)).expm()).tr()
rho_RC = (-beta_residual_bath*H_rc).expm()/Z 
rho_0 = qt.tensor(rho_s_0, rho_RC) 

#----- Expectation operators list -----#
exp_ops = [qt.tensor(ee, qt.qeye(M)), # excited state population
              qt.tensor(qt.sigmax(), qt.qeye(M)), # coherence
              qt.tensor(qt.qeye(2), a.dag()*a) # RC population
              ] 

#----- Run the master equation -----#
result = qt.mesolve(L_total, rho_0, tlist, e_ops=exp_ops,
                    options=qt.Options(nsteps = 100000000, 
                                        atol = 1e-7, rtol = 1e-7), 
                                        progress_bar=True)
#----- Extract expectation values -----#
excited_pop =  np.array(result.expect[0]).real
coherence = np.array(result.expect[1]).real
rc_pop = np.array(result.expect[2]).real

#----- Save results -----#

# store results in a dictionary
results = {
    'tlist': tlist.real,
    'excited_pop': excited_pop,
    'coherence': coherence,
    'rc_pop': rc_pop
    }

# Convert dict of resuilts to a pandas DataFrame
df = pd.DataFrame(results)

# Save DataFrame to a CSV file
output_path = os.path.join(DATA_DIR, f"dynamics_{UNITS}_A{ALPHA}_E{EPSILON}_D{DELTA}_O{OMEGA}_T{TEMPERATURE}_G{GAMMA}_w0{W0}_t0{T0}_tf{T1}_dt{DT}_M{M}.csv")

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save CSV
df.to_csv(output_path, index=False)
print(f"Saved dynamics to CSV.")