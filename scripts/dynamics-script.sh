#!/bin/bash 

# Set number of threads (adjust to your CPU)
export OMP_NUM_THREADS=4

### Numerical parameters for simulation ###
export T0=0.0 # initial time
export T1=10.0 # final time
export DT=0.1 # time step


### Bath parameters ###
export UNITS="eV" # units for energy
export TEMPERATURE=300 # temperature [K]
export GAMMA=0.001 # spectral density width
export W0=0.05 # spectral density peak location
export ALPHA=0.1 # coupling strength


### TLS parameters ###
export EPSILON=2 # energy splitting of the TLS
export DELTA=0.1 # driving amplitude
export OMEGA=0 # driving frequency
export CONV=1519.3 # conversion factor

### RC parameters
export M=10 # dimension of RC Hilbert space

# Run py files
python ../src/workflow/dynamics.py
