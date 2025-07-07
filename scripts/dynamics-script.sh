#!/bin/bash 

# Set number of threads (adjust to your CPU)
export OMP_NUM_THREADS=4

### Numerical parameters for simulation ###
export T0=0.0 # initial time
export T1=10.0 # final time
export DT=0.1 # time step


### Bath parameters ###
export TEMPERATURE=300 # temperature [K]
export GAMMA=10.0 # spectral density width
export W0=220.0 # spectral density peak location
export ALPHA=0.02 # coupling strength


### TLS parameters ###
export EPSILON=1000 # energy splitting of the TLS
export DELTA=20 # driving amplitude
export OMEGA=0 # driving frequency
export CONV=0.188 # conversion factor

### RC parameters
export M=2 # dimension of RC Hilbert space

# Run py files
python ../src/workflow/dynamics.py
