#!/bin/bash 

# Set number of threads (adjust to your CPU)
export OMP_NUM_THREADS=4

### Numerical parameters for simulation ###
export T0=0.0 # initial time
export T1=100.0 # final time
export DT=0.1 # time step


### Bath parameters ###
export BETA=1 # inverse temperature
export GAMMA=10.0 # spectral densit width
export W0=25.0 # spectral density peak location
export WC=5 # spectral density cutoff
export ALPHA_LIST="0.02" # coupling strength list, can be a single value or multiple values separated by spaces

### TLS parameters ###
# export TP_LIST="1 2 3" # list of protocol times, can be a single value or multiple values separated by spaces
export EPSILON=25.0 # energy splitting of the TLS
export DELTA=0.02 # driving of TLS

### RC parameters
export M=10 # dimensio of RC Hilbert space

# Run py files
python ../src/workflow/dynamics.py
