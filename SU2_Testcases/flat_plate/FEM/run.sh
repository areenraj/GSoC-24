#!bin/bash

rm *.csv *.vtu *.dat
mpirun -np 8 SU2_CFD lam_flatplate_reg.cfg
