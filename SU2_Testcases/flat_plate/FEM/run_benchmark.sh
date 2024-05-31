!bin/bash

rm *.csv *.vtu *.dat
time mpirun -np 8 SU2_CFD lam_flatplate_reg.cfg
