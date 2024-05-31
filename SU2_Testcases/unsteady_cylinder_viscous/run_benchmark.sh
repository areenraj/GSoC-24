#!bin/bash

rm *.csv *.vtu *.dat
time mpirun -np 8 SU2_CFD fem_unst_cylinder.cfg

