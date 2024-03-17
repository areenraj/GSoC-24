#!bin/bash

rm -r vol_solution
rm *.csv *.vtm
mpirun -np 8 SU2_CFD jet.cfg
