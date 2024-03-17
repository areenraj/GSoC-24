#!/bin/bash

rm -r *.vtu *.dat *.csv *.vtm flow*
mpirun -n 8 python3 -m mpi4py launch_spatial_CHT_FlatPlate.py --parallel -f spatial_CHT_FlatPlate_Conf.cfg
