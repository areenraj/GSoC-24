# GSOC SU2

## First Task - Completed 12-03-24

- Successfully compiled the code
- Ran the 2D Airfoil simulation and verified the output

## Second Task - Completed 14-04-24

- Used the SST Model with Default Values
- Enabled the Axisymmetric Option
- Modified the square.py script to accommodate a rectangular domain and inlet length

The arguments for the script are as follows

```
- l specifies the length
- b specifies the breadth
- i specifies the inlet length
```

Note: Please make sure that the nodes in both x and y directions are divisible by l and b respectively \

For running the case and cleaning the directly simultaneously please use
> bash run.sh

## Third Task - Completed 17-04-24

- Completed the Python Wrapper Case
- Ran it for 1350 iterations
- Extracted the temporal evolution of the temperature profile

For running the case with the wrapper in parallel
> bash run.sh

For cleaning the directory
> bash clean.sh

## Fourth Task - Completed 17-04-24

- Completed the spatially varying Python Wrapper Case
- Used a steady state CHT simulation
- Used the vertex information at the marker to set a spatial profile

For running the case with the wrapper in parallel
> bash run.sh

For cleaning the directory
> bash clean.sh

## Fifth Task - Complete 17-04-24

- Completed adding a new volume and history output of speed of sound to the compressible solver
- Edited the CFlowCompOutput.cpp file to add and set a new volume output that contains spatial data of speed of soun
- Edited the CFlowCompOutput.cpp file to add and set a new history and screen output that shows the area averaged speed of sound over the entire mesh

For running the case and cleaning the directly simultaneously please use
> bash run.sh


