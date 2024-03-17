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

## Third Task - Completed 14-04-24

- Completed the Python Wrapper Case
- Ran it for 1350 iterations
- Extracted the temporal evolution of the temperature profile

For runnin the case with the wrapper in parallel
> bash run.sh

For cleaning the directory
> bash clean.sh


