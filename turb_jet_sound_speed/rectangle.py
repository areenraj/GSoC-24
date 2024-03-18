#!/usr/bin/python

## \file square.py
#  \brief Python script for creating a .su2 mesh of a simple square domain.
#  \author Thomas D. Economon
#  \version 1.0.

# Import the option parser and parse input options
import math
from optparse import OptionParser

parser=OptionParser()
parser.add_option("-f", "--file", dest="filename", default="square.su2",
                  help="write mesh to FILE", metavar="FILE")
parser.add_option("-n", "--nNode", dest="nNode", default=3,
                  help="use this NNODE in x direction", metavar="NNODE")
parser.add_option("-m", "--mNode", dest="mNode", default=3,
                  help="use this MNODE in x direction", metavar="MNODE")
parser.add_option("-i", "--inlet", dest="inlet", default=0,
                  help="use this to specify the max y coordinate of inlet", metavar="INLET")
parser.add_option("-l", "--l", dest="l", default=1,
                  help="use this to specify the max x coordinate", metavar="l")
parser.add_option("-b", "--b", dest="b", default=1,
                  help="use this to specify the max y coordinate", metavar="b")
parser.add_option("-c", "--c", dest="c", default=0,
                  help="use this to specify the percentage of cells in x-direction in left side", metavar="c")
parser.add_option("-g", "--g", dest="g", default=0,
                  help="use this to specify the percentage length to accommodate the cells in left side", metavar="g")
parser.add_option("-k", "--k", dest="k", default=0,
                  help="use this to specify the percentage of cells in x-direction in right side", metavar="k")
parser.add_option("-t", "--t", dest="t", default=0,
                  help="use this to specify the percentage length to accommodate the cells in left side", metavar="t")
(options, args)=parser.parse_args()

# Set the VTK type for the interior elements and the boundary elements
KindElem  = 5 # Triangle
KindBound = 3 # Line

# Store the number of nodes and open the output mesh file
nNode     = int(options.nNode)
mNode     = int(options.mNode)
l = float(options.l)
b = float(options.b)
inl = float(options.inlet)
c  = float(options.c)
g = float(options.g)
k  = float(options.k)
t = float(options.t)

Mesh_File = open(options.filename,"w")

# Write the dimension of the problem and the number of interior elements
Mesh_File.write( "%\n" )
Mesh_File.write( "% Problem dimension\n" )
Mesh_File.write( "%\n" )
Mesh_File.write( "NDIME= 2\n" )
Mesh_File.write( "%\n" )
Mesh_File.write( "% Inner element connectivity\n" )
Mesh_File.write( "%\n" )
Mesh_File.write( "NELEM= %s\n" % (2*(nNode-1)*(mNode-1)))


nNode1 = math.ceil(c*nNode)
nNode3 = math.ceil(k*nNode)
nNode2 = nNode - nNode1 - nNode3

# Write the element connectivity
iElem = 0
for jNode in range(mNode-1):
    for iNode in range(nNode-1):
        iPoint = jNode*nNode + iNode
        jPoint = jNode*nNode + iNode + 1
        kPoint = (jNode + 1)*nNode + iNode
        Mesh_File.write( "%s \t %s \t %s \t %s \t %s\n" % (KindElem, iPoint, jPoint, kPoint, iElem) )
        iElem = iElem + 1
        iPoint = jNode*nNode + (iNode + 1)
        jPoint = (jNode + 1)*nNode + (iNode + 1)
        kPoint = (jNode + 1)*nNode + iNode
        Mesh_File.write( "%s \t %s \t %s \t %s \t %s\n" % (KindElem, iPoint, jPoint, kPoint, iElem) )
        iElem = iElem + 1

# Compute the number of nodes and write the node coordinates
nPoint = (nNode)*(mNode)
Mesh_File.write( "%\n" )
Mesh_File.write( "% Node coordinates\n" )
Mesh_File.write( "%\n" )
Mesh_File.write( "NPOIN= %s\n" % ((nNode)*(mNode)) )
iPoint = 0

for jNode in range(mNode):

    for iNode in range(nNode1-1):
        Mesh_File.write( "%15.14f \t %15.14f \t %s\n" % ( (g*l)*(float(iNode)/float(nNode1-1)), (b)*(float(jNode)/float(mNode-1)), iPoint ))
        iPoint = iPoint + 1

    for iNode in range(nNode2+1):
        Mesh_File.write( "%15.14f \t %15.14f \t %s\n" % ( g*l + ((1-g-t)*l)*(float(iNode)/float(nNode2+1)), (b)*(float(jNode)/float(mNode-1)), iPoint ))
        iPoint = iPoint + 1

    for iNode in range(nNode3):
        Mesh_File.write( "%15.14f \t %15.14f \t %s\n" % ( (1-t)*l + (t*l)*(float(iNode)/float(nNode3-1)), (b)*(float(jNode)/float(mNode-1)), iPoint ))
        iPoint = iPoint + 1

# Write the header information for the boundary markers
Mesh_File.write( "%\n" )
Mesh_File.write( "% Boundary elements\n" )
Mesh_File.write( "%\n" )
Mesh_File.write( "NMARK= 5\n" )

#Defining number of nodes for inlet
inElem = math.ceil(((mNode-1)/b)*inl)
inNode = inElem + 1            

# Write the boundary information for each marker
Mesh_File.write( "MARKER_TAG= lower\n" )
Mesh_File.write( "MARKER_ELEMS= %s\n" % (nNode-1))
for iNode in range(nNode-1):
    Mesh_File.write( "%s \t %s \t %s\n" % (KindBound, iNode, iNode + 1) )
Mesh_File.write( "MARKER_TAG= right\n" )
Mesh_File.write( "MARKER_ELEMS= %s\n" % (mNode-1))
for jNode in range(mNode-1):
    Mesh_File.write( "%s \t %s \t %s\n" % (KindBound, jNode*nNode + (nNode - 1),  (jNode + 1)*nNode + (nNode - 1) ) )
Mesh_File.write( "MARKER_TAG= upper\n" )
Mesh_File.write( "MARKER_ELEMS= %s\n" % (nNode-1))
for iNode in range(nNode-1):
    Mesh_File.write( "%s \t %s \t %s\n" % (KindBound, (nNode*mNode - 1) - iNode, (nNode*mNode - 1) - (iNode + 1)) )
Mesh_File.write( "MARKER_TAG= left\n" )
Mesh_File.write( "MARKER_ELEMS= %s\n" % (mNode-inNode))
for jNode in range(mNode-2, inNode-2, -1):
    Mesh_File.write( "%s \t %s \t %s\n" % (KindBound, (jNode + 1)*nNode, jNode*nNode ) )
Mesh_File.write( "MARKER_TAG= inlet\n" )
Mesh_File.write( "MARKER_ELEMS= %s\n" % (inElem))
for jNode in range(inNode-2, -1, -1):
    Mesh_File.write( "%s \t %s \t %s\n" % (KindBound, (jNode + 1)*nNode, jNode*nNode ) )


# Close the mesh file and exit
Mesh_File.close()
