// Gmsh project created on Mon Jun 24 22:14:37 2024
SetFactory("OpenCASCADE");
//+
Point(1) = {-0.5, 0, 0, 1.0};
//+
Point(2) = {2, 0, 0, 1.0};
//+
Point(3) = {2, 1, 0, 1.0};
//+
Point(4) = {-0.5, 1, 0, 1.0};
//+
Point(5) = {0, 0, 0, 1.0};
//+
Line(1) = {1, 5};
//+
Line(2) = {5, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 4};
//+
Line(5) = {4, 1};
//+
Physical Curve("inlet", 6) = {5};
//+
Physical Curve("symmetry", 7) = {1};
//+
Physical Curve("wall", 8) = {2};
//+
Physical Curve("outlet", 9) = {3, 4};
//+
Curve Loop(1) = {1,2,3,4,5};
//+
Plane Surface(1) = {1};
//+
Physical Surface("fluid", 10) = {1};
//+
SetOrder 4;
