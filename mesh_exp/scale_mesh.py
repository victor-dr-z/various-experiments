from dolfin import *

#mesh = RectangleMesh(Point(0,0,0), Point(1,1,1) ,5,5,5)
mesh = BoxMesh(Point(0,0,0), Point(1,1,1) ,5,5,5)
print(mesh.coordinates())
mesh.coordinates()[:,:] *= 0.001

print(mesh.coordinates())
