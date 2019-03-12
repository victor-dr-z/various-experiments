from mshr import *
import dolfin as df

box = Box(df.Point(0,0,0), df.Point(1,1,1))
c1 = Cylinder(df.Point(.5,.5,1),df.Point(.5,.5,0),.3,.3)
c2 = Cylinder(df.Point(0,.5,.5),df.Point(1,.5,.5),.3,.3)
c3 = Cylinder(df.Point(.5,0,.5),df.Point(.5,1,.5),.3,.3)
domain = box - c1 - c2 - c3

generator = CSGCGALMeshGenerator3D()
generator.parameters["edge_size"] = 0.025
generator.parameters["facet_angle"] = 25.0
generator.parameters["facet_size"] = 0.05

# Invoke the mesh generator
#mesh = generator.generate(CSGCGALDomain3D(domain))
mesh = generate_mesh(domain, 30)
#dolfin.plot(mesh, "3D mesh")
df.File('lattice_1.pvd') << mesh
df.File('lattice_1.xml') << mesh

#domain = Rectangle(dolfin.Point(0., 0.), dolfin.Point(1., 1.)) -Circle(dolfin.Point(0.0, 0.0), .35)
