
'''
from dolfin import *

mesh = UnitSquareMesh(10,10)
V = FunctionSpace(mesh,"Lagrange",1)
u = Function(V)
v = TestFunction(V)
x = SpatialCoordinate(mesh)
f = x[0]*x[1]
res = (1.0+u*u)*inner(grad(u),grad(v))*dx + inner(u,v)*dx - inner(f,v)*dx
Dres = derivative(res,u)

class CustomNonlinearProblem(NonlinearProblem):
    def F(self,b,x):
        return assemble(res,tensor=b)
    def J(self,A,x):
        return assemble(Dres,tensor=A)

problem = CustomNonlinearProblem()
solver = PETScSNESSolver()
solver.solve(problem,u.vector())
'''

from dolfin import *

mesh = UnitSquareMesh(10,10)
V = FunctionSpace(mesh,"Lagrange",1)
u = Function(V)
v = TestFunction(V)
x = SpatialCoordinate(mesh)
f = x[0]*x[1]
res = (1.0+u*u)*inner(grad(u),grad(v))*dx + inner(u,v)*dx - inner(f,v)*dx
Dres = derivative(res,u)

class CustomNonlinearProblem(NonlinearProblem):
    def F(self,b,x):
        return assemble(res,tensor=b)
    def J(self,A,x):
        return assemble(Dres,tensor=A)

problem = CustomNonlinearProblem()
solver = PETScSNESSolver()
ksp = solver.snes().getKSP()
ksp.setType(ksp.Type.PREONLY)
pc = ksp.getPC()
pc.setType('lu')
pc.setFactorSolverType('mumps')
solver.parameters['lu_solver'] = 'mu'
solver.solve(problem,u.vector())
for k,v in solver.parameters.items():
    print (k,v)
