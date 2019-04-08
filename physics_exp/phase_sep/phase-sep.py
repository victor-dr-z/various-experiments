

import random
from dolfin import *


class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = 0.63 + 0.02*(0.5 - random.random())
        values[1] = 0.0
    def value_shape(self):
        return (2,)


class CahnHilliardEquation(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)


lmbda  = 1.0e-02  # surface parameter
dt     = 5.0e-06  # time step
theta  = 1.0      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson


parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True


mesh = UnitSquareMesh.create(96, 96, CellType.Type.quadrilateral)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
ME = FunctionSpace(mesh, P1*P1)

# Trial and test functions of the space ``ME`` are now defined::

# Define trial and test functions
du    = TrialFunction(ME)
q, v  = TestFunctions(ME)


u   = Function(ME)  # current solution
u0  = Function(ME)  # solution from previous converged step

# Split mixed functions
dc, dmu = split(du)
c,  mu  = split(u)
c0, mu0 = split(u0)


u_init = InitialConditions(degree=1)
u.interpolate(u_init)
u0.interpolate(u_init)

e1 = Constant((1.,0))
e2 = Constant((0,1.))
m = [e1, -e1, e2, -e2]


c = variable(c)
f    = 0.25*c**2*(1-c)**2
dfdc = diff(f, c)

c_grad = grad(c)
abs_grad = abs(c_grad[0]) + abs(c_grad[1])
#abs_grad = abs(grad(c))

nv = grad(c) / abs_grad

def heaviside(x):
    '''if x.eval() < -DOLFIN_EPS:
        return Constant(0)
    elif x.eval()>DOLFIN_EPS:
        return Constant(1.)
    else:
        return Constant(0.5)'''
    return 0.5*(x+abs(x)) / abs(x)

ai = 0.
wi = 4.
gamma = 1 - sum(ai**wi * heaviside(dot(nv, mi)) for mi in m)
eps = 0.01


multiplier = sqrt(Constant(0.25)*c**2*(Constant(1) - c)**2)
L0 = c*q*dx - c0*q*dx +  multiplier * dt*dot(grad(mu), grad(q))*dx
#L1 = mu*v*dx - dfdc*v*dx - lmbda*dot(grad(c), grad(v))*dx
print(Identity(2)-outer(nv, nv))
L1 = mu*v*dx - gamma/eps*dfdc*v*dx - eps*gamma*dot(grad(c), grad(v))*dx #-\
    #eps*dot(abs_grad*(Identity(2)-outer(nv, nv))*dot(nv, grad(gamma)),grad(v))*dx

L = L0 + L1


a = derivative(L, u, du)


problem = CahnHilliardEquation(a, L)
solver = NewtonSolver()
solver.parameters["linear_solver"] = "lu"
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-6


file = File("result/ps-1/output.pvd", "compressed")

# Step in time
t = 0.0
T = 50*dt
while (t < T):
    t += dt
    u0.vector()[:] = u.vector()
    solver.solve(problem, u.vector())
    file << (u.split()[0], t)
