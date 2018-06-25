# coding: utf-8
from mlhiphy.calculus import dx
from mlhiphy.calculus import Constant
from mlhiphy.calculus import Unknown
from mlhiphy.kernels import compute_kernel, generic_kernel

from sympy import expand
from sympy import Lambda
from sympy import symbols
from sympy import exp


x, xi, xj = symbols('x xi xj')

alpha = Constant('alpha')
beta  = Constant('beta')
theta  = Constant('theta')

u = Unknown('u')
expr = alpha * u + dx(u) #+ dx(dx(u))

print('> generic_kernel := ', expand(generic_kernel(expr, u, (xi, xj))))

xi, xj = symbols('xi xj')
kuu = theta * exp(-0.5*((xi - xj)**2))

kuf = compute_kernel(expr, kuu, xi)
kfu = compute_kernel(expr, kuu, xj)
kff = compute_kernel(expr, kuu, (xi, xj))

print('> kuf := ', kuf)
print('> kfu := ', kfu)
print('> kff := ', kff)
