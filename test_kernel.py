# coding: utf-8
from calculus import dx
from calculus import Constant
from calculus import Unknown
from kernels import compute_kernel, generic_kernel

from sympy import expand
from sympy import Lambda
from sympy import symbols
from sympy import exp


x, xi, xj = symbols('x xi xj')

alpha = Constant('alpha')
beta  = Constant('beta')

u = Unknown('u')
expr = alpha * u + dx(u)

print('> generic_kernel := ', expand(generic_kernel(expr, u, (xi, xj))))

xi, xj, theta = symbols('xi xj theta')
kuu = theta * exp(-0.5*((xi - xj)**2))

kuf = compute_kernel(expr, kuu, xi)
kfu = compute_kernel(expr, kuu, xj)
kff = compute_kernel(expr, kuu, (xi, xj))

print('> kuf := ', kuf)
print('> kfu := ', kfu)
print('> kff := ', kff)
