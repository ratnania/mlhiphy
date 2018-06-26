# coding: utf-8
from mlhiphy.calculus import dx, dy
from mlhiphy.calculus import Constant
from mlhiphy.calculus import Unknown
from mlhiphy.kernels import compute_kernel, generic_kernel

from sympy import expand
from sympy import Lambda
from sympy import symbols
from sympy import exp
from sympy import Tuple


def test_1d():
    x, xi, xj = symbols('x xi xj')

    alpha = Constant('alpha')
    beta  = Constant('beta')
    mu    = Constant('mu')
    theta = Constant('theta')

    u = Unknown('u')
    #expr = alpha * u + dx(u)
    expr = mu * u + alpha * dx(u) + beta * dx(dx(u))

    print('> generic_kernel := ', expand(generic_kernel(expr, u, xi)))
    print('> generic_kernel := ', expand(generic_kernel(expr, u, xj)))
    print('> generic_kernel := ', expand(generic_kernel(expr, u, (xi, xj))))

    kuu = theta * exp(-0.5*((xi - xj)**2))

    kuf = compute_kernel(expr, kuu, xi)
    kfu = compute_kernel(expr, kuu, xj)
    kff = compute_kernel(expr, kuu, (xi, xj))

    print('> kuf := ', kuf)
    print('> kfu := ', kfu)
    print('> kff := ', kff)

def test_2d():
    x, xi, xj = symbols('x xi xj')
    y, yi, yj = symbols('y yi yj')

    X  = Tuple(x,y)
    Xi = Tuple(xi,yi)
    Xj = Tuple(xj,yj)

    alpha = Constant('alpha')
    beta  = Constant('beta')
    mu    = Constant('mu')
    theta = Constant('theta')

    u = Unknown('u')
    expr = alpha * u + dx(u) + dy(u)

    print('> generic_kernel := ', expand(generic_kernel(expr, u, Xi)))
    print('> generic_kernel := ', expand(generic_kernel(expr, u, Xj)))
#    print('> generic_kernel := ', expand(generic_kernel(expr, u, (Xi, Xj))))

#    kuu = theta * exp(-0.5*((Xi - Xj)**2))
#
#    kuf = compute_kernel(expr, kuu, Xi)
#    kfu = compute_kernel(expr, kuu, Xj)
#    kff = compute_kernel(expr, kuu, (Xi, Xj))
#
#    print('> kuf := ', kuf)
#    print('> kfu := ', kfu)
#    print('> kff := ', kff)

#############################################
if __name__ == '__main__':
#    test_1d()
    test_2d()
