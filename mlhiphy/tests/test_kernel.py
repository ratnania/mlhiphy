# coding: utf-8
from mlhiphy.calculus import dx, dy, dz
from mlhiphy.calculus import Constant
from mlhiphy.calculus import Unknown
from mlhiphy.kernels import compute_kernel, generic_kernel

from sympy import expand
from sympy import Lambda
from sympy import Function, Derivative
from sympy import symbols
from sympy import exp
from sympy import Tuple

def test_generic_kernel_1d():
    x, xi, xj = symbols('x xi xj')

    u = Unknown('u')

    # ... testing u
    assert(generic_kernel(u, u, xi) == Function('u')(xi))
    assert(generic_kernel(u, u, xj) == Function('u')(xj))
    assert(generic_kernel(u, u, (xi, xj)) == Function('u')(xi, xj))
    # ...

    # ... testing dx(u)
    assert(generic_kernel(dx(u), u, xi) == Derivative(Function('u')(xi), xi))
    assert(generic_kernel(dx(u), u, xj) == Derivative(Function('u')(xj), xj))
    assert(generic_kernel(dx(u), u, (xi, xj)) == Derivative(Function('u')(xi, xj), xi, xj))
    # ...

    # ... testing dx(dx(u))
    assert(generic_kernel(dx(dx(u)), u, xi) == Derivative(Function('u')(xi), xi, xi))
    assert(generic_kernel(dx(dx(u)), u, xj) == Derivative(Function('u')(xj), xj, xj))
    assert(generic_kernel(dx(dx(u)), u, (xi, xj)) == Derivative(Function('u')(xi, xj), xi, xi, xj, xj))
    # ...

def test_generic_kernel_2d():
    x, xi, xj = symbols('x xi xj')
    y, yi, yj = symbols('y yi yj')

    X  = Tuple(x,y)
    Xi = Tuple(xi,yi)
    Xj = Tuple(xj,yj)

    u = Unknown('u')

    # ... testing u
    assert(generic_kernel(u, u, xi) == Function('u')(xi))
    assert(generic_kernel(u, u, xj) == Function('u')(xj))
    assert(generic_kernel(u, u, (xi, xj)) == Function('u')(xi, xj))
    # ...

    # ... testing dx(u)
    assert(generic_kernel(dx(u), u, Xi) ==
           Derivative(Function('u')(*Xi), xi))

    assert(generic_kernel(dx(u), u, Xj) ==
           Derivative(Function('u')(*Xj), xj))

    assert(generic_kernel(dx(u), u, (Xi, Xj)) ==
           Derivative(Function('u')(*Xi, *Xj), xi, xj))
    # ...

    # ... testing dy(u)
    assert(generic_kernel(dy(u), u, Xi) ==
           Derivative(Function('u')(*Xi), yi))

    assert(generic_kernel(dy(u), u, Xj) ==
           Derivative(Function('u')(*Xj), yj))

    assert(generic_kernel(dy(u), u, (Xi, Xj)) ==
           Derivative(Function('u')(*Xi, *Xj), yi, yj))
    # ...

    # ... testing dx(dx(u))
    assert(generic_kernel(dx(dx(u)), u, Xi) ==
           Derivative(Function('u')(*Xi), xi, xi))

    assert(generic_kernel(dx(dx(u)), u, Xj) ==
           Derivative(Function('u')(*Xj), xj, xj))

    assert(generic_kernel(dx(dx(u)), u, (Xi, Xj)) ==
           Derivative(Function('u')(*Xi, *Xj), xi, xi, xj, xj))
    # ...

def test_generic_kernel_3d():
    x, xi, xj = symbols('x xi xj')
    y, yi, yj = symbols('y yi yj')
    z, zi, zj = symbols('z zi zj')

    X  = Tuple(x,y,z)
    Xi = Tuple(xi,yi,zi)
    Xj = Tuple(xj,yj,zj)

    u = Unknown('u')

    # ... testing u
    assert(generic_kernel(u, u, xi) == Function('u')(xi))
    assert(generic_kernel(u, u, xj) == Function('u')(xj))
    assert(generic_kernel(u, u, (xi, xj)) == Function('u')(xi, xj))
    # ...

    # ... testing dx(u)
    assert(generic_kernel(dx(u), u, Xi) ==
           Derivative(Function('u')(*Xi), xi))

    assert(generic_kernel(dx(u), u, Xj) ==
           Derivative(Function('u')(*Xj), xj))

    assert(generic_kernel(dx(u), u, (Xi, Xj)) ==
           Derivative(Function('u')(*Xi, *Xj), xi, xj))
    # ...

    # ... testing dy(u)
    assert(generic_kernel(dy(u), u, Xi) ==
           Derivative(Function('u')(*Xi), yi))

    assert(generic_kernel(dy(u), u, Xj) ==
           Derivative(Function('u')(*Xj), yj))

    assert(generic_kernel(dy(u), u, (Xi, Xj)) ==
           Derivative(Function('u')(*Xi, *Xj), yi, yj))
    # ...

    # ... testing dz(u)
    assert(generic_kernel(dz(u), u, Xi) ==
           Derivative(Function('u')(*Xi), zi))

    assert(generic_kernel(dz(u), u, Xj) ==
           Derivative(Function('u')(*Xj), zj))

    assert(generic_kernel(dz(u), u, (Xi, Xj)) ==
           Derivative(Function('u')(*Xi, *Xj), zi, zj))
    # ...

    # ... testing dx(dx(u))
    assert(generic_kernel(dx(dx(u)), u, Xi) ==
           Derivative(Function('u')(*Xi), xi, xi))

    assert(generic_kernel(dx(dx(u)), u, Xj) ==
           Derivative(Function('u')(*Xj), xj, xj))

    assert(generic_kernel(dx(dx(u)), u, (Xi, Xj)) ==
           Derivative(Function('u')(*Xi, *Xj), xi, xi, xj, xj))
    # ...

def test_1d():
    x, xi, xj = symbols('x xi xj')

    u = Unknown('u')

    alpha = Constant('alpha')
    beta  = Constant('beta')
    mu    = Constant('mu')
    theta = Constant('theta')

#    expr = alpha * u
#    expr = alpha * dx(u)
#    expr = alpha * u + beta * dx(u)
#    expr = mu * u + dx(u)
#    expr = mu * u + dx(dx(u))
#    expr = mu * u + alpha * dx(u) + beta * dx(dx(u))

    expr = mu * u + dx(u) + dx(dx(u))

#    print('> generic_kernel := ', expand(generic_kernel(expr, u, xi)))
#    print('> generic_kernel := ', expand(generic_kernel(expr, u, xj)))
    print('> generic_kernel := ', expand(generic_kernel(expr, u, (xi, xj))))

#    kuu = theta * exp(-0.5*((xi - xj)**2))
#
#    kuf = compute_kernel(expr, kuu, xi)
#    kfu = compute_kernel(expr, kuu, xj)
#    kff = compute_kernel(expr, kuu, (xi, xj))
#
#    print('> kuf := ', kuf)
#    print('> kfu := ', kfu)
#    print('> kff := ', kff)

def test_2d():
    x, xi, xj = symbols('x xi xj')
    y, yi, yj = symbols('y yi yj')

    X  = Tuple(x,y)
    Xi = Tuple(xi,yi)
    Xj = Tuple(xj,yj)

    u = Unknown('u')

    alpha = Constant('alpha')
    beta  = Constant('beta')
    mu    = Constant('mu')
    nu    = Constant('nu')
    zeta    = Constant('zeta')
    theta = Constant('theta')

#    expr = alpha * u
#    expr = alpha * dx(u)
#    expr = alpha * dy(u)
#    expr = alpha * u + beta * dx(u)
#    expr = alpha * u + beta * dy(u)
#    expr = mu * u + alpha * dx(u) + beta * dx(dx(u))
#    expr = mu * u + alpha * dx(u) + beta * dy(dy(u))
    expr = mu * u + alpha * dx(u) + beta * dx(dx(u)) + nu * dy(dy(u)) + zeta * dx(dy(u))

#    print('> generic_kernel := ', expand(generic_kernel(expr, u, Xi)))
#    print('> generic_kernel := ', expand(generic_kernel(expr, u, Xj)))
    print('> generic_kernel := ', expand(generic_kernel(expr, u, (Xi, Xj))))

#    kuu = theta * exp(-0.5*((xi - xj)**2 + (yi - yj)**2))
#
#    kuf = compute_kernel(expr, kuu, Xi)
#    kfu = compute_kernel(expr, kuu, Xj)
#    kff = compute_kernel(expr, kuu, (Xi, Xj))
#
#    print('> kuf := ', kuf)
#    print('> kfu := ', kfu)
#    print('> kff := ', kff)

def test_3d():
    x, xi, xj = symbols('x xi xj')
    y, yi, yj = symbols('y yi yj')
    z, zi, zj = symbols('z zi zj')

    X  = Tuple(x,y,z)
    Xi = Tuple(xi,yi,zi)
    Xj = Tuple(xj,yj,zj)

    u = Unknown('u')

    alpha = Constant('alpha')
    beta  = Constant('beta')
    mu    = Constant('mu')
    nu    = Constant('nu')
    theta = Constant('theta')

#    expr = alpha * u
#    expr = alpha * dx(u)
#    expr = alpha * dy(u)
#    expr = alpha * dz(u)
#    expr = alpha * u + beta * dx(u)
#    expr = alpha * u + beta * dy(u)
#    expr = alpha * u + beta * dz(u)
#    expr = mu * u + alpha * dx(u) + beta * dx(dx(u))
#    expr = mu * u + alpha * dx(u) + beta * dy(dy(u))
#    expr = mu * u + alpha * dx(u) + beta * dz(dz(u))
    expr = mu * u + alpha * dx(u) + beta * dy(dz(u)) + nu * dx(dz(u))

#    print('> generic_kernel := ', expand(generic_kernel(expr, u, Xi)))
#    print('> generic_kernel := ', expand(generic_kernel(expr, u, Xj)))
    print('> generic_kernel := ', expand(generic_kernel(expr, u, (Xi, Xj))))

#    kuu = theta * exp(-0.5*((xi - xj)**2 + (yi - yj)**2) + (zi - zj)**2))
#
#    kuf = compute_kernel(expr, kuu, Xi)
#    kfu = compute_kernel(expr, kuu, Xj)
#    kff = compute_kernel(expr, kuu, (Xi, Xj))
#
#    print('> kuf := ', kuf)
#    print('> kfu := ', kfu)
#    print('> kff := ', kff)

def test_est_2dkernel():
    """example from Harsha."""
    x, xi, xj = symbols('x xi xj')
    y, yi, yj = symbols('y yi yj')

    X  = Tuple(x,y)
    Xi = Tuple(xi,yi)
    Xj = Tuple(xj,yj)

    u = Unknown('u')

    phi = Constant('phi')
    theta = Constant('theta')

    expr = phi * u + dx(u) + dy(dy(u))

    print('> generic_kernel := ', expand(generic_kernel(expr, u, (Xi, Xj))))
    print('')

    kuu = theta * exp(-0.5*((xi - xj)**2 + (yi - yj)**2))

    kuf = compute_kernel(expr, kuu, Xi)
    kfu = compute_kernel(expr, kuu, Xj)
    kff = compute_kernel(expr, kuu, (Xi, Xj))

    print('> kuf := ', kuf)
    print('> kfu := ', kfu)
    print('> kff := ', kff)

#############################################
if __name__ == '__main__':
    test_generic_kernel_1d()
    test_generic_kernel_2d()
    test_generic_kernel_3d()
    test_1d()
    test_2d()
    test_3d()
    test_est_2dkernel()
