# coding: utf-8
from mlhiphy.calculus import dx
from mlhiphy.calculus import Constant
from mlhiphy.calculus import Unknown
from mlhiphy.calculus import _partial_derivatives

from sympy import preorder_traversal
from sympy import Derivative
from sympy import Add
from sympy import expand
from sympy import Lambda
from sympy import symbols
from sympy import exp
from sympy import diff
from sympy import Function

def generic_kernel(expr, func, y, args=None):
    if isinstance(y, (list, tuple)):
        args = y
        ei = func
        for xi in y:
            ej = generic_kernel(expr, ei, xi, args=args)
            ei = ej
        return ei
    else:
        if isinstance(func, Unknown):
            func = [func]
        elif isinstance(func, Add):
            fn = [i for i in func.free_symbols if isinstance(i, Unknown)]
            if not(len(fn) == 1):
                raise ValueError('expecting only one unknown')

            expr = expr.subs({fn[0]: func})
            func = fn
        else:
            raise NotImplemented('type = ', type(func))

        if not args:
            args = [y]

        for f in func:
            fnew  = Function(f.name)
            expr = expr.subs({dx(f): fnew(*args).diff(y)})

        ops = [a for a in preorder_traversal(expr) if isinstance(a, _partial_derivatives)]
        for i in ops:
            # if i = dx(u) then type(i) is dx
            op = type(i)
            derivs = [a for a in i.args if isinstance(a, Derivative)]
            # TODO validate
            for a in derivs:
                f = a.expr

                expr = expr.subs({i: a.diff(y)})

        return expr

def compute_kernel(expr, kuu, args):
    if not isinstance(args, (tuple, list)):
        args = [args]

    u = [i for i in expr.free_symbols if isinstance(i, Unknown)]
    if not(len(u) == 1):
        raise ValueError('Expecting one unknown')

    u = u[0]
    expr = generic_kernel(expr, u, args)

    expr = expr.subs({u: kuu})

    xi = args[0]

    # must be done before subs on xi and xj
    if len(args) > 1:
        xj = args[1]
        expr = expr.subs({Derivative(u(xi, xj), xi, xj):
                          diff(kuu, xi, xj)})

    expr = expr.subs({Derivative(u(*args), xi): diff(kuu, xi)})
    if len(args) > 1:
        expr = expr.subs({Derivative(u(*args), xj): diff(kuu, xj)})

    return expr
