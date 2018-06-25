# coding: utf-8

import numpy as np

from sympy.core.containers import Tuple
from sympy import symbols
from sympy import Symbol
from sympy import Lambda
from sympy.core import Basic
from sympy import Function
from sympy import preorder_traversal
from sympy import diff


from calculus   import Constant
from calculus import (dx, dy, dz)
from calculus import LinearOperator
from calculus import Field
from calculus import _generic_ops, _partial_derivatives

# ...
def test_0():
    x,y, a = symbols('x y a')

    # ...
    expr = x+y
    print('> expr := {0}'.format(expr))

    expr = LinearOperator(expr)
    print('> gelatized := {0}'.format(expr))
    print('')
    # ...

    # ...
    expr = 2*x+y
    print('> expr := {0}'.format(expr))

    expr = LinearOperator(expr)
    print('> gelatized := {0}'.format(expr))
    print('')
    # ...

    # ...
    expr = a*x+y
    print('> expr := {0}'.format(expr))

    expr = LinearOperator(expr)
    print('> gelatized := {0}'.format(expr))
    # ...

    # ...
    expr = 2*a*x+y
    print('> expr := {0}'.format(expr))

    expr = LinearOperator(expr)
    print('> gelatized := {0}'.format(expr))
    # ...
# ...

# ...
def test_1():
    u, v, a = symbols('u v a')

    # ...
    expr = u+v
    print('> expr := {0}'.format(expr))

    expr = dx(expr)
    print('> gelatized := {0}'.format(expr))
    print('')
    # ...

    # ...
    expr = 2*u*v
    print('> expr := {0}'.format(expr))

    expr = dx(expr)
    print('> gelatized := {0}'.format(expr))
    print('')
    # ...

    # ... dx should not operate on u^2,
    #     since we consider only linearized weak formulations
    expr = u*u
    print('> expr := {0}'.format(expr))

    expr = dx(expr)
    print('> gelatized := {0}'.format(expr))
    print('')
    # ...
# ...

# ...
def test_2():
    u, v = symbols('u v')
    F = Field('F')

    # ...
    expr = F*v*u
    print('> expr := {0}'.format(expr))

    expr = dx(expr)
    print('> gelatized := {0}'.format(expr))
    print('')
    # ...
# ...

def test_kernel():
    from sympy import exp

    x_i, x_j, theta = symbols('x_i x_j theta')
    alpha = Constant('alpha')
    beta  = Constant('beta')
    tau  = Constant('tau')
    kuu = theta * exp(-1/(2)*((x_i - x_j)**2))
    print(kuu)

# .....................................................
if __name__ == '__main__':
    test_0()
    test_1()
    test_2()
    test_kernel()
