# coding: utf-8

# TODO add action of diff operators on sympy known functions

import numpy as np

from sympy.core.sympify import sympify
from sympy.simplify.simplify import simplify
from sympy import Symbol
from sympy import Lambda
from sympy import Function
from sympy import bspline_basis
from sympy import lambdify
from sympy import cos
from sympy import sin
from sympy import Rational
from sympy import diff
from sympy import Matrix
from sympy import latex
from sympy import Integral
from sympy import I as sympy_I
from sympy.core import Basic
from sympy.core.singleton import S
from sympy.simplify.simplify import nsimplify
from sympy.utilities.lambdify import implemented_function
from sympy.matrices.dense import MutableDenseMatrix
from sympy import Mul, Add
from sympy import Tuple
from sympy import postorder_traversal
from sympy import preorder_traversal

from sympy.core.expr import Expr
from sympy.core.containers import Tuple
from sympy import Integer, Float

from sympy import symbols, Tuple, Symbol, sympify, expand
from sympy import Add, Mul
from sympy import preorder_traversal, Expr
from sympy import simplify
from sympy import S
from sympy.core.compatibility import is_sequence
from sympy import Basic
from sympy import Function
from sympy import Indexed

#try:
#    from pyccel.ast.core import IndexedElement as Indexed
#
#    ENABLE_PYCCEL = True
#except:
#    from sympy import Indexed
#
#    ENABLE_PYCCEL = False


#if ENABLE_PYCCEL:
#    _coeffs_registery += (Variable,)

# ...
class Constant(Symbol):
    """
    Represents a constant symbol.

    Examples

    """
    pass
# ...

# ...
class Field(Symbol):
    """
    Represents a Field variable.

    Examples

    """
    pass
# ...

# ...
class Unknown(Symbol):
    """
    Represents an unknown function

    Examples

    """
    pass
# ...

_coeffs_registery = (Integer, Float, Constant)

# ...
class LinearOperator(Function):
    """

    Examples
    ========

    """

    nargs = None
    name = 'Grad'

    def __new__(cls, *args, **options):
        # (Try to) sympify args first

        if options.pop('evaluate', True):
            r = cls.eval(*args)
        else:
            r = None

        if r is None:
            return Basic.__new__(cls, *args, **options)
        else:
            return r

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            return Indexed(self, *indices, **kw_args)
        else:
            return Indexed(self, indices, **kw_args)

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        expr = _args[0]
        if isinstance(expr, Add):
            args = expr.args
            args = [cls.eval(a) for a in expr.args]
            return Add(*args)

        if isinstance(expr, Mul):
            coeffs  = [a for a in expr.args if isinstance(a, _coeffs_registery)]
            vectors = [a for a in expr.args if not(a in coeffs)]

            a = S.One
            if coeffs:
                a = Mul(*coeffs)

            b = S.One
            if vectors:
                b = cls(Mul(*vectors), evaluate=False)

            return Mul(a, b)

        return cls(expr, evaluate=False)
# ...

# ...
class DifferentialOperator(LinearOperator):
    """
    This class is a linear operator that applies the Leibniz formula

    Examples
    ========

    """
    coordinate = None

    @classmethod
    def eval(cls, *_args):
        """."""

        expr = _args[0]
        if isinstance(expr, Add):
            args = expr.args
            args = [cls.eval(a) for a in expr.args]
            return Add(*args)

        if isinstance(expr, Mul):
            coeffs  = [a for a in expr.args if isinstance(a, _coeffs_registery)]
            vectors = [a for a in expr.args if not(a in coeffs)]

            c = S.One
            if coeffs:
                c = Mul(*coeffs)

            V = S.One
            if vectors:
                if len(vectors) == 1:
                    V = cls(Mul(vectors[0]), evaluate=False)

                elif len(vectors) == 2:
                    a = vectors[0]
                    b = vectors[1]

                    fa = cls(a, evaluate=False)
                    fb = cls(b, evaluate=False)

                    V = a * fb + fa * b

                else:
                    V = cls(Mul(*vectors), evaluate=False)

            return Mul(c, V)

        return cls(expr, evaluate=False)
# ...

# ...
class dx(DifferentialOperator):
    coordinate = 'x'
    grad_index = 0 # index in grad
    pass

class dy(DifferentialOperator):
    coordinate = 'y'
    grad_index = 1 # index in grad
    pass

class dz(DifferentialOperator):
    coordinate = 'z'
    grad_index = 2 # index in grad
    pass

class dt(DifferentialOperator):
    coordinate = 't'
    grad_index = None # index in grad
    pass

_partial_derivatives = (dx, dy, dz, dt)
# ...

# ...
class DotBasic(Function):
    """

    Examples
    ========

    """

    nargs = None
    name = 'Dot'

    def __new__(cls, *args, **options):
        # (Try to) sympify args first

        if options.pop('evaluate', True):
            r = cls.eval(*args)
        else:
            r = None

        if r is None:
            return Basic.__new__(cls, *args, **options)
        else:
            return r

class Dot_1d(DotBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]
        v = _args[1]

        return u * v

class Dot_2d(DotBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]
        v = _args[1]

        return u[0]*v[0] + u[1]*v[1]

class Dot_3d(DotBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]
        v = _args[1]

        return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
# ...

# ...
class CrossBasic(Function):
    """

    Examples
    ========

    """

    nargs = None
    name = 'Cross'

    def __new__(cls, *args, **options):
        # (Try to) sympify args first

        if options.pop('evaluate', True):
            r = cls.eval(*args)
        else:
            r = None

        if r is None:
            return Basic.__new__(cls, *args, **options)
        else:
            return r

class Cross_2d(CrossBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]
        v = _args[1]

        return u[0]*v[1] - u[1]*v[0]

class Cross_3d(CrossBasic):
    """

    Examples
    ========

    """

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            return Indexed(self, *indices, **kw_args)
        else:
            return Indexed(self, indices, **kw_args)

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]
        v = _args[1]

        return Tuple(u[1]*v[2] - u[2]*v[1],
                     u[2]*v[0] - u[0]*v[2],
                     u[0]*v[1] - u[1]*v[0])
# ...


# ...
class GradBasic(Function):
    """

    Examples
    ========

    """

    nargs = None
    name = 'Grad'

    def __new__(cls, *args, **options):
        # (Try to) sympify args first

        if options.pop('evaluate', True):
            r = cls.eval(*args)
        else:
            r = None

        if r is None:
            return Basic.__new__(cls, *args, **options)
        else:
            return r

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            return Indexed(self, *indices, **kw_args)
        else:
            return Indexed(self, indices, **kw_args)

class Grad_1d(GradBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]

        return dx(u)

class Grad_2d(GradBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]

        return Tuple(dx(u), dy(u))

class Grad_3d(GradBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]

        return Tuple(dx(u), dy(u), dz(u))
# ...


# ...
class CurlBasic(Function):
    """

    Examples
    ========

    """

    nargs = None
    name = 'Curl'

    def __new__(cls, *args, **options):
        # (Try to) sympify args first

        if options.pop('evaluate', True):
            r = cls.eval(*args)
        else:
            r = None

        if r is None:
            return Basic.__new__(cls, *args, **options)
        else:
            return r

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            return Indexed(self, *indices, **kw_args)
        else:
            return Indexed(self, indices, **kw_args)

class Curl_2d(CurlBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]

        return Tuple( dy(u),
                     -dx(u))

class Curl_3d(CurlBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]

        return Tuple(dy(u[2]) - dz(u[1]),
                     dz(u[0]) - dx(u[2]),
                     dx(u[1]) - dy(u[0]))
# ...

# ...
class Rot_2d(Function):
    """

    Examples
    ========

    """

    nargs = None
    name = 'Grad'

    def __new__(cls, *args, **options):
        # (Try to) sympify args first

        if options.pop('evaluate', True):
            r = cls.eval(*args)
        else:
            r = None

        if r is None:
            return Basic.__new__(cls, *args, **options)
        else:
            return r

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            return Indexed(self, *indices, **kw_args)
        else:
            return Indexed(self, indices, **kw_args)

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]

        return dy(u[0]) - dx(u[1])
# ...

# ...
class DivBasic(Function):
    """

    Examples
    ========

    """

    nargs = None
    name = 'Div'

    def __new__(cls, *args, **options):
        # (Try to) sympify args first

        if options.pop('evaluate', True):
            r = cls.eval(*args)
        else:
            r = None

        if r is None:
            return Basic.__new__(cls, *args, **options)
        else:
            return r

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            return Indexed(self, *indices, **kw_args)
        else:
            return Indexed(self, indices, **kw_args)

class Div_1d(DivBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]

        return dx(u)

class Div_2d(DivBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]

        return dx(u[0]) + dy(u[1])

class Div_3d(DivBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]

        return dx(u[0]) + dy(u[1]) + dz(u[2])
# ...


# ...
_coord_registery = ['x', 'y', 'z']
# ...

# ...
_operators_1d = [Dot_1d,
                 Grad_1d, Div_1d]

_operators_2d = [Dot_2d, Cross_2d,
                 Grad_2d, Curl_2d, Rot_2d, Div_2d]

_operators_3d = [Dot_3d, Cross_3d,
                 Grad_3d, Curl_3d, Div_3d]
# ...


# ... generic operators
class GenericFunction(Function):

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            return Indexed(self, *indices, **kw_args)
        else:
            return Indexed(self, indices, **kw_args)

class Dot(GenericFunction):
    pass

class Cross(GenericFunction):
    pass

class Grad(GenericFunction):
    pass

class Curl(GenericFunction):
    pass

class Rot(GenericFunction):
    pass

class Div(GenericFunction):
    pass

_generic_ops  = (Dot, Cross,
                 Grad, Curl, Rot, Div)
# ...
