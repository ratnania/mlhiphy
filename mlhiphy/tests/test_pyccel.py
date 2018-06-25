# coding: utf-8
from mlhiphy.calculus import dx
from mlhiphy.calculus import Constant
from mlhiphy.calculus import Unknown
from mlhiphy.kernels import compute_kernel, generic_kernel
from mlhiphy.templates import template_scalar, template_header_scalar

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


import os
def mkdir_p(dir):
    if os.path.isdir(dir):
        return
    os.makedirs(dir)

def write_code(name, code, ext='py', folder='.pyccel'):
    filename = '{name}.{ext}'.format(name=name, ext=ext)
    if folder:
        mkdir_p(folder)
        filename = os.path.join(folder, filename)

    f = open(filename, 'w')
    for line in code:
        f.write(line)
    f.close()

def compile_kernel(name, expr, kuu, args, export_pyfile=True):
    from sympy import IndexedBase

    if not isinstance(args, (tuple, list)):
        args = [args]

    # ...
    kernel = compute_kernel(expr, kuu, args)
    # ...

    # ...
    params = [i for i in kernel.free_symbols if not(i in args)]
    params_str = ', '.join([i.name for i in params])
    # ...

    # ...
    dtypes = ['double' for i in params]
    dtypes_str = ', '.join([i for i in dtypes])
    # ...

    # ...
    ijk = symbols('i j k')
    X = IndexedBase('x')
    for xi,i in zip(args, ijk):
        kernel = kernel.subs({xi: X[i]})
    # ...

    # ...
    template_str = 'template_{pattern}'.format(pattern='scalar')
    try:
        template = eval(template_str)
    except:
        raise ValueError('Could not find the corresponding template {}'.format(template_str))

    code = template.format(__KERNEL_NAME__=name,
                           __KERNEL__=kernel,
                           __PARAMS__=params_str)
#    print(code)
    # ...

    # ...
    template_str = 'template_header_{pattern}'.format(pattern='scalar')
    try:
        template = eval(template_str)
    except:
        raise ValueError('Could not find the corresponding template {}'.format(template_str))

    header = template.format(__KERNEL_NAME__=name,
                             __PARAM_TYPES__=dtypes_str)
#    print(header)
    # ...

    # ... export the python code of the module
    if export_pyfile:
        write_code(name, code, ext='py', folder='.pyccel')
    # ...

    from pyccel.epyccel import epyccel
    kernel = epyccel(code, header, name=name)

    # ... TODO debug
#    _kernel = epyccel(code, header, name=name)
#
#    from mlhiphy.templates import template_main
#    template = template_main.format(__KERNEL_NAME__=name,
#                                    __PARAMS__=params_str)
#    kernel = eval(template)
    # ...

    return kernel

_kff = compile_kernel('kff', expr, kuu, (xi, xj))
from numpy import zeros
def kff(x, *args):
    n = x.size
    k = zeros((n,n), order='F')
    args = list(args) + [k]
    return _kff(n, x, *args)

from numpy import linspace
x = linspace(0., 1., 100)
alpha = 0.1
theta = 0.4
y = kff(x, alpha, theta)
