# coding: utf-8

# TODO imports from numpy should be done inside compile_kernel


template_main = """
def {__KERNEL_NAME__}(x, {__PARAMS__}):
    n = x.size
    from numpy import zeros
    k = zeros((n,n), order='F')
    args = list(args) + [k]
    return _kernel(n, x, *args)
"""

# .............................................
#          KERNEL     scalar case
# .............................................
template_scalar = """
def {__KERNEL_NAME__}(n, x, {__PARAMS__}, k):
    from numpy import exp

    for i in range(0, n):
        for j in range(0, n):
            k[i,j] = {__KERNEL__}
    return k
"""

template_header_scalar = '#$ header procedure {__KERNEL_NAME__}(int, double [:], {__PARAM_TYPES__}, double[:,:])'
# .............................................
