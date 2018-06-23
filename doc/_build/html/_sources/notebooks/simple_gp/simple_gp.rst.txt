
Simple example of a Gaussian process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example illustrates how we move from process to
distribution and also shows that the Gaussian process defines a
distribution over functions.

:math:`f \sim \mathcal{GP}(m,k)`

:math:`m(x) = \frac{x^2}{4}`

:math:`k(x,x') = exp(-\frac{1}{2}(x-x')^2)`

:math:`y = f + \epsilon`

:math:`\epsilon \sim \mathcal{N}(0, \sigma^2)`

.. code:: ipython3

    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt

.. code:: ipython3

    x = np.arange(-5,5,0.2)
    n = x.size
    s = 1e-9

.. code:: ipython3

    m = np.square(x) * 0.25

.. code:: ipython3

    a = np.repeat(x, n).reshape(n, n)
    k = np.exp(-0.5*np.square(a - a.transpose())) + s*np.identity(n)

.. code:: ipython3

    r = np.random.multivariate_normal(m, k, 1)
    y = np.reshape(r, n)

.. code:: ipython3

    plt.plot(x,y)
    plt.show()



.. image:: output_6_0.png


