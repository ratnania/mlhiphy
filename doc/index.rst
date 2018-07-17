.. TUM-DI-LAB documentation master file, created by
   sphinx-quickstart on Sun Jun 17 09:19:22 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TUM-DI-LAB's documentation!
======================================

.. toctree::
   :glob:
   :numbered:   
   :caption: Contents

About Gaussian Processes
========================

.. toctree::
   :maxdepth: 1

   report/01-intro_to_gp
   report/02-simple_gp.nblink
   report/03-linear_op_gp
   
Parameter estimation with Gaussian Processes
============================================

.. toctree::
   :maxdepth: 1

   report/04-par_est.nblink
   report/05-par_est2.nblink
   report/06-par_est3.nblink
   report/11-par_est0.nblink
   
Linear PDEs
===========

.. toctree::
   :maxdepth: 1

   report/07-heat.nblink
   report/10-wave.nblink

Non-linear PDEs
===============

.. toctree::
   :maxdepth: 1

   report/08-burgers1.nblink
   report/09-burgers2.nblink

Approach with pyGPs
===================

.. toctree::
   :maxdepth: 1

   report/12-pyGPs.nblink


Conclusion
==============
It has been demonstrated that estimating parameters using a Gaussian process approach works well for a variety of PDEs arising from linear operators.
Even though we have worked with time schemes in the example of Burgers' Equation, this methodology in general circumvents the need for
any discretization methods. When working with truly noisy data, the parameter s (which we have set to 1e-7 in most cases) can be varied or even optimized as well. This approach unfortunately can't really be applied to non-linear equations, only by using workarounds like replacing the non-linear term with the mean as we did in Burgers' Equation, essentially transforming the non-linear equation to a linear one. Using this remedy, we still get good results as we saw in Chapters 4.2.4 and 4.2.5.
In our case, five to eight data samples were sufficient to learn the parameters within a margin of error of 0.058 and 0.0009 in the linear, and 0.5 and 0.16 in the non-linear cases. Using 20 data samples, we were able to estimate the parameters to an accuracy of 0.6 percent.  


Problems with the RBF kernel
--------------------------------
One of the problems we noticed with the RBF kernel is that for more than 30 data points, the final covariance matrix was frequently ill-conditioned. This happens more often, when the length-scale is large. For a kernel represented by :math:`k(x,y) = e^{-\frac{1}{2l}||x-y||^2}`, the length-scale parameter we refer to is :math:`l`. With the increased number of points, the probability of points being close to each other increases and then the respective columns of the covariance matrix will be almost equal, especially when working with a large length-scale.  Hence, this kernel is not good enough for practical purposes on large datasets. The immediate solution that comes to mind to tackle this problem is to do a singular value decomposition. :cite:`Fasshauer2012` outlines the RBF-QR algorithm to avoid these issues. Stable computations can also be acheived using Hermite polynomials :cite:`Fornberg2011`, :cite:`Yurova2017`.


Non-linearity
----------------
In Burgers' Equation we approximated the non-linear term by a constant term. This yields good results for some special cases but is not a generic approach. The problem arises because the product of Gaussian processes does not result in a Gaussian process. Nevertheless, we could utilize the fact that the product of Gaussian distributions is also Gaussian in order to come up with a proper solution. Another approach is to assume priors over the kernel hyperparameters and infer parameters with MCMC sampling schemes :cite:`Calderhead2009`.


Kernel computations
-----------------------
The current framework involves two computations for every operator over the kernel. It is easy to do this by hand for simple operators but even then there is scope for manual error. It would be nice to have a tool to compute the transformed kernels automatically. Some progress has been made at this front resulting in a package for symbolic kernel computations available on this project's GitHub repository :cite:`Ratnani2018`.
   

.. bibliography:: refs.bib
   :style: unsrt
   :all:



.. raw:: latex

   \appendix

Appendix
============
   
.. toctree::
   
   report/Appendix/01-pyGPs_demo.nblink
   report/Appendix/02-pyGPs_covariance.nblink
   report/Appendix/03-gpy_demo.nblink
