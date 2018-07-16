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
It has been demonstrated that the Gaussian process approach works well for a variety of PDEs arising from linear operators. The non-linear cases are difficult as in the case of the Burgers equation.


Problems with the RBF kernel
--------------------------------
One of the problems we noticed with the RBF kernel is that for more than 30 data points, the final covariance matrix was frequently ill-conditioned. This happens in cases where the shape parameter is very small. For a kernel represented as :math:`k(x,y) = e^{-l^2||x-y||^2}`, the shape parameter we refer to is :math:`l`. When this is small, the rows of the covariance matrix become numerically less rank than the number of data points. Hence, this kernel is not good enough for practical purposes on huge datasets. The immediate solution that comes to mind for this problem is to do a singular value decomposition. :cite:`Fasshauer2012` outlines the RBF-QR algorithm to avoid these issues. Stable computations can also be acheived using Hermite polynomials :cite:`Fornberg2011`, :cite:`Yurova2017`.


Non-linearity
----------------
For the Burger's equation we used an approximation to convert the non-linear terms to linear. This works for some special cases but is not a generic approach. The problem arises because the product of Gaussian processes do not result in a Gaussian process. Nevertheless, we could utilise the fact that the product of Gaussian distributions is also Gaussian to come up with a proper solution. Another approach is to assume priors over the kernel hyperparameters and infer parameters with MCMC sampling schemes :cite:`Calderhead2009`.


Kernel computations
-----------------------
The current framework involves two computations for every operator over the kernel. It is easy to do this by hand for simple operators but even then there is scope for manual error. It would be nice to have a tool to compute the transformed kernels automatically. Some progress has been made at this front resulting in a package for symbolic kernel computations available on this project's github repository :cite:`Ratnani2018`.
   

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