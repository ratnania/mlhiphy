.. TUM-DI-LAB documentation master file, created by
   sphinx-quickstart on Sun Jun 17 09:19:22 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TUM-DI-LAB's documentation!
======================================

.. raw:: latex

   \begin{abstract}
   Recent research has suggested great benefits from applying machine learning tools for the verification of parameters in PDEs.
   Building upon this research, we implement and analyze the estimation of parameters in PDEs using Gaussian Processes.
   Knowing only the parameter-dependend (linear) relationship between noisy data, we can infer this parameter by placing a Gaussian Prior on the data and by optimizing a certain log-marginal likelihood function. Here we rely heavily on the fact, that a linear transformation of a Gaussian Process is again a Gaussian Process.
   
   After introducing the concept of Gaussian Processes, we apply this methodology to the Heat Equation, a modified version of the Burgers' Equation and to the Wave Equation. By doing this, we show how the framework can be successfully used in one or more dimensions and to some extent for the estimation of multiple parameters and for those in non-linear transformations.
   \end{abstract}

.. toctree::
   :glob:
   :maxdepth: 2
   :numbered:   
   :caption: Contents

   report/*
   report/Appendix/*



.. [LOPS14] Lord, G. J., Powell, C. E., & Shardlow, T. (2014). An Introduction to Computational Stochastic PDEs. Cambridge: Cambridge University Press. http://doi.org/10.1017/CBO9781139017329

.. Limitations of available tools

.. Linear PDEs

.. Non-linear PDEs

.. PDEs without discretization

.. Results and Analysis

.. Conclusion

