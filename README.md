# mlhiphy
Machine Learning for Hidden Physics and Partial Differential Equations


### Understanding Gaussian Processes

* [A simple Gaussian Process](http://nbviewer.jupyter.org/github/ratnania/mlhiphy/blob/master/intro/simple_gp.ipynb)
* [Using pyGPs](http://nbviewer.jupyter.org/github/ratnania/mlhiphy/blob/master/Simple%20GP%20with%20pyGPs.ipynb)
* [Using GPy](http://nbviewer.jupyter.org/github/ratnania/mlhiphy/blob/master/intro/simple_gp_GPy.ipynb)



### Using Gaussian processes to estimate parameters in Linear operators

* [Using pyGPs (has limitations)](http://nbviewer.jupyter.org/github/ratnania/mlhiphy/blob/master/parameter_estimation/par_est_with_pyGPs.ipynb)
* [With custom code for kernels - 1D example](http://nbviewer.jupyter.org/github/ratnania/mlhiphy/blob/master/parameter_estimation/par_est.ipynb)
* [2D example](http://nbviewer.jupyter.org/github/ratnania/mlhiphy/blob/master/parameter_estimation/par_est_2dkernel.ipynb)


### Parameter estimation for the Heat equation

* [Using pyGPs (has limitations)](http://nbviewer.jupyter.org/github/ratnania/mlhiphy/blob/master/Heat_Equation_with_pyGPs.ipynb)
* [with the Backward Euler scheme](http://nbviewer.jupyter.org/github/ratnania/mlhiphy/blob/master/heat_eqn_numerical_gp.ipynb)
* [Non-homogeneous problem](http://nbviewer.jupyter.org/github/ratnania/mlhiphy/blob/master/heat_eqn_non_homo_numerical_gp.ipynb)
* [Using 2D kernel](http://nbviewer.jupyter.org/github/ratnania/mlhiphy/blob/master/heat_eqn.ipynb)


### Installation of the Python package

The Python package **mlhiphy** can be installed in the traditional way


* **Standard mode**::

```shell
    python3 -m pip install .
```

* **Development mode**::

```shell
    python3 -m pip install --user -e .
```

### Automatic computation of Kernels

1. [1d example](http://nbviewer.jupyter.org/github/ratnania/mlhiphy/blob/devel-ara/autoker/01_example_1d.ipynb)
2. [2d example](http://nbviewer.jupyter.org/github/ratnania/mlhiphy/blob/devel-ara/autoker/02_example_2d.ipynb)
3. [3d example](http://nbviewer.jupyter.org/github/ratnania/mlhiphy/blob/devel-ara/autoker/03_example_3d.ipynb)


### Version 2!:

In https://github.com/Slowpuncher24/mlhiphy_v2 you can find a considerably improved version of mlhiphy. It features:

* A much more efficient and stable implementation of the negative log-likelihood. This vastly improves the algorithm, as the optimization of the negative log-likelihood is at its center. This was done by utilizing the block matrix structure of the covariance matrix and by using the Cholesky decomposition.
* The inference of up to four hidden parameters in three dimensions, as opposed to one hidden parameter in two dimensions (respectively counting the temporal dimension as one).
* An alternative implementation of the negative log-likelihood for the noise-free case, where we only have to optimize over one hyperparameter less (the signal variance can be written in terms of other values).
* The implementation and tests of using the Matérn-5/2-kernel, which is the most promising alternative to the SE kernel.
* The implementation and tests of a viable alternative to the Nelder-Mead optimization algorithm, namely a variant of the nonlinear conjugate gradient method (it is scipy's implementation up to a minor tweak to the line search algorithm).









