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

