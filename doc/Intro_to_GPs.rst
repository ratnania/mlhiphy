Introduction to Gaussian Processes

==================================

We begin by introducing the concept of Gaussian Processes

**Gaussian Process (Definition):**
A Gaussian Process is a collection of random variables, such that every finite collection
of these random variables has a multivariate normal distribution.

Gaussian Processes are thus a special type of stochastic processes. Notably, a Gaussian Process is fully determined by its mean function :math:`m(x)` and its covariance function :math:`k(x,x')`. We will not index the random variables with points in time, as it is usual for stochastic processes, but with points in space. 
.. math::
`f(x) \sim \mathcal{GP}(m(x), k(x, x'))`

file:///C:/Users/SurfaceAdmin/Documents/TUM%20Master/Data%20Innovation%20Lab/Max%20Planck%20Institute%20for%20Plasma%20Physics/Maziar%20Raissi%20_%20Gaussian%20Processes%20Tutorial_files/gp-ml-rasmussen.pdf