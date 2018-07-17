Preface
=======

A great challenge for researchers with many applications in the applied sciences is to use vast available data sets and blend them with differential equations. For instance these methods can be used for verification and validation processes, which need to be undertaken in order to design well-functioning simulation software. This is also of special interest to the Max Planck Institute.
Concretely, we are interested in finding the relationship between two black-box functions :math:`u` and :math:`f` based on possibly noisy data :math:`\{X_u, u(X_u)\}` and :math:`\{X_f, f(X_f)\}`. That is, in specifying the form of the transformation :math:`\mathcal{L}_x^{\phi} u(x) = f(x)`. We presume to know the form of the transformation :math:`\mathcal{L}_x^{\phi}` up to a set of unknown parameters :math:`\phi`. 
This setup falls under the broad range of so-called *inverse problems*, which occur often in diverse scientific disciplines.
An example application would be to take the classical problem of heat conduction in a medium with unknown conductivity properties. The distribution of temperature is then governed by the Heat Equation, which is a type of PDE. The thermal diffusivity coefficient would, however, be unknown and needs to be estimated.
Using Gaussian Processes to create a framework shedding light on the optimal parameters of the transformation :math:`\mathcal{L}_x^{\phi}` has many advantages. They can be used as flexible priors, describing a distribution over functions and provide a powerful training procedure, coming from a probabilistic viewpoint. They can themselves be seen as a one-layer neural network with infinitely many hidden units. In contrast to meshless methods, the optimal (hyper-)parameters can be *learned* by minimizing the negative log-likelihood function and don't have to be guessed or tuned manually. In contrast to latent force models, which are one of the few existing frameworks for combining machine learning tools with differential equations, there is no need to solve the differential equation either analytically or numerically.

This paper is based mostly on the research conducted by Raissi et al. :cite:`Raissi2017a`. In their paper, applications consist of a fractional equation, an integro-differential equation, a reaction-diffusion PDE and the Heat Equation. Whilst their code was written in Matlab, we implement their methodology in Python and by doing this, also making it more accessible.


.. rubric:: Notes regarding the contents


In order to retain comparability, in each chapter we will work with a set of points :math:`X` with elements in :math:`[0,1]^n` where n is the number of dimensions we are working with. We always use either 10 or 20 points with the corresponding function values :math:`(X,Y_u, Y_f)` as data samples. In this range, the estimates are good and the computation cost low. 

We will mostly stick with a noise parameter of :math:`s=10^{-7}`, due to a trade-off between having a well-conditioned matrix and accuracy: Our data samples are generated without any noise, so the lower the noise parameter, the more accurate our estimation should be. Setting :math:`s=0` would increase the likelihood of having to work with an ill-conditioned or singular matrix, since two columns in the covariance matrix corresponding to two points being close to each other would have almost equal values. We want to avoid this, since, when calculating the negative log-likelihood, we have to calculate the inverse of the covariance matrix. When working with noisy data on the other hand, one could optimize over the parameter s as well.

A number of code cells from the notebook are missing in the report. This has been done to improve readability. For the complete notebook, we refer to our GitHub repository.