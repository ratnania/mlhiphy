pyGPs

^^^^^

When starting to work on this project, we sought a suitable Python-Package giving us the capabilities we needed concerning Gaussian Processes. We therefore implemented the early code using the pyGPs-package. It is a package best suited for classical Gaussian Process Regression or Classification. After installing pyGPs and testing it, using the provided test data there appeared an 'Intel MKL FATAL ERROR: Cannot load mkl_intel_thread.dll' - error, which we were not able to resolve. By switching from Windows to a Linux Cluster, this issue could be circumvented.

Our approach with pyGPs went as follows:

1. We assume Gaussian Priors with zero mean and an RBF Kernel for u and f, i.e.: 

   .. math::

		u(x) \sim \mathcal{GP}(0, k_{uu}(x, x'; \sigma_u, l_u)) \\
		f(x) \sim \mathcal{GP}(0, k_{ff}(x, x'; \sigma_f, l_f))

2. Given the data $\{X_u, Y_u\}$ and $\{X_f, Y_f\}$ we can now optimize the hyperparameters of the two Gaussian 
   Processes separately using pyGPs.

3. Since :math:`f=\mathcal{L}_x^{\phi} u(x)`, we know that the covariance matrix k_f for f is given by
   [reference to GPtrafo-eq]. As an approximation, we set :math:`k_f = k_{ff}`. Thus also 

   .. math::

		`k_f(x_i, x_i) = k_{ff}(x_i, x_i)` 
	
   must hold for all data points :math:`x_i`. Rearranging leads to some function

   .. math::

		`\phi = g(s_u, s_f, l_u, l_f)`,
	
   which we can evaluate.
   
Now this approach worked for simple examples (cf. [ref. to par_est.ipynb] and [ref. to par_est_with_pyGPs.ipynb]), though it failed for more complicated ones. As an attempt to resolve this problem, we wanted to avoid the approximation in step 3 and work with the correct covariance matrix instead. The pyGPs-package allows the user to define custom covariance functions. This was of no avail mainly due to incongruities in the pyGPs source code regarding the derivatives of covariance functions in combination with poor documentation in that regard and the resulting complexity of the task itself. As an example, we have included a custom covariance function in the appendix ([reference to custom_covariance_heat_eq.ipynb]).