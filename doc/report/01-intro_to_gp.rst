Introduction to Gaussian processes
======================================

This chapter develops the theory behind Gaussian processes and Gaussian fields. In simple terms, stochastic processes are families of random variables parameterized by a scalar variable :math:`t \in \mathbb{R}` denoting time. A random field, on the other hand, is a stochastic process parameterized by :math:`x \in \mathbb{R}^d` for :math:`d > 1` often denoting space. Since we will deal with different dimensions throughout the text, we will use the term 'Gaussian process' for both of the cases to improve readability. We begin by a review of univariate and multivariate normal distributions and use them to understand the properties of Gaussian processes.

Gaussian distributions
--------------------------

The univariate Gaussian (normal) distribution has a probability density given by

.. math::

   p ( x; \mu, \sigma ) = \frac { 1 } { \sqrt { 2 \pi } \sigma } \exp \left\{ - \frac { 1 } { 2 \sigma ^ { 2 } } ( x - \mu ) ^ { 2 } \right\}

where :math:`x, \mu, \sigma \in \mathbb{R}`. It is commonly denoted as :math:`x \sim \mathcal{N}(\mu, \sigma)`.

If :math:`y \sim \mathcal{N}(\mu, \sigma)` and :math:`\alpha \in \mathbb{R}`, then

.. math::

   \alpha y \sim \mathcal{N}(\alpha \mu, \alpha^2 \sigma)


A p-dimensional multivariate Gaussian (or Normal) distribution has a joint probability density given by

.. math::

   p ( \mathbf { X} | \mathbf { \mu } , \Sigma ) = ( 2 \pi ) ^ { - p / 2 } | \Sigma | ^ { - 1 / 2 } \exp \left( - \frac { 1 } { 2 } ( \mathbf { X } - \mathbf { \mu } ) ^ { T } \Sigma ^ { - 1 } ( \mathbf { X } - \mathbf { \mu } ) \right)

where :math:`\mathbf{\mu} \in \mathbb{R}^p` is the mean vector and :math:`\Sigma \in \mathbb{R}^p \times \mathbb{R}^p` is the covariance matrix. This is commonly denoted as :math:`\mathbf{X} \sim \mathcal{N}(\mathbf{\mu}, \Sigma)` or :math:`\mathbf{X} \sim \mathcal{N}_p(\mathbf{\mu}, \Sigma)` to specify the dimension.

If :math:`Y = BX + b` where :math:`B \in \mathbb{R}^{q \times p}, rank(B) = q` and :math:`b \in \mathbb{R}^m`, then

.. math::

   Y \sim \mathcal{N}(B\mathbf{\mu} + b, B \Sigma B^T)


Let :math:`\mathbf{X} \sim \mathcal{N}_p(\mathbf{\mu}, \Sigma)` and decompose :math:`\mathbf{X}, \mathbf{\mu}, \Sigma` as 

.. math::

   \mathbf{X} = \begin{pmatrix}
   X_1 \\
   X_2
   \end{pmatrix},
   \mathbf{\mu} = \begin{pmatrix}
   \mu_1 \\ \mu_2
   \end{pmatrix},
   \Sigma = \begin{pmatrix}
   \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \\
   \end{pmatrix}

.. where p = p _ { 1 } + p _ { 2 } , X _ { 1 } \in \mathbb{R} ^ { p _ { 1 } } , X _ { 2 } \in \mathbb{R} ^ { p _ { 2 } } \\
   \mu _ { 1 } \in \mathbb{R} ^ { p _ { 1 } } , \mu _ { 2 } \in \mathbb{R} ^ { p _ { 2 } } \\
   \Sigma _ { 11 } \in \mathbb{R}^{p_1} \times \mathbb{R}^{p_1} , \Sigma _ { 12 } \in \mathbb{R} ^{p_1} \times \mathbb{R}^{p_2} , \Sigma _ { 21 } \in \mathbb{R}^{p_2} \times \mathbb{R}^{p_1} , \text { and } \Sigma _ { 22 } \in \mathbb{R}^{p_2} \times \mathbb{R}^{p_2}

Then, the conditional probability of :math:`X_2` is given by

.. math::

   X _ { 2 } | X _ { 1 } \sim N _ { p _ { 2 } }(\mu _ { 2 } + \Sigma _ { 21 } \Sigma _ { 11 } ^ { - 1 } \left( X _ { 1 } - \mu _ { 1 } \right), \Sigma _ { 22 } - \Sigma _ { 21 } \Sigma _ { 11 } ^ { - 1 } \Sigma _ { 12 })




Gaussian processes
----------------------

The following definitions and theorems are to introduce the concept of Gaussian processes and fields. For detailed discussions and proofs, please refer to :cite:`Powell2014`.

**Definition** (Stochastic process):

    Given a set :math:`\mathcal{T} \subset \mathbb{R}`, a measurable space  :math:`( H , \mathcal{H} )`, and a probability space :math:`(\Omega, \mathcal{F}, \mathbb{P})`, a H-valued stochastic process is a set of H-valued random variables :math:`\{X(t): t \in \mathcal{T}\}`. We simply write X(t) to denote the process. To exphasise the dependence on :math:`\omega` and that :math:`X : \Omega \times \mathcal{T} \rightarrow \mathbb{R}`, we may write :math:`X(t,\omega)`.

**Definition** (second order process):

   A real-valued stochastic process is second order if :math:`X(t) \in L^2(\Omega)` for each :math:`t \in \mathcal{T}`. The mean function is defined by :math:`\mu(t) := \mathbb{E}[X(t)]` and the covariance function (also referred to as the *kernel*) is defined by :math:`k(s, t) : = Cov(X(s), X(t)))` for all :math:`s,t \in \mathcal{T}`.

**Definition** (real-valued Gaussian process):

   A real-valued second-order stochastic process :math:`\{X(t): t \in \mathcal{T}\}` is Gaussian if :math:`\mathbf{X} = [X(t_1), \dotsc, X(t_M)]^T` follows a multivariate Gaussian distribution for any :math:`t_1, \dotsc, t_M \in \mathcal{T}` and any :math:`M \in \mathbb{N}`.


**Theorem**:

   Let :math:`\mathcal{T} \subset \mathbb{R}`. The following statements are equivalent.

   (1) There exists a real-valued second-order stochastic process :math:`X(t)`  with mean function :math:`\mu(t)` and kernel :math:`k(s, t)`.

   (2) The function :math:`\mu` maps from :math:`\mathcal{T} \rightarrow \mathbb{R}` and the function :math:`k` maps from :math:`\mathcal{T} \times \mathcal{T} \rightarrow \mathbb{R}`. Further :math:`k` is symmetric and non-negative definite.



**Corollary**:

   The probability distribution :math:`\mathbb { P } _ { X }` on :math:`\left( \mathbb { R } ^ { \mathcal { T } } , \mathcal { B } \left( \mathbb { R } ^ { \mathcal { T } } \right) \right)` of a real-valued Gaussian process :math:`X(t)` is uniquely determined by its mean :math:`\mu : \mathcal { T } \rightarrow \mathbb { R }` and kernel :math:`k : \mathcal { T } \times \mathcal { T } \rightarrow \mathbb { R }`.

**Definition** (random field):

   For a set :math:`D \subset \mathbb { R } ^ { d }`, a (real-valued) random field :math:`\{ u ( x ) : x \in D \}` is a set of real-valued random variables on a probability space :math:`( \Omega , \mathcal { F } , \mathbb { P } )`. In subsequent text, we drop :math:`\omega \in \Omega` and simply write :math:`u(x)`, although it should be noted that :math:`u : D \times \Omega \rightarrow \mathbb { R }`.

**Definition** (second-order field):

   For a set :math:`D \subset \mathbb { R } ^ { d }`, a random field :math:`\{ u ( x ) : x \in D \}` is second- order if :math:`u ( \mathbf { x } ) \in L ^ { 2 } ( \Omega ) \forall x \in D`. We say a second-order random field has mean function :math:`u ( \mathbf { x } ) \in L ^ { 2 } ( \Omega )` and kernel

   :math:`k ( \mathbf { x } , \mathbf { y } ) = \operatorname { Cov } ( u ( \mathbf { x } ) , u ( \mathbf { y } ) ) : = \mathbb { E } [ ( u ( \mathbf { x } ) - \mu ( \mathbf { x } ) ) ( u ( \mathbf { y } ) - \mu ( \mathbf { y } ) ) ] , \quad \mathbf { x } , \mathbf { y } \in D`

**Definition** (Gaussian random field):

   A Gaussian random field :math:`\{ u ( x ) : x \in D \}` is a second-order random field such that :math:`u = \left[ u \left( x _ { 1 } \right) , u \left( x _ { 2 } \right) , \ldots , u \left( x _ { M } \right) \right] ^ { T }` follows the multivariate Gaussian distribution for any :math:`x _ { 1 } , \ldots , x _ { M } \in D` and any :math:`M \in \mathbb { N }`. We denote it here as :math:`\mathbf { u } \sim \mathbf { GP } ( \mathbf { \mu } , k )` where :math:`\mu _ { i } = \mu \left( x _ { i } \right)` and :math:`k _ { i j } = k \left( x _ { i } , x _ { j } \right)`.



Kernels
-----------



Squared Exponential Kernel 
++++++++++++++++++++++++++++++
It is also called Radial Basis Function kernel (RBF kernel), or Gaussian kernel, which is as following:

.. math::

   k _ { \mathrm { SE } } \left( x , x ^ { \prime } \right) = \sigma ^ { 2 } \exp \left( - \frac { \left( x - x ^ { \prime } \right) ^ { 2 } } { 2 l ^ { 2 } } \right)

The lengthscale :math:`l` determines the length of the "wiggles" in the function. The variance :math:`\sigma^{2}` determines the average distance of the function away from its mean. All the standard kernel has this parameter in front as a scale factor. 

It has become the default kernel for GPs and pyGPs, and we have also chose this kernel for our project, which will be explain in the later section.


Rational Quadratic Kernel
++++++++++++++++++++++++++++++++

.. math::

   k _ { \mathrm { RQ } } \left( x , x ^ { \prime } \right) = \sigma ^ { 2 } \left( 1 + \frac { \left( x - x ^ { \prime } \right) ^ { 2 } } { 2 \alpha \ell ^ { 2 } } \right) ^ { - \alpha }

This kernel is equivalent to adding together many RBF kernels with different lengthscales, where GP priors should have functions which vary smoothly across many lengthscales. If :math:`\alpha \rightarrow \infty`, the RQ is identical to the RBF.




Periodic Kernel 
+++++++++++++++++++

.. math::
   k _ { \operatorname { Per } } \left( x , x ^ { \prime } \right) = \sigma ^ { 2 } \exp \left( - \frac { 2 \sin ^ { 2 } \left( \pi | x - x ^ { \prime } | / p \right) } { \ell ^ { 2 } } \right)


It is obvious that the periodic kernel (derived by David Mackay) is for the function with repeating structures. Its parameters are easily interpretable:

The period :math:`p` is the distnace between repititions of the function.

The lengthscale :math:`l` is the lengthscale function in the same way as in the SE kernel.

Linear Kernel 
++++++++++++++++++

.. math::

   k _ { \mathrm { Lin } } \left( x , x ^ { \prime } \right) = \sigma^ { 2 } ( x - c ) \left( x ^ { \prime } - c \right)


The linear kernel, unlike other kernels, has non-stationary covariance function, which means that it does not solely depend on :math:`x - x ^{ \prime }` . And :math:`c` determines the x-coordinate of the point that all the lines in the posterior go though.

Our Choice
+++++++++++++++

Since our project is based on the Raissi's paper, so we also follow his choice of the kernel. The reason has been stated in his paper:

   In particular, the squared exponential covariance function chosen above implies smooth approximations. More complex function classes can be accommodated by appropriately choosing kernels. For example, non-stationary kernels employing nonlinear warpings of the input space can be constructed to capture discontinuous response. ::


In the following chapter, we will see a simple case of solving a function parameter with different kernels. 




