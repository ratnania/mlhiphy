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


.. topic:: Definition (Stochastic process):

   [LOPS14] [5.2] Given a set :math:`\mathcal{T} \subset \mathbb{R}`, a measurable space  :math:`( H , \mathcal{H} )`, and a probability space :math:`(\Omega, \mathcal{F}, \mathbb{P})`, a H-valued stochastic process is a set of H-valued random variables :math:`\{X(t): t \in \mathcal{T}\}`. We simply write X(t) to denote the process. To exphasise the dependence on :math:`\omega` and that :math:`X : \Omega \times \mathcal{T} \rightarrow \mathbb{R}`, we may write :math:`X(t,\omega)`.

**Definition** (second order) [LOPS14] [5.9] A real-valued stochastic process is second order if :math:`X(t) \in L^2(\Omega)` for each :math:`t \in \mathcal{T}`. The mean function is defined by :math:`\mu(t) := \mathbb{E}[X(t)]` and the covariance function is defined by :math:`C(s, t) : = Cov(X(s), X(t)))` for all :math:`s,t \in \mathcal{T}`.

**Definition** (real-valued Gaussian process) [LOPS14] [5.10] A real-valued second-order stochastic process :math:`\{X(t): t \in \mathcal{T}\}` is Gaussian if :math:`\mathbf{X} = [X(t_1), \dotsc, X(t_M)]^T` follows a multivariate Gaussian distribution for any :math:`t_1, \dotsc, t_M \in \mathcal{T}` and any :math:`M \in \mathbb{N}`.

**Theorem** [LOPS14] [5.17] (Daniel-Kolmogorov)


**Theorem** [LOPS14] [5.18] Let :math:`\mathcal{T} \subset \mathbb{R}`. The following statements are equivalent.

(1) There exists a real-valued second-order stochastic process :math:`X(t)`  with mean function :math:`\mu(t)` and covariance function :math:`C(s, t)`.

(2) The function :math:`\mu` maps from :math:`\mathcal{T} \rightarrow \mathbb{R}` and the function :math:`C` maps from :math:`\mathcal{T} \times \mathcal{T} \rightarrow \mathbb{R}`. Further C is symmetric and non-negative definite.

*Proof*

:math:`(1) \implies (2)`: As the process is second-order, the mean :math:`\mu(t)` and covariance :math:`C(s, t)` are well defined in :math:`\mathbb{R}`. Then, :math:`C(s, t)` is non-negative definite, because for any :math:`a_1, \dotsc , a_N \in \mathbb{R}`
and :math:`t_1, \dotsc, t_N \in \mathcal{T}`

.. math::

   \left.\begin{aligned} \sum _ { j , k = 1 } ^ { N } a _ { j } C \left( t _ { j } , t _ { k } \right) a _ { k } & = E \left[ \sum _ { j , k = 1 } ^ { N } \left( X \left( t _ { j } \right) - \mu \left( t _ { j } \right) \right) \left( X \left( t _ { k } \right) - \mu \left( t _ { k } \right) \right) a _ { j } a _ { k } \right] \\ & = E \left[ | \sum _ { j = 1 } ^ { N } a _ { j } \left( X \left( t _ { j } \right) - \mu \left( t _ { j } \right) \right) | ^ { 2 } \right] \geq 0 \end{aligned} \right.

:math:`C(s, t)` is symmetric as :math:`Cov(X(s), X(t)) = Cov(X(t), X(s))` for all :math:`t,s \in \mathcal{T}`.

:math:`(2) \implies (1)`: Consider any :math:`t_1, \dotsc, t_M \in \mathcal{T}` and let :math:`C^N \in \mathbb{R}^{N \times N}` be the matrix with entries :math:`c_{jk} = C(t_j, t_k)` for :math:`j, k = 1, \dotsc, N`. :math:`C^N` is symmetric and is non-negative definite because

.. math::

   \mathbf { a } ^ { T } C _ { N } \mathbf { a } = \sum _ { j , k = 1 } ^ { N } a _ { j } C \left( t _ { j } , t _ { k } \right) a _ { k } \geq 0 , \quad \forall a \in \mathbb { R } ^ { N }

and hence is a valid covariance matrix.


**Corollary** [LOPS14] [5.19] The probability distribution :math:`\mathbb { P } _ { X }` on :math:`\left( \mathbb { R } ^ { \mathcal { T } } , \mathcal { B } \left( \mathbb { R } ^ { \mathcal { T } } \right) \right)` of a real-valued Gaussian process :math:`X(t)` is uniquely determined by its mean :math:`\mu : \mathcal { J } \rightarrow \mathbb { R }` and covariance function :math:`C : \mathcal { T } \times \mathcal { T } \rightarrow \mathbb { R }`.

**Definition** [LOPS14] 7.1 (random field)

**Definition** [LOPS14] 7.3 (second-order)

**Definition** [LOPS14] 7.5 (Gaussian random field)

Kernels
-----------

**Definition** [LOPS14] [5.31] (mean-square continuity)

In machine learning literature, the covariance functions are referred to as kernels.

List of kernels
+++++++++++++++++++



[LOPS] Theorem 6.5