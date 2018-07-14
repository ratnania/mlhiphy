Linear operators on GPs
======================================

Regularity of Stochastic Processes
--------------------------------------

**Definition** (mean-square continuity):

   Let :math:`\{ X ( t ) : t \in \mathcal { T } \}` be a mean-zero process. The kernel :math:`k` is continuous at :math:`(t,t)` if and only if :math:`\mathbb { E } \left[ ( X ( t + h ) - X ( t ) ) ^ { 2 } \right] \rightarrow 0` as :math:`h \rightarrow 0`. In particular, if :math:`k \in \mathrm { C } ( \mathcal { T } \times \mathcal { T } )`, then :math:`\{ X ( t ) : t \in \mathcal { T } \}` is mean-square continuous.

**Definition** (mean-square derivative):

   A process :math:`\{ X ( t ) : t \in \mathcal { T } \}` is mean-square differentiable with mean-square derivative :math:`\frac { d X ( t ) } { d t }` if, for all :math:`t \in \mathcal { T }`, we have as :math:`h \rightarrow 0`

   :math:`\| \frac { X ( t + h ) - X ( t ) } { h } - \frac { d X ( t ) } { d t } \| _ { L ^ { 2 } ( \Omega ) } = \mathbb { E } \left[ | \frac { X ( t + h ) - X ( t ) } { h } - \frac { d X ( t ) } { d t } | ^ { 2 } \right] ^ { 1 / 2 } \rightarrow 0`.

**Theorem**:

   Let :math:`\{ X ( t ) : t \in \mathcal { T } \}` be a stochastic process with mean zero. Suppose that the kernel :math:`k \in \mathrm { C } ^ { 2 } ( \mathcal { T } \times \mathcal { T } )`. Then :math:`X(t)` is mean-square differentiable and the derivative :math:`\frac { d X ( t ) } { d t }` has kernel :math:`\frac { \partial ^ { 2 } k ( s , t ) } { \partial s \partial t }`.

*Proof*:

   For any :math:`s,t \in \mathcal{T}` and real constants :math:`h_s,h_t > 0`,

   .. math::

        \operatorname { Cov } ( \frac { X ( s + h_s ) - X ( s ) } { h_s } ,  \frac { X ( t + h_t ) - X ( t ) } { h_t } ) &= \frac { 1 } { h_s h_t } \mathbb { E } [ ( X ( s + h_s ) - X ( s ) ) ( X ( t + h_t ) - X ( t ) ) ] \\
        &= \frac { 1 } { h_s h_t } ( k ( s + h_s , t + h_t ) - k ( s + h_s , t ) - k ( s , t + h_t ) + k ( s , t ) )

   
   A simple calculation with Taylor series shows that the right-hand side converges to :math:`\frac { \partial ^ { 2 } k ( s , t ) } { \partial s \partial t }` as :math:`h_s,h_t \rightarrow 0`.


With a similar approach and setting as the previous theorem, we can calculate the covariance between a Gaussian process and it's mean-square derivative.

   .. math::

      \operatorname { Cov } ( X ( s ), & \frac { X ( t + h ) - X ( t ) } { h } ) = \frac { 1 } { h } \mathbb { E } [ ( X ( s ) ) ( X ( t + h ) - X ( t ) ) ] = \frac { 1 } { h } ( k ( s, t + h ) - k ( s , t ) )

   The right hand side converges to :math:`\frac{\partial}{\partial t}k(s,t)` as :math:`h \rightarrow 0`.




.. Example 5.36 properties of Gaussian covariance

.. Example 6.7 Gaussian covariance is well-defined

**Theorem** (mean-square regularity):

   Let :math:`u(x)` be a mean-zero second-order random field. If the kernel :math:`k \in C(D × D)`, then :math:`u(x)` is mean-square continuous so that :math:`\| u ( \mathbf{x} + \mathbf{h} ) - u ( \mathbf{x} ) \| _ { L ^ { 2 } ( \Omega ) } \rightarrow 0` as :math:`h \rightarrow 0 \forall x \in D`. If :math:`k \in C^2(D \times D)`,then :math:`u(x)` is mean-square differentiable. That is, a random field :math:`\frac { \partial u ( x ) } { \partial x _ { i } }` exists such that

   .. math::

      \| \frac { u \left( \mathbf { x } + h e _ { i } \right) - u ( \mathbf { x } ) } { h } - \frac { \partial u ( \mathbf { x } ) } { \partial x _ { i } } \| _ { L ^ { 2 } ( \Omega ) } \rightarrow 0 \quad \text { as } h \rightarrow 0
    
   and :math:`\frac { \partial u ( x ) } { \partial x _ { i } }` has kernel :math:`k _ { i } ( x , y ) = \frac { \partial ^ { 2 } C ( x , y ) } { \partial x _ { i } \partial y _ { i } }`.



Description of covariance transformations