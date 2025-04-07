Normal Distribution
===================

.. automodule:: goal.models.base.gaussian.normal
   :noindex:
   :no-members:

Class Hierarchy
---------------

.. inheritance-diagram:: goal.models.base.gaussian.normal
   :parts: 2
   :top-classes: goal.geometry.exponential_family.combinators.LocationShape,  goal.geometry.manifold.linear.SquareMap, goal.geometry.exponential_family.base.Analytic, goal.geometry.exponential_family.base.ExponentialFamily

\

Base Components
---------------

.. autoclass:: goal.models.base.gaussian.normal.Euclidean
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.models.base.gaussian.normal.Covariance
   :members:
   :undoc-members:
   :show-inheritance:

Normal Distribution
-------------------

.. autoclass:: goal.models.base.gaussian.normal.Normal
   :members:
   :undoc-members:
   :show-inheritance:

Specialized Normal Types
------------------------

.. py:data:: goal.models.base.gaussian.normal.FullNormal

   **Type alias:** ``Normal[PositiveDefinite]``

   Normal distribution with unrestricted positive definite covariance matrix.
   
   This variant can represent arbitrary correlations between dimensions with a full covariance matrix. Most flexible but requires $O(d^2)$ parameters for d-dimensional data and $O(d^3)$ operations for key computations.

.. py:data:: goal.models.base.gaussian.normal.DiagonalNormal

   **Type alias:** ``Normal[Diagonal]``

   Normal distribution with diagonal covariance matrix.
   
   This variant assumes independence between dimensions, with a diagonal covariance matrix representing only per-dimension variances. Requires $O(d)$ parameters and $O(d)$ operations for key computations.

.. py:data:: goal.models.base.gaussian.normal.IsotropicNormal

   **Type alias:** ``Normal[Scale]``

   Normal distribution with scalar multiple of identity covariance matrix.
   
   This variant has equal variance in all dimensions and no correlations. Requires only a single scale parameter regardless of dimensionality and enables highly efficient computations.

.. py:data:: goal.models.base.gaussian.normal.StandardNormal

   **Type alias:** ``Normal[Identity]``

   Normal distribution with identity covariance matrix.
   
   This variant has unit variance in all dimensions and no correlations. Requires no shape parameters and enables optimal computational efficiency.

Covariance Types
----------------

.. py:data:: goal.models.base.gaussian.normal.FullCovariance

   **Type alias:** ``Covariance[PositiveDefinite]``

   Unrestricted positive definite covariance matrix representation.
   
   Stores all $\frac{d(d+1)}{2}$ unique elements of a symmetric positive definite matrix. Supports arbitrary correlation structures with maximum flexibility at the cost of higher parameter count and computational complexity.

.. py:data:: goal.models.base.gaussian.normal.DiagonalCovariance

   **Type alias:** ``Covariance[Diagonal]``

   Diagonal covariance matrix representation.
   
   Stores only $d$ diagonal elements representing per-dimension variances. Assumes independence between dimensions, enabling $O(d)$ storage and operations while still allowing heterogeneous variances across dimensions.

.. py:data:: goal.models.base.gaussian.normal.IsotropicCovariance

   **Type alias:** ``Covariance[Scale]``

   Scalar multiple of identity covariance matrix representation.
   
   Represents the covariance matrix using a single scalar value, allowing for extremely efficient $O(1)$ storage and fast vectorized computations. Models equal variance in all dimensions with no correlations.

Type Hacks
----------

.. autofunction:: goal.models.base.gaussian.normal.cov_to_lin
.. autofunction:: goal.models.base.gaussian.normal.lin_to_cov
