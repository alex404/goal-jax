Transitions, Emissions, and Markov Chains
=========================================

.. automodule:: goal.models.dynamical
   :noindex:
   :no-members:

Linear Gaussian
---------------

.. autoclass:: goal.models.dynamical.LinearGaussianTransition
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.models.dynamical.NormalEmission
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: goal.models.dynamical.create_linear_gaussian_transition

Categorical
-----------

.. autoclass:: goal.models.dynamical.CategoricalKernel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.models.dynamical.CategoricalTransition
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.models.dynamical.CategoricalEmission
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: goal.models.dynamical.create_categorical_transition

MLP
---

.. autoclass:: goal.models.dynamical.MLPTransition
   :members:
   :undoc-members:
   :show-inheritance:

Homogeneous Markov Chains
-------------------------

.. autoclass:: goal.models.dynamical.HomogeneousGaussianMarkovChain
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: goal.models.dynamical.create_homogeneous_gaussian_markov_chain
