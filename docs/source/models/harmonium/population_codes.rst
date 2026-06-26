Population Codes
================

.. automodule:: goal.models.harmonium.population_codes
   :noindex:
   :no-members:

Class Hierarchy
---------------

.. inheritance-diagram:: goal.models.harmonium.population_codes
   :parts: 2
   :top-classes: goal.geometry.exponential_family.harmonium.Harmonium, goal.geometry.exponential_family.base.ExponentialFamily

\

Classes
-------

.. autoclass:: goal.models.harmonium.population_codes.PoissonVonMisesHarmonium
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.models.harmonium.population_codes.VonMisesPopulationCode
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.models.harmonium.population_codes.BoltzmannNormalHarmonium
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.models.harmonium.population_codes.BoltzmannPopulationCode
   :members:
   :undoc-members:
   :show-inheritance:

Factory Functions
-----------------

.. autofunction:: goal.models.harmonium.population_codes.poisson_mixture

.. autofunction:: goal.models.harmonium.population_codes.com_poisson_mixture

.. autofunction:: goal.models.harmonium.population_codes.chordal_boltzmann_population_code

.. autofunction:: goal.models.harmonium.population_codes.diagonal_boltzmann_population_code

.. seealso::

   :doc:`/examples` --- **population_codes**: Bayesian stimulus decoding from cosine-tuned neural responses; **poisson_mixture**: Poisson vs CoM-Poisson spike count mixtures; **chordal_boltzmann_ppc**: a chordal Boltzmann PPC that stays conjugate to a Gaussian latent while learning data.
