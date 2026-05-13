"""Discrete hidden Markov model.

The HMM emission and transition are both Categorical-Categorical conjugated harmoniums, expressed directly as ``AnalyticMixture[Categorical]`` --- no specialized categorical-conjugation classes are needed.
"""

from dataclasses import dataclass
from typing import override

import jax
from jax import Array

from ...geometry.exponential_family.dynamical import (
    AnalyticLatentProcess,
    AnalyticTransition,
)
from ..base.categorical import Categorical
from ..harmonium.mixture import AnalyticMixture


@dataclass(frozen=True)
class HiddenMarkovModel(AnalyticLatentProcess[Categorical, Categorical]):
    """Discrete state-space model.

    The model is

    .. math::
        z_0 \\sim {\\rm Cat}(\\pi_0), \\quad
        z_t \\mid z_{t-1} \\sim {\\rm Cat}(A[z_{t-1}, :]), \\quad
        x_t \\mid z_t \\sim {\\rm Cat}(B[z_t, :])

    Composed of a ``Categorical`` prior over $z_0$, an ``AnalyticMixture[Categorical]`` emission for $p(x_t \\mid z_t)$, and an ``AnalyticTransition[Categorical]`` whose kernel is a square ``AnalyticMixture[Categorical]`` for $p(z_t \\mid z_{t-1})$.
    """

    n_obs: int
    n_states: int

    @property
    @override
    def lat_man(self) -> Categorical:
        return Categorical(self.n_states)

    @property
    @override
    def ems_hrm(self) -> AnalyticMixture[Categorical]:
        return AnalyticMixture(Categorical(self.n_obs), self.n_states)

    @property
    @override
    def trn_map(self) -> AnalyticTransition[Categorical]:
        kernel = AnalyticMixture(Categorical(self.n_states), self.n_states)
        return AnalyticTransition(kernel=kernel)

    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize prior, emission, and transition parameters with small random noise."""
        keys = jax.random.split(key, 3)
        prior_params = self.lat_man.initialize(keys[0], location, shape)
        ems_full = self.ems_hrm.initialize(keys[1], location, shape)
        ems_params = self.ems_hrm.likelihood_function(ems_full)
        trns_params = self.trn_map.initialize(keys[2], location, shape)
        return self.join_coords(prior_params, ems_params, trns_params)
