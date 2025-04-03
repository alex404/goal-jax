"""Curved exponential families with constrained parameterizations.

This module implements exponential families whose parameters are constrained to
lie on a submanifold of the full parameter space. The key operations are:

1. Embedding: Map from constrained (curved) natural parameters to full natural parameters
2. Projection: Map from full mean parameters to constrained (curved) mean parameters

These operations facilitate efficient learning and inference while maintaining the
mathematical structure of the exponential family.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self, cast, override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.base import Point
from ..manifold.combinators import Pair
from ..manifold.embedding import Embedding
from ..manifold.linear import AffineMap
from ..manifold.matrix import MatrixRep
from .base import (
    Differentiable,
    ExponentialFamily,
    Natural,
)
from .harmonium import DifferentiableConjugated


@dataclass(frozen=True)
class CurvedExponentialFamily[Complete: ExponentialFamily](ExponentialFamily, ABC):
    """
    An exponential family whose parameters are constrained to a submanifold.

    This class represents a curved parameterization where parameters are
    restricted to a submanifold of the full parameter space.
    """

    com_man: Complete

    # Fields
    @property
    @abstractmethod
    def embedding(self) -> Embedding[Complete, Self]:
        """Subspace relationship between full and curved parameters."""


@dataclass(frozen=True)
class ConjugatePair[
    IntRep: MatrixRep,
    Observable: Differentiable,
    IntObservable: ExponentialFamily,
    IntLatent: ExponentialFamily,
    PriorLatent: Differentiable,
](Pair[AffineMap[IntRep, IntLatent, IntObservable, Observable], PriorLatent], ABC):
    """Exponential family with curved latent parameters constrained to a submanifold."""

    # Contract

    @abstractmethod
    def conjugation_baseline(
        self,
        lkl_params: Point[
            Natural, AffineMap[IntRep, IntLatent, IntObservable, Observable]
        ],
    ) -> Array:
        """Compute the baseline parameters for conjugation."""

    # Methods

    def log_partition_function(self, params: Point[Natural, Self]) -> Array:
        """Compute the log partition function using conjugation parameters.

        The log partition function can be written as:

        $$\\psi(\\theta) = \\psi_Z(\\theta_Z + \\rho) + \\chi$$

        where $\\psi_Z$ is the log partition function of the latent exponential family.
        """
        # Split parameters
        lkl_params, prr_params = self.split_params(params)

        # Get conjugation parameters
        chi = self.conjugation_baseline(lkl_params)

        # Use latent family's partition function
        return self.snd_man.log_partition_function(prr_params) + chi

    def likelihood_at(
        self, params: Point[Natural, Self], z: Array
    ) -> Point[Natural, Observable]:
        """Compute natural parameters of likelihood distribution $p(x \\mid z)$ at a given $z$.

        Given a latent state $z$ with sufficient statistics $s(z)$, computes natural parameters $\\theta_X$ of the conditional distribution:

        $$p(x \\mid z) \\propto \\exp(\\theta_X \\cdot s_X(x) + s_X(x) \\cdot \\Theta_{XZ} \\cdot s_Z(z))$$
        """
        lkl_params, _ = self.split_params(params)
        mz = self.fst_man.dom_man.sufficient_statistic(z)
        return self.fst_man(lkl_params, mz)

    def sample(self, key: Array, params: Point[Natural, Self], n: int = 1) -> Array:
        key1, key2 = jax.random.split(key)
        _, prr_params = self.split_params(params)
        z_sample = self.snd_man.sample(key1, prr_params, n)

        # Vectorize sampling from conditional distributions
        x_params = jax.vmap(self.likelihood_at, in_axes=(None, 0))(params, z_sample)

        # Sample from conditionals p(x|z) in parallel
        x_sample = jax.vmap(self.fst_man.cod_sub.sup_man.sample, in_axes=(0, 0, None))(
            jax.random.split(key2, n), x_params, 1
        ).reshape((n, -1))

        # Concatenate samples along data dimension
        return jnp.concatenate([x_sample, z_sample], axis=-1)


@dataclass(frozen=True)
class ConjugateEmbedding[
    IntRep: MatrixRep,
    Observable: Differentiable,
    IntObservable: ExponentialFamily,
    IntLatent: ExponentialFamily,
    PriorLatent: Differentiable,
    PostLatent: Differentiable,
](
    Embedding[
        DifferentiableConjugated[
            IntRep, Observable, IntObservable, IntLatent, PostLatent
        ],
        ConjugatePair[IntRep, Observable, IntObservable, IntLatent, PriorLatent],
    ]
):
    """
    Embedding for curved harmoniums using conjugation relationship.
    """

    # Fields
    con_hrm: DifferentiableConjugated[
        IntRep, Observable, IntObservable, IntLatent, PostLatent
    ]
    con_par: ConjugatePair[IntRep, Observable, IntObservable, IntLatent, PriorLatent]
    lat_sub: Embedding[PostLatent, PriorLatent]

    @property
    @override
    def sup_man(
        self,
    ) -> DifferentiableConjugated[
        IntRep, Observable, IntObservable, IntLatent, PostLatent
    ]:
        return self.con_hrm

    @property
    @override
    def sub_man(
        self,
    ) -> ConjugatePair[IntRep, Observable, IntObservable, IntLatent, PriorLatent]:
        # Create the sub-manifold representing curved parameters
        return self.con_par

    @override
    def embed(
        self,
        p: Point[
            Natural,
            ConjugatePair[IntRep, Observable, IntObservable, IntLatent, PriorLatent],
        ],
    ) -> Point[
        Natural,
        DifferentiableConjugated[
            IntRep, Observable, IntObservable, IntLatent, PostLatent
        ],
    ]:
        """Embed curved parameters to full parameters using conjugation."""
        lkl_params, prr_params = self.con_par.split_params(p)
        lat_params0 = self.lat_sub.embed(prr_params)
        rho = self.con_hrm.conjugation_parameters(lkl_params)
        lat_params = lat_params0 - rho
        obs_params, int_params = self.con_hrm.lkl_man.split_params(lkl_params)
        return self.con_hrm.join_params(obs_params, int_params, lat_params)


@dataclass(frozen=True)
class CurvedHarmonium[
    IntRep: MatrixRep,
    Observable: Differentiable,
    IntObservable: ExponentialFamily,
    IntLatent: ExponentialFamily,
    PriorLatent: Differentiable,
    PostLatent: Differentiable,
    Complete: ExponentialFamily,
](
    ConjugatePair[IntRep, Observable, IntObservable, IntLatent, PriorLatent],
    CurvedExponentialFamily[Complete],
    ABC,
):
    """
    Harmonium with curved parameterization.

    This class implements a harmonium where the latent parameters
    are constrained to a submanifold of the full parameter space.
    """

    # Fields

    lat_sub: Embedding[PostLatent, PriorLatent]

    # Properties

    @property
    def _abc_hrm(
        self,
    ) -> DifferentiableConjugated[
        IntRep, Observable, IntObservable, IntLatent, PostLatent
    ]:
        return cast(
            DifferentiableConjugated[
                IntRep, Observable, IntObservable, IntLatent, PostLatent
            ],
            self.com_man,
        )

    @property
    @override
    def embedding(
        self,
    ) -> Embedding[Complete, Self]:
        return cast(
            Embedding[Complete, Self],
            ConjugateEmbedding(self._abc_hrm, self, self.lat_sub),
        )

    @override
    def conjugation_baseline(
        self,
        lkl_params: Point[
            Natural, AffineMap[IntRep, IntLatent, IntObservable, Observable]
        ],
    ) -> Array:
        """Implement the conjugation baseline using the full harmonium."""
        return self._abc_hrm.conjugation_baseline(lkl_params)
