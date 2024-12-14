"""Core definitions for hierarchical harmoniums - three-layer exponential family models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import jax.numpy as jnp
from jax import Array

from .exponential_family import Backward, ExponentialFamily, Generative, Mean, Natural
from .harmonium import BackwardConjugated
from .linear import AffineMap
from .manifold import Point
from .rep.matrix import MatrixRep


@dataclass(frozen=True)
class HierarchicalHarmonium[
    Rep: MatrixRep,
    Observable: Generative,
    SubObservable: ExponentialFamily,
    SubLowLatentY: ExponentialFamily,
    LatentY: Backward,
    SubHighLatentY: ExponentialFamily,
    LatentZ: Backward,
    SubLatentZ: ExponentialFamily,
](
    BackwardConjugated[
        Rep,
        Observable,
        SubObservable,
        SubLowLatentY,
        BackwardConjugated[Rep, LatentY, SubHighLatentY, SubLatentZ, LatentZ],
    ],
):
    """A three-layer hierarchical harmonium composed of two harmoniums where the
    latent layer of the lower harmonium matches the observable layer of the upper one.

    Structure:
        - Lower harmonium: p(x|y) between X and Y
        - Upper harmonium: p(y|z) between Y and Z
        - Together forms: p(x,y,z) = p(x|y)p(y|z)p(z)
    """

    # Lower harmonium between X and Y
    lwr_man: BackwardConjugated[Rep, Observable, SubObservable, SubLowLatentY, LatentY]

    @property
    def upr_man(
        self,
    ) -> BackwardConjugated[Rep, LatentY, SubHighLatentY, SubLatentZ, LatentZ]:
        return self.snd_man

    def conjugation_parameters(
        self,
        lkl_params: Point[
            Natural, AffineMap[Rep, SubLowLatentY, SubObservable, Observable]
        ],
    ) -> tuple[
        Array,
        Point[
            Natural,
            BackwardConjugated[Rep, LatentY, SubHighLatentY, SubLatentZ, LatentZ],
        ],
    ]:
        """Compute lower conjugation parameters for the hierarchical model."""
        chi, rho0 = self.lwr_man.conjugation_parameters(lkl_params)
        with self.upr_man as um:
            lkl_rho = um.lkl_man.join_params(rho0, Point(jnp.zeros(um.int_man.dim)))
            rho = um.join_params(lkl_rho, Point(jnp.zeros(um.lat_man.dim)))

        # Combine the conjugation parameters
        return chi, rho

    def to_natural_likelihood(
        self, p: Point[Mean, Self]
    ) -> Point[Natural, AffineMap[Rep, SubLowLatentY, SubObservable, Observable]]:
        """Use mean parameters to compute natural parameters for the lower likelihood."""

        lkl_means, lat_means = self.split_params(p)
        ylkl_means = self.upr_man.split_params(lat_means)[0]
        ylat_means = self.upr_man.lkl_man.split_params(ylkl_means)[0]
        lwr_means = self.lwr_man.join_params(lkl_means, ylat_means)

        # Combine the likelihoods
        return self.lwr_man.to_natural_likelihood(lwr_means)
