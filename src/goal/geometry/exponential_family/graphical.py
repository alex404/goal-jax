"""Hierarchical harmoniums enable modeling complex, multi-level dependencies by stacking harmonium models. This stacked structure supports deep representations where information flows between multiple levels of latent variables, while maintaining analytical tractability through conjugate relationships.

In theory, a hierarchical harmonium is a conjugated harmonium where the latent manifold is itself a conjugated harmonium. This creates a structured joint distribution:

- A lower harmonium provides likelihood $p(x|y)$ between observable $x$ and first latent $y$
- An upper harmonium provides likelihood $p(y|z)$ between first latent $y$ and second latent $z$
- Together they form a joint distribution $p(x,y,z) = p(x|y)p(y|z)p(z)$

Key algorithms (conjugation, natural parameters, sampling) are implemented recursively through the model hierarchy, enabling efficient inference and learning despite the model's complexity.

Note: Implementation of these models pushes the boundaries of Python's type system, requiring careful handling of generic types to maintain proper relationships between layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.base import Point
from ..manifold.embedding import LinearEmbedding, TupleEmbedding
from ..manifold.linear import AffineMap
from ..manifold.matrix import MatrixRep
from .base import ExponentialFamily, Mean, Natural
from .harmonium import (
    AnalyticConjugated,
    DifferentiableConjugated,
    Harmonium,
)

### Helper Classes ###


@dataclass(frozen=True)
class ObservableEmbedding[
    Rep: MatrixRep,
    Observable: ExponentialFamily,
    IntObservable: ExponentialFamily,
    IntLatent: ExponentialFamily,
    PostLatent: ExponentialFamily,
    Latent: ExponentialFamily,
](
    TupleEmbedding[
        Observable,
        Harmonium[Rep, Observable, IntObservable, IntLatent, PostLatent, Latent],
    ]
):
    """Embedding of the observable manifold of a harmonium in to the harmonium itself."""

    # Fields

    hrm_man: Harmonium[Rep, Observable, IntObservable, IntLatent, PostLatent, Latent]
    """The harmonium that contains the observable manifold."""

    # Overrides

    @property
    @override
    def tup_idx(
        self,
    ) -> int:
        return 0

    @property
    @override
    def amb_man(
        self,
    ) -> Harmonium[Rep, Observable, IntObservable, IntLatent, PostLatent, Latent]:
        return self.hrm_man

    @property
    @override
    def sub_man(self) -> Observable:
        return self.hrm_man.obs_man


@dataclass(frozen=True)
class LatentHarmoniumEmbedding[
    Rep: MatrixRep,
    Observable: ExponentialFamily,
    PostObservable: ExponentialFamily,
    IntObservable: ExponentialFamily,
    IntLatent: ExponentialFamily,
    PostLatent: ExponentialFamily,
    Latent: ExponentialFamily,
](
    LinearEmbedding[
        Harmonium[Rep, PostObservable, IntObservable, IntLatent, PostLatent, Latent],
        Harmonium[Rep, Observable, IntObservable, IntLatent, PostLatent, Latent],
    ]
):
    """Embedding of one harmonium into another, where the observable space of the former can be embedded into the latter. This is designed for hierarchical harmoniums with constrained posterior spaces."""

    # Fields

    pst_obs_emb: LinearEmbedding[PostObservable, Observable]
    """The embedding of the constrained observable manifold into the unconstrained observable manifold."""

    hrm: Harmonium[Rep, PostObservable, IntObservable, IntLatent, PostLatent, Latent]
    """The harmonium that contains the contrained observable manifold."""

    con_hrm: Harmonium[Rep, Observable, IntObservable, IntLatent, PostLatent, Latent]
    """The harmonium that contains the unconstrained observable manifold."""

    # Overrides

    @property
    @override
    def sub_man(
        self,
    ) -> Harmonium[Rep, PostObservable, IntObservable, IntLatent, PostLatent, Latent]:
        return self.hrm

    @property
    @override
    def amb_man(
        self,
    ) -> Harmonium[Rep, Observable, IntObservable, IntLatent, PostLatent, Latent]:
        return self.con_hrm

    @override
    def project(
        self,
        p: Point[
            Mean,
            Harmonium[Rep, Observable, IntObservable, IntLatent, PostLatent, Latent],
        ],
    ) -> Point[
        Mean,
        Harmonium[Rep, PostObservable, IntObservable, IntLatent, PostLatent, Latent],
    ]:
        obs_params, int_params, lat_params = self.amb_man.split_params(p)
        prj_obs_params = self.pst_obs_emb.project(obs_params)
        return self.sub_man.join_params(prj_obs_params, int_params, lat_params)

    @override
    def embed(
        self,
        p: Point[
            Natural,
            Harmonium[
                Rep, PostObservable, IntObservable, IntLatent, PostLatent, Latent
            ],
        ],
    ) -> Point[
        Natural,
        Harmonium[Rep, Observable, IntObservable, IntLatent, PostLatent, Latent],
    ]:
        obs_params, int_params, lat_params = self.sub_man.split_params(p)
        emb_obs_params = self.pst_obs_emb.embed(obs_params)
        return self.amb_man.join_params(emb_obs_params, int_params, lat_params)


### Helper Functions ###


def hierarchical_sample[
    HrmType: Harmonium,  # pyright: ignore[reportMissingTypeArgument]
    LatType: ExponentialFamily,
](
    harmonium: HrmType,
    lwr_hrm: DifferentiableConjugated,  # pyright: ignore[reportMissingTypeArgument]
    con_lat_man: LatType,
    key: Array,
    params: Point[Natural, HrmType],
    n: int = 1,
) -> Array:
    """Sample from a hierarchical harmonium structure.

    This implements ancestral sampling:
    1. Sample from the prior p(z) in the upper latent space
    2. Extract the lower latent samples y from the upper samples yz
    3. Sample from the conditional p(x|y) for each latent sample

    Parameters
    ----------
    harmonium : HrmType
        The full hierarchical harmonium
    lwr_hrm : DifferentiableConjugated
        The lower harmonium in the hierarchy
    con_lat_man : LatType
        The conjugated latent manifold (upper harmonium's observable space)
    key : Array
        JAX random key
    params : Point[Natural, HrmType]
        Natural parameters of the hierarchical model
    n : int
        Number of samples to generate

    Returns
    -------
    Array
        Samples from the joint distribution, concatenated as [x, y, z]
    """
    # Split up the sampling key
    key1, key2 = jax.random.split(key)

    # Sample from adjusted latent distribution p(z)
    nat_prior = harmonium.prior(params)  # pyright: ignore[reportAttributeAccessIssue]
    yz_sample = con_lat_man.sample(key1, nat_prior, n)  # pyright: ignore[reportAttributeAccessIssue]
    y_sample = yz_sample[:, : lwr_hrm.con_lat_man.data_dim]

    # Vectorize sampling from conditional distributions
    x_params = jax.vmap(harmonium.likelihood_at, in_axes=(None, 0))(params, y_sample)

    # Sample from conditionals p(x|z) in parallel
    x_sample = jax.vmap(harmonium.obs_man.sample, in_axes=(0, 0, None))(
        jax.random.split(key2, n), x_params, 1
    ).reshape((n, -1))

    # Concatenate samples along data dimension
    return jnp.concatenate([x_sample, yz_sample], axis=-1)


def hierarchical_conjugation_parameters[
    UpperHrm: ExponentialFamily,
](
    lwr_hrm: DifferentiableConjugated,  # pyright: ignore[reportMissingTypeArgument]
    target_hrm: UpperHrm,
    lkl_params: Point[Natural, AffineMap],  # pyright: ignore[reportMissingTypeArgument]
) -> Point[Natural, UpperHrm]:
    """Compute conjugation parameters for hierarchical structure.

    This embeds the lower harmonium's conjugation parameters into the
    upper harmonium's parameter space. The embedding is always via the
    observable space of the upper harmonium, since in hierarchical models
    the lower latent space = upper observable space.

    Parameters
    ----------
    lwr_hrm : DifferentiableConjugated
        The lower harmonium in the hierarchy
    target_hrm : UpperHrm
        The upper harmonium we're embedding into
    lkl_params : Point
        Likelihood parameters from the lower harmonium

    Returns
    -------
    Point[Natural, UpperHrm]
        Conjugation parameters in the upper harmonium's space
    """
    hrm_lat_emb = ObservableEmbedding(target_hrm)  # pyright: ignore[reportArgumentType]
    return hrm_lat_emb.embed(lwr_hrm.conjugation_parameters(lkl_params))  # pyright: ignore[reportReturnType]


def hierarchical_to_natural_likelihood[
    HrmType: Harmonium,  # pyright: ignore[reportMissingTypeArgument]
](
    harmonium: HrmType,
    lwr_hrm: AnalyticConjugated,  # pyright: ignore[reportMissingTypeArgument]
    target_hrm: ExponentialFamily,
    params: Point[Mean, HrmType],
) -> Point[Natural, AffineMap]:  # pyright: ignore[reportMissingTypeArgument]
    """Convert mean parameters to natural likelihood parameters for hierarchical models.

    This projects the hierarchical mean parameters down to the lower harmonium's
    space and converts them to natural parameters.

    Parameters
    ----------
    harmonium : HrmType
        The full hierarchical harmonium
    lwr_hrm : AnalyticConjugated
        The lower harmonium in the hierarchy
    target_hrm : ExponentialFamily
        The upper harmonium (for embedding construction)
    params : Point[Mean, HrmType]
        Mean parameters of the hierarchical model

    Returns
    -------
    Point[Natural, AffineMap]
        Natural parameters for the lower likelihood
    """
    obs_means, lwr_int_means, lat_means = harmonium.split_params(params)
    hrm_lat_emb = ObservableEmbedding(target_hrm)  # pyright: ignore[reportArgumentType]
    lwr_lat_means = hrm_lat_emb.project(lat_means)  # pyright: ignore[reportArgumentType]
    lwr_means = lwr_hrm.join_params(obs_means, lwr_int_means, lwr_lat_means)
    return lwr_hrm.to_natural_likelihood(lwr_means)
