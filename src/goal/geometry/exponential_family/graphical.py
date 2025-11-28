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

from jax import Array

from ..manifold.embedding import LinearEmbedding, TupleEmbedding
from .base import ExponentialFamily
from .harmonium import (
    AnalyticConjugated,
    DifferentiableConjugated,
    Harmonium,
)

### Helper Classes ###


@dataclass(frozen=True)
class ObservableEmbedding[
    Observable: ExponentialFamily,
    Harm: Harmonium[Any, Any, Any, Any],
](TupleEmbedding[Observable, Harm]):
    """Embedding of the observable manifold of a harmonium into the harmonium itself.

    This embedding projects a harmonium coordinates onto the observable component, or embeds an observable point into the full harmonium space by setting interaction and latent parameters to zero. Generally speaking, projection is only safe on mean coordinates, while embedding is only safe on natural coordinates.
    """

    # Fields

    hrm_man: Harm
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
    ) -> Harm:
        return self.hrm_man

    @property
    @override
    def sub_man(self) -> Observable:
        return self.hrm_man.obs_man


@dataclass(frozen=True)
class PosteriorEmbedding[
    Posterior: ExponentialFamily,
    Harm: Harmonium[Any, Any, Any, Any],
](TupleEmbedding[Posterior, Harm]):
    """Embedding of the posterior manifold of a harmonium into the harmonium itself.

    This embedding projects harmonium coordinates onto the posterior manifold, or embeds an latent point into the full harmonium space by setting interaction and observable parameters to zero. Generally speaking, projection is only safe on mean coordinates, while embedding is only safe on natural coordinates.
    """

    # Fields

    hrm_man: Harm
    """The harmonium that contains the latent manifold."""

    # Overrides

    @property
    @override
    def tup_idx(
        self,
    ) -> int:
        return 2

    @property
    @override
    def amb_man(
        self,
    ) -> Harm:
        return self.hrm_man

    @property
    @override
    def sub_man(self) -> Posterior:
        return self.hrm_man.pst_man


@dataclass(frozen=True)
class LatentHarmoniumEmbedding[
    PostHarmonium: Harmonium[Any, Any, Any, Any],
    PriorHarmonium: Harmonium[Any, Any, Any, Any],
](LinearEmbedding[PostHarmonium, PriorHarmonium]):
    """Embedding of one harmonium into another via observable space embedding.

    This embedding connects two harmoniums that share the same interaction and latent structure
    but differ in their observable parameterization. It's used in hierarchical models where
    the posterior uses a restricted observable representation (e.g., diagonal covariance) for
    computational efficiency, while the prior uses a fuller representation for conjugation
    parameter computation.

    **Usage in hierarchical models**:

    In a hierarchical harmonium, the lower harmonium's latent manifold equals the upper
    harmonium's observable manifold. When the posterior and prior use different observable
    parameterizations, this embedding maps between them by:

    - Projecting prior parameters down to the restricted posterior observable space
    - Embedding posterior observable parameters up to the fuller prior observable space

    **Structure**:

    Both harmoniums must have:

    - The same interaction matrix representation and latent manifold
    - Observable manifolds :math:`\\text{Obs}_1 \\subseteq \\text{Obs}_2` where the posterior
      uses the restricted :math:`\\text{Obs}_1` and the prior uses the fuller :math:`\\text{Obs}_2`
    - An embedding :math:`e: \\text{Obs}_1 \\to \\text{Obs}_2` provided via `pst_obs_emb`

    **Parameters**:

    For harmonium points :math:`(o, i, l)` where :math:`o` is observable,
    :math:`i` is interaction, and :math:`l` is latent:

    - Projection: :math:`(o_2, i, l) \\mapsto (e^{-1}(o_2), i, l)` where :math:`e^{-1}` projects to :math:`\\text{Obs}_1`
    - Embedding: :math:`(o_1, i, l) \\mapsto (e(o_1), i, l)` where :math:`e` embeds into :math:`\\text{Obs}_2`

    **Runtime requirements**:

    - `pst_obs_emb` embeds `pst_hrm.obs_man` into `prr_hrm.obs_man`
    - `pst_hrm` and `prr_hrm` have matching interaction and latent structures
    """

    # Fields

    pst_obs_emb: LinearEmbedding[Any, Any]
    """The embedding of the constrained observable manifold into the unconstrained observable manifold."""

    pst_hrm: PostHarmonium
    """The harmonium that contains the constrained observable manifold."""

    prr_hrm: PriorHarmonium
    """The harmonium that contains the unconstrained observable manifold."""

    # Overrides

    @property
    @override
    def sub_man(self) -> PostHarmonium:
        return self.pst_hrm

    @property
    @override
    def amb_man(self) -> PriorHarmonium:
        return self.prr_hrm

    @override
    def project(self, coords: Array) -> Array:
        """Project harmonium parameters to restricted observable space.

        Args:
            p: Parameters in prior harmonium space (mean coordinates)

        Returns:
            Parameters in posterior harmonium space (mean coordinates)
        """
        obs_params, int_params, lat_params = self.amb_man.split_coords(coords)
        prj_obs_params = self.pst_obs_emb.project(obs_params)
        return self.sub_man.join_coords(prj_obs_params, int_params, lat_params)

    @override
    def embed(self, coords: Array) -> Array:
        """Embed posterior parameters into prior observable space.

        Args:
            p: Parameters in posterior harmonium space (natural coordinates)

        Returns:
            Parameters in prior harmonium space (natural coordinates)
        """
        obs_params, int_params, lat_params = self.sub_man.split_coords(coords)
        emb_obs_params = self.pst_obs_emb.embed(obs_params)
        return self.amb_man.join_coords(emb_obs_params, int_params, lat_params)


### Helper Functions ###


def hierarchical_conjugation_parameters(
    lwr_hrm: DifferentiableConjugated[Any, Any, Any, Any, Any],
    upr_hrm: Harmonium[Any, Any, Any, Any],
    lkl_params: Array,
) -> Array:
    """Compute conjugation parameters for hierarchical structure.

    This embeds the lower harmonium's conjugation parameters into the
    upper harmonium's parameter space. The embedding is always via the
    observable space of the upper harmonium, since in hierarchical models
    the lower prior space = upper observable space.

    Runtime requirement: lwr_hrm.prr_man == upr_hrm.obs_man

    Parameters
    ----------
    lwr_hrm : LowerHarmonium
        The lower harmonium in the hierarchy
    upr_hrm : UpperHarmonium
        The upper harmonium we're embedding into (observable = lower prior)
    lkl_params : Array
        Likelihood parameters from the lower harmonium (natural coordinates)

    Returns
    -------
    Array
        Conjugation parameters in the upper harmonium's space (natural coordinates)
    """
    assert lwr_hrm.prr_man == upr_hrm.obs_man, (
        "Lower harmonium's prior space must match upper harmonium's observable space"
    )

    hrm_lat_emb = ObservableEmbedding(upr_hrm)
    return hrm_lat_emb.embed(lwr_hrm.conjugation_parameters(lkl_params))


def hierarchical_to_natural_likelihood(
    harmonium: Harmonium[Any, Any, Any, Any],
    lwr_hrm: AnalyticConjugated[Any, Any, Any, Any],
    upr_hrm: Harmonium[Any, Any, Any, Any],
    params: Array,
) -> Array:
    """Convert mean parameters to natural likelihood parameters for hierarchical models.

    This projects the hierarchical mean parameters down to the lower harmonium's
    space and converts them to natural parameters.

    Parameters
    ----------
    harmonium : HrmType
        The full hierarchical harmonium
    lwr_hrm : AnalyticConjugated
        The lower harmonium in the hierarchy
    upr_hrm : ExponentialFamily
        The upper harmonium (for embedding construction)
    params : Array
        Mean parameters of the hierarchical model

    Returns
    -------
    Array
        Natural parameters for the lower likelihood
    """
    obs_means, lwr_int_means, lat_means = harmonium.split_coords(params)
    hrm_lat_emb = ObservableEmbedding(upr_hrm)
    lwr_lat_means = hrm_lat_emb.project(lat_means)
    lwr_means = lwr_hrm.join_coords(obs_means, lwr_int_means, lwr_lat_means)
    return lwr_hrm.to_natural_likelihood(lwr_means)
