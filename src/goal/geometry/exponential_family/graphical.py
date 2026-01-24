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
from ..manifold.linear import LinearMap
from .base import ExponentialFamily
from .harmonium import (
    AnalyticConjugated,
    DifferentiableConjugated,
    Harmonium,
    SymmetricConjugated,
)

### Helper Classes ###


@dataclass(frozen=True)
class ObservableEmbedding[
    Observable: ExponentialFamily,
    Posterior: ExponentialFamily,
](TupleEmbedding[Observable, Harmonium[Observable, Posterior]]):
    """Embedding of the observable manifold of a harmonium into the harmonium itself.

    This embedding projects a harmonium coordinates onto the observable component, or embeds an observable point into the full harmonium space by setting interaction and latent parameters to zero. Generally speaking, projection is only safe on mean coordinates, while embedding is only safe on natural coordinates.
    """

    # Fields

    hrm_man: Harmonium[Observable, Posterior]
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
    ) -> Harmonium[Observable, Posterior]:
        return self.hrm_man

    @property
    @override
    def sub_man(self) -> Observable:
        return self.hrm_man.obs_man


@dataclass(frozen=True)
class InteractionEmbedding[
    Observable: ExponentialFamily,
    Posterior: ExponentialFamily,
](TupleEmbedding[LinearMap[Posterior, Observable], Harmonium[Observable, Posterior]]):
    """Embedding of the observable manifold of a harmonium into the harmonium itself.

    This embedding projects a harmonium coordinates onto the observable component, or embeds an observable point into the full harmonium space by setting interaction and latent parameters to zero. Generally speaking, projection is only safe on mean coordinates, while embedding is only safe on natural coordinates.
    """

    # Fields

    hrm_man: Harmonium[Observable, Posterior]
    """The harmonium that contains the observable manifold."""

    # Overrides

    @property
    @override
    def tup_idx(
        self,
    ) -> int:
        return 1

    @property
    @override
    def amb_man(
        self,
    ) -> Harmonium[Observable, Posterior]:
        return self.hrm_man

    @property
    @override
    def sub_man(self) -> LinearMap[Posterior, Observable]:
        return self.hrm_man.int_man


@dataclass(frozen=True)
class PosteriorEmbedding[
    Observable: ExponentialFamily,
    Posterior: ExponentialFamily,
](TupleEmbedding[Posterior, Harmonium[Observable, Posterior]]):
    """Embedding of the posterior manifold of a harmonium into the harmonium itself.

    This embedding projects harmonium coordinates onto the posterior manifold, or embeds an latent point into the full harmonium space by setting interaction and observable parameters to zero. Generally speaking, projection is only safe on mean coordinates, while embedding is only safe on natural coordinates.
    """

    # Fields

    hrm_man: Harmonium[Observable, Posterior]
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
    ) -> Harmonium[Observable, Posterior]:
        return self.hrm_man

    @property
    @override
    def sub_man(self) -> Posterior:
        return self.hrm_man.pst_man


@dataclass(frozen=True)
class LatentHarmoniumEmbedding[
    PostHarmonium: Harmonium[Any, Any],
    PriorHarmonium: Harmonium[Any, Any],
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


### Hierarchical Harmonium Classes ###


@dataclass(frozen=True)
class DifferentiableHierarchical[
    LowerHarmonium: DifferentiableConjugated[Any, Any, Any],
    PstUpperHarmonium: DifferentiableConjugated[Any, Any, Any],
    PrrUpperHarmonium: DifferentiableConjugated[Any, Any, Any],
](
    DifferentiableConjugated[Any, PstUpperHarmonium, PrrUpperHarmonium],
):
    """Differentiable hierarchical conjugated harmonium.

    Composes a lower harmonium (Observable → MiddleLatent) with an upper harmonium
    (MiddleLatent → TopLatent) to form a hierarchical structure. Supports asymmetric
    posterior/prior structures where the prior uses a fuller parameterization.

    The joint distribution factors as: p(x,y,z) = p(x|y)p(y|z)p(z)

    Type Parameters:
        LowerHarmonium: The lower harmonium type (Observable → MiddleLatent)
        PstUpperHarmonium: Posterior upper harmonium type (possibly restricted)
        PrrUpperHarmonium: Prior upper harmonium type (for conjugation)
    """

    lwr_hrm: LowerHarmonium
    """Lower harmonium: maps observable to middle latent."""

    pst_upr_hrm: PstUpperHarmonium
    """Posterior upper harmonium with possibly restricted structure."""

    prr_upr_hrm: PrrUpperHarmonium
    """Prior upper harmonium for conjugation parameter computation."""

    def __post_init__(self) -> None:
        """Validate that component harmoniums are compatible."""
        assert self.lwr_hrm.pst_man == self.pst_upr_hrm.obs_man, (
            "Lower harmonium's posterior must match posterior upper harmonium's observable"
        )
        assert self.lwr_hrm.prr_man == self.prr_upr_hrm.obs_man, (
            "Lower harmonium's prior must match prior upper harmonium's observable"
        )

    @property
    @override
    def int_man(self) -> Any:
        """Interaction manifold: lower interaction prepended with upper observable embedding."""
        return self.lwr_hrm.int_man.prepend_embedding(
            ObservableEmbedding(self.pst_upr_hrm)
        )

    @property
    @override
    def pst_prr_emb(
        self,
    ) -> LatentHarmoniumEmbedding[PstUpperHarmonium, PrrUpperHarmonium]:
        """Embedding from posterior to prior latent space via LatentHarmoniumEmbedding."""
        return LatentHarmoniumEmbedding(
            self.lwr_hrm.pst_prr_emb,
            self.pst_upr_hrm,
            self.prr_upr_hrm,
        )

    @override
    def extract_likelihood_input(self, prr_sample: Array) -> Array:
        """Extract middle latent (y) from full prior sample (y,z) for likelihood evaluation."""
        return prr_sample[:, : self.lwr_hrm.prr_man.data_dim]

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Compute conjugation parameters for the hierarchical structure.

        Embeds the lower harmonium's conjugation parameters into the upper
        harmonium's parameter space via the observable embedding.
        """
        upr_obs_emb = ObservableEmbedding(self.prr_upr_hrm)
        return upr_obs_emb.embed(self.lwr_hrm.conjugation_parameters(lkl_params))


@dataclass(frozen=True)
class SymmetricHierarchical[
    LowerHarmonium: SymmetricConjugated[Any, Any],
    UpperHarmonium: DifferentiableConjugated[Any, Any, Any],
](
    SymmetricConjugated[Any, UpperHarmonium],
):
    """Symmetric hierarchical conjugated harmonium.

    Composes a lower harmonium with an upper harmonium where both use symmetric
    posterior/prior structure (posterior = prior manifold).

    The joint distribution factors as: p(x,y,z) = p(x|y)p(y|z)p(z)

    Type Parameters:
        LowerHarmonium: The lower harmonium type (must be symmetric)
        UpperHarmonium: The upper harmonium type
    """

    lwr_hrm: LowerHarmonium
    """Lower harmonium: maps observable to middle latent."""

    upr_hrm: UpperHarmonium
    """Upper harmonium: maps middle latent to top latent."""

    def __post_init__(self) -> None:
        """Validate that component harmoniums are compatible."""
        assert self.lwr_hrm.lat_man == self.upr_hrm.obs_man, (
            "Lower harmonium's latent must match upper harmonium's observable"
        )

    @property
    @override
    def lat_man(self) -> UpperHarmonium:
        """The latent manifold is the upper harmonium."""
        return self.upr_hrm

    @property
    @override
    def int_man(self) -> Any:
        """Interaction manifold: lower interaction prepended with upper observable embedding."""
        return self.lwr_hrm.int_man.prepend_embedding(ObservableEmbedding(self.upr_hrm))

    @override
    def extract_likelihood_input(self, prr_sample: Array) -> Array:
        """Extract middle latent (y) from full prior sample (y,z) for likelihood evaluation."""
        return prr_sample[:, : self.lwr_hrm.lat_man.data_dim]

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Compute conjugation parameters for the hierarchical structure."""
        upr_obs_emb = ObservableEmbedding(self.upr_hrm)
        return upr_obs_emb.embed(self.lwr_hrm.conjugation_parameters(lkl_params))


@dataclass(frozen=True)
class AnalyticHierarchical[
    LowerHarmonium: AnalyticConjugated[Any, Any],
    UpperHarmonium: AnalyticConjugated[Any, Any],
](
    AnalyticConjugated[Any, UpperHarmonium],
):
    """Analytic hierarchical conjugated harmonium.

    Composes a lower harmonium with an upper harmonium, enabling closed-form EM
    and bidirectional parameter conversion.

    The joint distribution factors as: p(x,y,z) = p(x|y)p(y|z)p(z)

    Type Parameters:
        LowerHarmonium: The lower harmonium type (must be analytic)
        UpperHarmonium: The upper harmonium type
    """

    lwr_hrm: LowerHarmonium
    """Lower harmonium: maps observable to middle latent."""

    upr_hrm: UpperHarmonium
    """Upper harmonium: maps middle latent to top latent."""

    def __post_init__(self) -> None:
        """Validate that component harmoniums are compatible."""
        assert self.lwr_hrm.lat_man == self.upr_hrm.obs_man, (
            "Lower harmonium's latent must match upper harmonium's observable"
        )

    @property
    @override
    def lat_man(self) -> UpperHarmonium:
        """The latent manifold is the upper harmonium."""
        return self.upr_hrm

    @property
    @override
    def int_man(self) -> Any:
        """Interaction manifold: lower interaction prepended with upper observable embedding."""
        return self.lwr_hrm.int_man.prepend_embedding(ObservableEmbedding(self.upr_hrm))

    @override
    def extract_likelihood_input(self, prr_sample: Array) -> Array:
        """Extract middle latent (y) from full prior sample (y,z) for likelihood evaluation."""
        return prr_sample[:, : self.lwr_hrm.lat_man.data_dim]

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Compute conjugation parameters for the hierarchical structure."""
        upr_obs_emb = ObservableEmbedding(self.upr_hrm)
        return upr_obs_emb.embed(self.lwr_hrm.conjugation_parameters(lkl_params))

    @override
    def to_natural_likelihood(self, means: Array) -> Array:
        """Convert mean parameters to natural likelihood parameters.

        Projects the hierarchical mean parameters down to the lower harmonium's
        space and converts them to natural parameters.
        """
        obs_means, lwr_int_means, lat_means = self.split_coords(means)
        upr_obs_emb = ObservableEmbedding(self.upr_hrm)
        lwr_lat_means = upr_obs_emb.project(lat_means)
        lwr_means = self.lwr_hrm.join_coords(obs_means, lwr_int_means, lwr_lat_means)
        return self.lwr_hrm.to_natural_likelihood(lwr_means)
