"""Hierarchical harmoniums: stacked conjugated harmoniums for multi-level latent variable models.

A hierarchical harmonium composes a lower harmonium (observable \\to middle latent) with an upper harmonium (middle latent \\to top latent), forming a joint distribution \\(p(x,y,z) = p(x|y)p(y|z)p(z)\\). Conjugation, natural parameters, and sampling are implemented recursively through the hierarchy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, override

from jax import Array

from ..manifold.embedding import LinearEmbedding, TupleEmbedding
from ..manifold.linear import LinearMap
from .base import Gibbs
from .harmonium import (
    AnalyticConjugated,
    DifferentiableConjugated,
    Harmonium,
    SymmetricConjugated,
)

### Helper Classes ###


@dataclass(frozen=True)
class ObservableEmbedding[
    Observable: Gibbs,
    Posterior: Gibbs,
](TupleEmbedding[Observable, Harmonium[Observable, Posterior]]):
    """Embeds the observable manifold of a harmonium into the full harmonium space (index 0).

    Projection extracts observable parameters from harmonium coordinates; embedding sets interaction and latent parameters to zero.
    """

    # Fields

    hrm_man: Harmonium[Observable, Posterior]

    # Overrides

    @property
    @override
    def tup_idx(self) -> int:
        return 0

    @property
    @override
    def amb_man(self) -> Harmonium[Observable, Posterior]:
        return self.hrm_man

    @property
    @override
    def sub_man(self) -> Observable:
        return self.hrm_man.obs_man


@dataclass(frozen=True)
class InteractionEmbedding[
    Observable: Gibbs,
    Posterior: Gibbs,
](TupleEmbedding[LinearMap[Posterior, Observable], Harmonium[Observable, Posterior]]):
    """Embeds the interaction manifold of a harmonium into the full harmonium space (index 1).

    Projection extracts interaction parameters from harmonium coordinates; embedding sets observable and latent parameters to zero.
    """

    # Fields

    hrm_man: Harmonium[Observable, Posterior]

    # Overrides

    @property
    @override
    def tup_idx(self) -> int:
        return 1

    @property
    @override
    def amb_man(self) -> Harmonium[Observable, Posterior]:
        return self.hrm_man

    @property
    @override
    def sub_man(self) -> LinearMap[Posterior, Observable]:
        return self.hrm_man.int_man


@dataclass(frozen=True)
class PosteriorEmbedding[
    Observable: Gibbs,
    Posterior: Gibbs,
](TupleEmbedding[Posterior, Harmonium[Observable, Posterior]]):
    """Embeds the posterior manifold of a harmonium into the full harmonium space (index 2).

    Projection extracts posterior parameters from harmonium coordinates; embedding sets observable and interaction parameters to zero.
    """

    # Fields

    hrm_man: Harmonium[Observable, Posterior]

    # Overrides

    @property
    @override
    def tup_idx(self) -> int:
        return 2

    @property
    @override
    def amb_man(self) -> Harmonium[Observable, Posterior]:
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
    """Embeds one harmonium into another by embedding only the observable component.

    Used in hierarchical models where the posterior uses a restricted observable representation (e.g., diagonal covariance) while the prior uses a fuller one. Interaction and latent components pass through unchanged.

    Mathematically, for harmonium points \\((o, i, l)\\): embedding maps \\((o_1, i, l) \\mapsto (e(o_1), i, l)\\) and projection maps \\((o_2, i, l) \\mapsto (e^{-1}(o_2), i, l)\\).
    """

    # Fields

    pst_obs_emb: LinearEmbedding[Any, Any]
    """Embedding of the restricted observable manifold into the full observable manifold."""

    pst_hrm: PostHarmonium
    """Harmonium with the restricted observable manifold."""

    prr_hrm: PriorHarmonium
    """Harmonium with the full observable manifold."""

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
        obs_params, int_params, lat_params = self.amb_man.split_coords(coords)
        prj_obs_params = self.pst_obs_emb.project(obs_params)
        return self.sub_man.join_coords(prj_obs_params, int_params, lat_params)

    @override
    def embed(self, coords: Array) -> Array:
        obs_params, int_params, lat_params = self.sub_man.split_coords(coords)
        emb_obs_params = self.pst_obs_emb.embed(obs_params)
        return self.amb_man.join_coords(emb_obs_params, int_params, lat_params)


# Hierarchical Harmonium Classes


class DifferentiableHierarchical[
    LowerHarmonium: DifferentiableConjugated[Any, Any, Any],
    PstUpperHarmonium: DifferentiableConjugated[Any, Any, Any],
    PrrUpperHarmonium: DifferentiableConjugated[Any, Any, Any],
](
    DifferentiableConjugated[Any, PstUpperHarmonium, PrrUpperHarmonium],
    ABC,
):
    """Differentiable hierarchical harmonium with asymmetric posterior/prior structure.

    Composes a lower harmonium (observable \\to middle latent) with an upper harmonium
    (middle latent \\to top latent). The prior may use a fuller parameterization than
    the posterior for conjugation.
    """

    # Contract

    @property
    @abstractmethod
    def lwr_hrm(self) -> LowerHarmonium:
        """Lower harmonium: observable \\to middle latent."""
        ...

    @property
    @abstractmethod
    def pst_upr_hrm(self) -> PstUpperHarmonium:
        """Posterior upper harmonium (possibly restricted)."""
        ...

    @property
    @abstractmethod
    def prr_upr_hrm(self) -> PrrUpperHarmonium:
        """Prior upper harmonium (for conjugation)."""
        ...

    # Overrides

    @property
    @override
    def int_man(self) -> Any:
        return self.lwr_hrm.int_man.prepend_embedding(
            ObservableEmbedding(self.pst_upr_hrm)
        )

    @property
    @override
    def pst_prr_emb(
        self,
    ) -> LinearEmbedding[PstUpperHarmonium, PrrUpperHarmonium]:
        return LatentHarmoniumEmbedding(
            self.lwr_hrm.pst_prr_emb,
            self.pst_upr_hrm,
            self.prr_upr_hrm,
        )

    @override
    def extract_likelihood_input(self, prr_sample: Array) -> Array:
        return prr_sample[:, : self.lwr_hrm.prr_man.data_dim]

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Embeds the lower harmonium's conjugation parameters into the upper harmonium's observable space."""
        upr_obs_emb = ObservableEmbedding(self.prr_upr_hrm)
        return upr_obs_emb.embed(self.lwr_hrm.conjugation_parameters(lkl_params))


class SymmetricHierarchical[
    LowerHarmonium: SymmetricConjugated[Any, Any],
    UpperHarmonium: DifferentiableConjugated[Any, Any, Any],
](
    SymmetricConjugated[Any, UpperHarmonium],
    DifferentiableHierarchical[LowerHarmonium, UpperHarmonium, UpperHarmonium],
    ABC,
):
    """Symmetric hierarchical harmonium where the lower harmonium has matching posterior and prior manifolds.

    Mirrors ``SymmetricConjugated`` at the hierarchical level: collapses the posterior
    and prior upper harmoniums into a single ``upr_hrm``, providing
    ``pst_upr_hrm = prr_upr_hrm = upr_hrm``.
    """

    # Contract

    @property
    @abstractmethod
    def upr_hrm(self) -> UpperHarmonium:
        """Upper harmonium: middle latent \\to top latent."""
        ...

    # Overrides

    @property
    @override
    def lat_man(self) -> UpperHarmonium:
        return self.upr_hrm

    @property
    @override
    def pst_upr_hrm(self) -> UpperHarmonium:
        return self.upr_hrm

    @property
    @override
    def prr_upr_hrm(self) -> UpperHarmonium:
        return self.upr_hrm


class AnalyticHierarchical[
    LowerHarmonium: AnalyticConjugated[Any, Any],
    UpperHarmonium: AnalyticConjugated[Any, Any],
](
    SymmetricHierarchical[LowerHarmonium, UpperHarmonium],
    AnalyticConjugated[Any, UpperHarmonium],
    ABC,
):
    """Analytic hierarchical harmonium enabling closed-form EM and bidirectional parameter conversion."""

    # Overrides

    @override
    def to_natural_likelihood(self, means: Array) -> Array:
        """Projects hierarchical mean parameters down to the lower harmonium's space and converts to natural likelihood parameters."""
        obs_means, lwr_int_means, lat_means = self.split_coords(means)
        upr_obs_emb = ObservableEmbedding(self.upr_hrm)
        lwr_lat_means = upr_obs_emb.project(lat_means)
        lwr_means = self.lwr_hrm.join_coords(obs_means, lwr_int_means, lwr_lat_means)
        return self.lwr_hrm.to_natural_likelihood(lwr_means)
