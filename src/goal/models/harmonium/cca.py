"""Canonical correlation analysis: two observable Gaussians sharing one latent.

The first model in this library over a graph with **more than one root**. Where a linear
Gaussian model is a chain $x - z$, this is a fork

.. code-block:: text

       z
      / \\
     x   y

with $x$ and $y$ conditionally independent given $z$. That independence is what makes the
model tractable, and it is carried by the observable manifold rather than asserted: the
observable is a :class:`~goal.geometry.exponential_family.combinators.ExponentialFamilyPair`,
whose log-partition function is already the sum of its components'.

Mathematically, the conjugation equation's left-hand side therefore factorizes,

.. math::
    \\psi_{XY}(\\theta_{XY} + \\Theta \\cdot \\mathbf s_Z(z))
      = \\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z))
      + \\psi_Y(\\theta_Y + \\Theta_{YZ} \\cdot \\mathbf s_Z(z)),

and since each branch is a linear Gaussian model with its own conjugation, the joint
conjugation parameters are just their sum, $\\rho = \\rho_X + \\rho_Y$. The offset needs no
special handling for the same reason: the default $\\chi = \\psi_{XY}(\\theta_{XY})$ already
sums across the pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

from jax import Array

from ...geometry import (
    AnalyticPair,
    DifferentiableConjugated,
    LinearClique,
    PositiveDefinite,
    Rectangular,
)
from ..base.gaussian.normal import FullNormal, Normal, full_normal
from .lgm import (
    GeneralizedGaussianLocationEmbedding,
    NormalCovarianceEmbedding,
    NormalLGM,
)


@dataclass(frozen=True)
class NormalPair[FstRep: PositiveDefinite, SndRep: PositiveDefinite](
    AnalyticPair[Normal[FstRep], Normal[SndRep]]
):
    """Two normals over disjoint data slices, side by side.

    Two graph nodes, not one: a model may couple to each component separately, which is
    what a fork does.
    """

    # Fields

    fst_dim: int
    """Data dimension of the first normal."""

    fst_rep: FstRep
    """Covariance structure of the first normal."""

    snd_dim: int
    """Data dimension of the second normal."""

    snd_rep: SndRep
    """Covariance structure of the second normal."""

    # Overrides

    @property
    @override
    def fst_man(self) -> Normal[FstRep]:
        return Normal(self.fst_dim, self.fst_rep)

    @property
    @override
    def snd_man(self) -> Normal[SndRep]:
        return Normal(self.snd_dim, self.snd_rep)


@dataclass(frozen=True)
class CanonicalCorrelationAnalysis[
    FstRep: PositiveDefinite,
    SndRep: PositiveDefinite,
    PstRep: PositiveDefinite,
](
    DifferentiableConjugated[NormalPair[FstRep, SndRep], Normal[PstRep], FullNormal],
):
    """Two observable normals coupled through one shared Gaussian latent.

    Data points are the concatenation $[x, y, z]$, and the two observable blocks may have
    different dimensions and different covariance structures. Fitting is gradient-based:
    the model is :class:`~goal.geometry.exponential_family.harmonium.DifferentiableConjugated`
    rather than analytic, because inverting the conjugation sum branch-wise needs structure
    a fork does not supply.
    """

    # Fields

    fst_dim: int
    """Data dimension of the first observable."""

    fst_rep: FstRep
    """Covariance structure of the first observable."""

    snd_dim: int
    """Data dimension of the second observable."""

    snd_rep: SndRep
    """Covariance structure of the second observable."""

    lat_dim: int
    """Dimension of the shared latent."""

    pst_rep: PstRep
    """Covariance structure of the posterior latent."""

    # Overrides

    @property
    @override
    def cross_placements(self) -> tuple[tuple[tuple[int, ...], LinearClique], ...]:
        """One branch per root: $(x,z)$ and $(y,z)$, giving the fork $x - z - y$.

        Nodes $0$ and $1$ are the two observables and node $2$ the shared latent. The two
        roots come from the observable being a pair, so the levels come out $(2, 1)$. Each
        branch couples the two locations; that one branch reaches only its own side of the
        observable pair is derived from the graph, not declared here.
        """
        rect = Rectangular()
        return (
            self.cross_placement(rect, {0: self._branch_emb(0), 2: self._lat_emb}),
            self.cross_placement(rect, {1: self._branch_emb(1), 2: self._lat_emb}),
        )

    @property
    @override
    def obs_man(self) -> NormalPair[FstRep, SndRep]:
        """Override to construct directly from fields, avoiding circular dependency."""
        return NormalPair(self.fst_dim, self.fst_rep, self.snd_dim, self.snd_rep)

    @property
    @override
    def pst_man(self) -> Normal[PstRep]:
        return Normal(self.lat_dim, self.pst_rep)

    @property
    @override
    def pst_prr_emb(self) -> NormalCovarianceEmbedding[PstRep, PositiveDefinite]:
        return NormalCovarianceEmbedding(self.pst_man, full_normal(self.lat_dim))

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Sum the two branches' conjugation parameters.

        Each branch is a linear Gaussian model in its own right, so this delegates rather
        than deriving anything new. Summing is valid because the observable pair's
        log-partition function already splits across the branches.
        """
        obs_bias, int_params = self.lkl_fun_man.split_coords(lkl_params)
        fst_bias, snd_bias = self.obs_man.split_coords(obs_bias)
        fst_int, snd_int = self.int_man.coord_blocks(int_params)

        fst_lgm, snd_lgm = self.fst_lgm, self.snd_lgm
        rho_fst = fst_lgm.conjugation_parameters(
            fst_lgm.lkl_fun_man.join_coords(fst_bias, fst_int)
        )
        rho_snd = snd_lgm.conjugation_parameters(
            snd_lgm.lkl_fun_man.join_coords(snd_bias, snd_int)
        )
        return rho_fst + rho_snd

    # Methods

    @property
    def fst_lgm(self) -> NormalLGM[FstRep, PstRep]:
        """The first branch as a standalone linear Gaussian model."""
        return NormalLGM(self.fst_dim, self.fst_rep, self.lat_dim, self.pst_rep)

    @property
    def snd_lgm(self) -> NormalLGM[SndRep, PstRep]:
        """The second branch as a standalone linear Gaussian model."""
        return NormalLGM(self.snd_dim, self.snd_rep, self.lat_dim, self.pst_rep)

    # Private

    def _branch_emb(self, idx: int) -> GeneralizedGaussianLocationEmbedding[Any]:
        """What one branch's coupling uses inside that branch: its location."""
        branch = self.obs_man.fst_man if idx == 0 else self.obs_man.snd_man
        return GeneralizedGaussianLocationEmbedding(branch)

    @property
    def _lat_emb(self) -> GeneralizedGaussianLocationEmbedding[Normal[PstRep]]:
        """What every branch's coupling uses inside the shared latent: its location."""
        return GeneralizedGaussianLocationEmbedding(self.pst_man)
