"""Structured Variational Inference for Harmoniums.

This example implements a VAE-like variational inference framework where the
approximate posterior has the same structural form as the true conjugate posterior,
but with a learnable input-independent correction term rho_Z.

Key insight: The approximate posterior takes the form:
    hat{theta}_{Z|X}(x) = theta_Z + Theta_{XZ}^T s_X(x) - rho_Z

This differs from the exact conjugate posterior only by the correction -rho_Z.

The ELBO objective is:
    L(x) = E_q[log p(x|z)] - D_KL(q_{Z|X=x} || p_Z)

where:
- The reconstruction term uses MC sampling for E_q[psi_X(theta_X + Theta_{XZ} s_Z(z))]
- The KL term is tractable when pst_man is Analytic
"""
