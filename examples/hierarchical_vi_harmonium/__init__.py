"""Hierarchical Variational Inference for Mixture of VonMises Clustering.

This example implements hierarchical variational inference for MNIST digit clustering
using a three-level generative model:

**Model Structure:**
    - Observable (X): Binomial (MNIST pixels)
    - Latent (Z): Product of VonMises (continuous circular representations)
    - Latent (K): Categorical (cluster assignments)

**Generative Model:**
    p(x, z, k) = p(k) * p(z|k) * p(x|z)

    where:
    - p(k) = Categorical(pi)
    - p(z|k) = ProductVonMises(theta_Z + Theta_ZK * e_k)  (mixture of VonMises)
    - p(x|z) = Binomial(n, sigmoid(theta_X + Theta_XZ * s_Z(z)))

**Key insight:** X is conditionally independent of K given Z, so the interaction
Theta_{X,ZK} only connects X to Z, not to K.

**Recognition Model:**
    theta_hat_{ZK|X}(x) = theta_{ZK} + Theta_{X,ZK}^T s_X(x) - [rho_Z, 0, 0]

    where rho_Z is a learnable correction term that only affects the VonMises bias,
    not the mixture interaction or categorical parameters.
"""
