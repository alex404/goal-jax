# Hierarchical Model Parameter Sweep Results

## Context
The full interaction model achieves NMI=0.71, R²=0.98 via tuned config. The hierarchical model uses a single rectangular interaction block X↔(Y,K) via `ObservableEmbedding`, forcing all X↔K information through Y. This file tracks sweeps to find the best hierarchical configuration.

## Results Table

| Run | n_latent | n_clusters | lr | kl_warmup | conj_weight | entropy_w | post_entropy_w | n_steps | NMI | R² | Purity | Recon | Notes |
|-----|----------|------------|------|-----------|-------------|-----------|----------------|---------|-----|-----|--------|-------|-------|
| H0 (smoke) | 128 | 10 | 3e-4 | 1000 | 0.0 | 1.0 | 1.0 | 1000 | 0.35 | 0.00 | 41.8% | 8.64 | Default params, short run |

## Phase 1: Baseline with full-model-optimal config

| Run | n_latent | n_clusters | lr | kl_warmup | conj_weight | entropy_w | post_entropy_w | n_steps | NMI | R² | Purity | Recon | Notes |
|-----|----------|------------|------|-----------|-------------|-----------|----------------|---------|-----|-----|--------|-------|-------|
| H1 | 128 | 30 | 1.5e-3 | 120000 | 0.1 | 5.0 | 10.0 | 300000 | 0.57 | 0.996 | 76.8% | 3.34 | Baseline from full-model config |

## Phase 2: Latent dimension

| Run | n_latent | n_clusters | lr | kl_warmup | conj_weight | entropy_w | post_entropy_w | n_steps | NMI | R² | Purity | Recon | Notes |
|-----|----------|------------|------|-----------|-------------|-----------|----------------|---------|-----|-----|--------|-------|-------|
| H2 | 64 | 30 | 1.5e-3 | 120000 | 0.1 | 5.0 | 10.0 | 300000 | 0.56 | 0.997 | 76.9% | 4.72 | Similar to H1 |
| H3 | 256 | 30 | 1.5e-3 | 120000 | 0.1 | 5.0 | 10.0 | 300000 | 0.55 | 0.995 | 76.3% | 2.93 | Worse NMI, skip H4(512) |

## Phase 3: Cluster count

| Run | n_latent | n_clusters | lr | kl_warmup | conj_weight | entropy_w | post_entropy_w | n_steps | NMI | R² | Purity | Recon | Notes |
|-----|----------|------------|------|-----------|-------------|-----------|----------------|---------|-----|-----|--------|-------|-------|
| H5 | 128 | 10 | 1.5e-3 | 120000 | 0.1 | 5.0 | 10.0 | 300000 | 0.50 | 0.997 | 57.9% | 3.36 | Fewer clusters hurts NMI |
| H6 | 128 | 20 | 1.5e-3 | 120000 | 0.1 | 5.0 | 10.0 | 300000 | 0.55 | 0.996 | 69.9% | 3.38 | Improving |
| H7 | 128 | 40 | 1.5e-3 | 120000 | 0.1 | 5.0 | 10.0 | 300000 | 0.56 | 0.997 | 77.7% | 3.25 | Similar NMI to H1, higher purity |

## Phase 4: Learning rate & warmup

| Run | n_latent | n_clusters | lr | kl_warmup | conj_weight | entropy_w | post_entropy_w | n_steps | NMI | R² | Purity | Recon | Notes |
|-----|----------|------------|------|-----------|-------------|-----------|----------------|---------|-----|-----|--------|-------|-------|
| H8 | 128 | 30 | 1e-3 | 120000 | 0.1 | 5.0 | 10.0 | 300000 | 0.55 | 0.996 | 74.0% | 3.38 | Slower LR hurts |
| H9 | 128 | 30 | 2e-3 | 120000 | 0.1 | 5.0 | 10.0 | 300000 | 0.56 | 0.997 | 75.9% | 3.33 | Slightly worse than H1 |
| H10 | 128 | 30 | 1.5e-3 | 50000 | 0.1 | 5.0 | 10.0 | 300000 | 0.56 | 0.996 | 75.9% | 3.30 | Shorter warmup similar |
| H11 | 128 | 30 | 1.5e-3 | 150000 | 0.1 | 5.0 | 10.0 | 300000 | 0.55 | 0.996 | 74.6% | 3.31 | Longer warmup slightly worse |

## Phase 5: Conjugation weight

| Run | n_latent | n_clusters | lr | kl_warmup | conj_weight | entropy_w | post_entropy_w | n_steps | NMI | R² | Purity | Recon | Notes |
|-----|----------|------------|------|-----------|-------------|-----------|----------------|---------|-----|-----|--------|-------|-------|
| H12 | 128 | 30 | 1.5e-3 | 120000 | 0.0 | 5.0 | 10.0 | 300000 | 0.59 | 0.598 | 78.7% | 3.28 | Best NMI! But R² collapsed |
| H13 | 128 | 30 | 1.5e-3 | 120000 | 0.05 | 5.0 | 10.0 | 300000 | 0.56 | 0.994 | 74.9% | 3.13 | Good balance |
| H14 | 128 | 30 | 1.5e-3 | 120000 | 0.2 | 5.0 | 10.0 | 300000 | 0.55 | 0.998 | 73.4% | 3.57 | Too much conj hurts NMI |

## Phase 6: Entropy weights

| Run | n_latent | n_clusters | lr | kl_warmup | conj_weight | entropy_w | post_entropy_w | n_steps | NMI | R² | Purity | Recon | Notes |
|-----|----------|------------|------|-----------|-------------|-----------|----------------|---------|-----|-----|--------|-------|-------|
| H15 | 128 | 30 | 1.5e-3 | 120000 | 0.1 | 1.0 | 1.0 | 300000 | ~0.55 | ~0.99 | — | — | Killed mid-run (~step 115k), tracking below H1 |
| H16 | 128 | 30 | 1.5e-3 | 120000 | 0.1 | 10.0 | 20.0 | 300000 | — | — | — | — | Killed before meaningful progress |

## Best Configuration

**H1** remains the best balanced configuration:
```
--interaction hierarchical --n-latent 128 --n-clusters 30 --learning-rate 1.5e-3
--kl-warmup-steps 120000 --conj-weight 0.1 --entropy-weight 5 --posterior-entropy-weight 10
```

## Key Findings

1. **Insensitive to most hyperparameters**: NMI stays in the 0.55-0.57 range across wide variations in latent dim (64-256), LR (1e-3 to 2e-3), warmup (50k-150k), and cluster count (20-40). The model is remarkably stable.

2. **Conjugation weight is the most impactful knob**: Removing it entirely (H12, conj=0.0) gives the best NMI (0.59) but R² collapses to 0.60 — the regression approximation ψ_X ≈ ρ·s(x) breaks down. Higher conj weight (0.2) preserves R² but suppresses NMI. conj=0.1 is the sweet spot for balanced performance.

3. **NMI ceiling around 0.57-0.59**: The hierarchical model plateaus well below the full model's 0.71. This is the expected consequence of lacking direct X↔K coupling — all cluster information must flow through the continuous latent Y.

4. **Reconstruction is comparable**: ~3.3 vs full model's similar range. The bottleneck is clustering, not generation.

## Comparison: Hierarchical vs Full

| Metric | Hierarchical Best (H1) | Full Best |
|--------|----------------------|-----------|
| NMI | 0.57 | 0.71 |
| R² | 0.996 | 0.98 |
| Purity | 76.8% | ~82% |
| Recon | 3.34 | ~3.3 |

The hierarchical model achieves ~80% of the full model's NMI. The gap confirms that direct X↔K coupling (the xk interaction block in the full model) provides meaningful clustering signal that can't be recovered by routing through Y alone.
