# Hyperparameter Exploration Results

All runs: `--mode gradient --interaction full`, ent-prior=5, ent-post=10

## Results Table

### Phase 1-2 (Runs 1-24): Baseline exploration

| Run | Steps | LR | KL-Warmup | LR-Warmup | Conj | Latent | Clusters | NMI | R² | Active | ELBO | Notes |
|-----|-------|----|-----------|-----------|------|--------|----------|-----|-----|--------|------|-------|
| 1 | 50K | 1e-3 | 5000 | 500 | 0.1 | 128 | 10 | 0.60 | 0.23 | 8 | -842 | Unified beta baseline |
| 2 | 50K | 1e-3 | 5000 | 500 | 0.0 | 128 | 10 | 0.63 | 0.07 | 8 | -623 | No conj (pure ELBO) |
| 3 | 50K | 1e-3 | 10000 | 500 | 0.1 | 128 | 10 | 0.59 | 0.30 | 8 | -836 | Long warmup |
| 4 | 50K | 1e-3 | 2000 | 500 | 0.1 | 128 | 10 | 0.61 | 0.25 | 8 | -836 | Short warmup |
| 5 | 100K | 1e-3 | 10000 | 500 | 0.1 | 128 | 10 | 0.59 | 0.61 | 8 | -823 | More steps, long warmup |
| 6 | 100K | 1e-3 | 20000 | 500 | 0.1 | 128 | 10 | 0.59 | 0.60 | 8 | -821 | Very long warmup |
| 7 | 50K | 1e-3 | 5000 | 500 | 0.1 | 128 | 10 | 0.62 | 0.24 | 8 | -846 | Minimal entropy |
| 8 | 50K | 1e-3 | 5000 | 500 | 0.1 | 128 | 10 | 0.59 | 0.27 | 9 | -837 | No entropy |
| 9 | 50K | 1e-3 | 10000 | 500 | 0.1 | 128 | 10 | 0.59 | 0.37 | 8 | -850 | Low prior, mod post |
| 10 | 50K | 1e-3 | 10000 | 500 | 0.1 | 128 | 10 | 0.59 | 0.23 | 9 | -849 | No prior ent, high post |
| 11 | 50K | 1e-3 | 5000 | 500 | 0.1 | 128 | 15 | 0.64 | 0.19 | 10 | -810 | Mild overclustering |
| 12 | 50K | 1e-3 | 5000 | 500 | 0.1 | 128 | 20 | 0.67 | 0.19 | 12 | -796 | 20 clusters |
| 13 | 100K | 1e-3 | 10000 | 500 | 0.1 | 128 | 20 | 0.67 | 0.51 | 12 | -776 | 20 clusters + 100K |
| 14 | 50K | 1e-3 | 5000 | 500 | 0.1 | 128 | 20 | 0.67 | 0.21 | 15 | -814 | 20 clusters, no prior ent |
| 15 | 50K | 2e-3 | 5000 | 500 | 0.1 | 128 | 10 | 0.57 | 0.51 | 7 | -830 | Higher LR |
| 16 | 50K | 3e-3 | 5000 | 500 | 0.1 | 128 | 10 | 0.55 | 0.60 | 6 | -816 | LR=3e-3, cluster collapse |
| 17 | 50K | 1e-3 | 5000 | 500 | 0.5 | 128 | 10 | 0.59 | 0.30 | 7 | -1014 | Stronger conj |
| 18 | 50K | 1e-3 | 5000 | 500 | 1.0 | 128 | 10 | 0.61 | 0.31 | 8 | -1105 | Strong conj |
| 19 | 100K | 2e-3 | 10000 | 500 | 0.1 | 128 | 20 | 0.59 | 0.80 | 7 | -777 | 20 clusters, high LR, 100K |
| 20 | 100K | 1e-3 | 10000 | 500 | 0.0 | 128 | 20 | 0.69 | 0.05 | 13 | -596 | 20 clusters, no conj, 100K |
| 21 | 200K | 1e-3 | 10000 | 500 | 0.1 | 128 | 30 | 0.67 | 0.77 | 17 | -718 | 30 clusters, 200K |
| 22 | 200K | 1e-3 | 10000 | 500 | 0.1 | 128 | 40 | 0.61 | 0.80 | 14 | -715 | 40 clusters, too sparse |
| 23 | 200K | 1.5e-3 | 10000 | 500 | 0.1 | 128 | 30 | 0.70 | 0.91 | 13 | -713 | Phase 1-2 best |
| 24 | 200K | 1e-3 | 10000 | 500 | 0.05 | 128 | 40 | 0.61 | 0.86 | 14 | -674 | 40 clusters, light conj |

### Phase 3A: Long beta warmup (Runs 25-28)

| Run | Steps | LR | KL-Warmup | LR-Warmup | Conj | Latent | Clusters | NMI | R² | Active | ELBO | ||rho|| | Notes |
|-----|-------|----|-----------|-----------|------|--------|----------|-----|-----|--------|------|---------|-------|
| 25 | 200K | 1.5e-3 | 50K | 500 | 0.1 | 128 | 30 | 0.69 | 0.94 | 14 | -701 | 212 | 25% beta warmup |
| 26 | 200K | 1.5e-3 | 100K | 500 | 0.1 | 128 | 30 | 0.69 | 0.96 | 15 | -689 | 258 | 50% beta warmup |
| 27 | 300K | 1.5e-3 | 150K | 500 | 0.1 | 128 | 30 | 0.70 | 0.98 | 13 | -675 | 399 | 50% warmup + 300K |
| 28 | 200K | 1.5e-3 | 100K | 500 | 0.1 | 128 | 20 | 0.68 | 0.96 | 11 | -693 | 269 | 50% warmup, 20 clusters |

### Phase 3B: LR warmup enabling higher LR (Runs 29-31)

| Run | Steps | LR | KL-Warmup | LR-Warmup | Conj | Latent | Clusters | NMI | R² | Active | ELBO | ||rho|| | Notes |
|-----|-------|----|-----------|-----------|------|--------|----------|-----|-----|--------|------|---------|-------|
| 29 | 200K | 2e-3 | 50K | 5K | 0.1 | 128 | 30 | 0.65 | 0.95 | 17 | -693 | 241 | Higher LR + LR warmup |
| 30 | 200K | 2e-3 | 100K | 10K | 0.1 | 128 | 30 | 0.66 | 0.96 | 18 | -686 | 261 | Both warmups long |
| 31 | 200K | 3e-3 | 100K | 10K | 0.1 | 128 | 30 | 0.67 | 0.98 | 14 | -685 | 401 | LR=3e-3 with warmup |

### Phase 3C: Latent dimensions (Runs 32-35)

| Run | Steps | LR | KL-Warmup | LR-Warmup | Conj | Latent | Clusters | NMI | R² | Active | ELBO | ||rho|| | Notes |
|-----|-------|----|-----------|-----------|------|--------|----------|-----|-----|--------|------|---------|-------|
| 32 | 200K | 1.5e-3 | 50K | 500 | 0.1 | 64 | 30 | 0.64 | 0.89 | 17 | -786 | 153 | Fewer latents |
| 33 | 200K | 1.5e-3 | 50K | 500 | 0.1 | 256 | 30 | 0.38 | 0.97 | 4 | -665 | 317 | More latents, cluster collapse |
| 34 | 200K | 1.5e-3 | 50K | 500 | 0.1 | 64 | 20 | 0.63 | 0.92 | 17 | -802 | 162 | Fewer latents + clusters |
| 35 | 200K | 1.5e-3 | 50K | 500 | 0.1 | 256 | 40 | 0.53 | 0.96 | 5 | -660 | 309 | More latents + clusters |

### Phase 3D: Best combos (Runs 36-37)

| Run | Steps | LR | KL-Warmup | LR-Warmup | Conj | Latent | Clusters | NMI | R² | Active | ELBO | ||rho|| | Notes |
|-----|-------|----|-----------|-----------|------|--------|----------|-----|-----|--------|------|---------|-------|
| 36 | 300K | 1.5e-3 | 150K | 500 | 0.1 | 128 | 35 | 0.66 | 0.98 | 14 | -677 | 387 | 35 clusters (worse NMI) |
| **37** | **300K** | **1.5e-3** | **120K** | **500** | **0.1** | **128** | **30** | **0.71** | **0.98** | **14** | **-685** | **377** | **NEW BEST** |

## Key Findings

### Best Results
- **Best NMI**: 0.71 (Run 37)
- **Best R²**: 0.98 (Runs 27, 31, 36, 37)
- **Best combined (NEW BEST)**: NMI=0.71, R²=0.98 (Run 37: 300K steps, 40% beta warmup, LR=1.5e-3, 30 clusters)

### Winning Config (Run 37)
```
python -m examples.variational_mnist.train --mode gradient --interaction full --n-latent 128 \
    --n-steps 300000 --learning-rate 1.5e-3 --kl-warmup-steps 120000 \
    --conj-weight 0.1 --entropy-weight 5 --posterior-entropy-weight 10 --n-clusters 30
```

### Phase 3 Insights

#### A. Long beta warmup is the biggest win
- R² jumps from 0.91 (10K warmup) to 0.94-0.98 with 50K-150K warmup
- Hypothesis confirmed: longer warmup lets clusters form freely, then gradually introduces linearization
- 40-50% of total steps as warmup is optimal
- NMI is preserved (0.69-0.71) because clusters stabilize before beta pressure arrives

#### B. Higher LR still hurts NMI despite LR warmup
- LR=2e-3 drops NMI to 0.65-0.66 regardless of LR warmup (5K or 10K steps)
- LR=3e-3 gets NMI=0.67 with long warmup — better than without (0.55) but still worse than 1.5e-3
- LR warmup helps prevent the worst collapse but can't fully compensate
- LR=1.5e-3 remains the sweet spot

#### C. n-latent=128 is optimal
- n-latent=64: Both metrics degrade (NMI=0.63-0.64, R²=0.89-0.92) — not enough capacity
- n-latent=256: Catastrophic cluster collapse (NMI=0.38-0.53, only 4-5 active clusters)
  - Extra capacity concentrates into few "mega-clusters" instead of being distributed
  - Even 40 clusters can't compensate — the model prefers fewer, more expressive clusters
- n-latent=128 is the Goldilocks zone

#### D. 30 clusters confirmed, 35 doesn't help
- 35 clusters with same config as Run 27 → NMI drops to 0.66
- The extra clusters just die (14 active out of 35 vs 13 out of 30)
- 30 clusters is the right amount of overclustering for 10 true classes

### The NMI-R² Tradeoff (Updated)
- Phase 1-2: Overclustering (30) + moderate LR (1.5e-3) + long training (200K) → NMI=0.70, R²=0.91
- Phase 3: Adding long beta warmup (40-50% of steps) → R² jumps to 0.98 without NMI cost
- The key insight: beta warmup gives the model a "free lunch" — clusters form during low-beta phase, then conjugation converges during high-beta phase
- 300K steps with 40% warmup (Run 37) is the optimal balance: enough warmup for cluster formation, enough post-warmup steps for R² convergence
