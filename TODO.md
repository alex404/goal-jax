# TODO

## Switch variational conjugation models to the directed parameterization; implement parameter-space whitening

Store $(\theta^\star_\ell, \Theta_\ell, \phi_\ell)$ instead of $(\theta_\ell, \phi_\ell)$, and expose the harmonium parameters through an accessor, $\theta_\ell = \theta^\star_\ell - \rho_\ell(\theta^\star_{\ell-1})$. The recognition bias of level $\ell$ becomes $\theta^\star_\ell + \rho_\ell(\hat\theta_{\ell-1}(x)) - \rho_\ell(\theta^\star_{\ell-1})$, so the shared conjugation function is retained and $\phi$ appears only in the recognition model.

Motivation: the fixed-target term of the M-step then depends only on generative parameters and is identified, as in the EM with target whitening used in goal-apps (HMoG, MFA). In the harmonium parameterization it is degenerate in the split between $\theta_\ell$ and $\phi$, and only the gap term resolves it.

Steps:

1. Port `scratch/teacher/whiten.py` to directed coordinates: the generative parameters transform as in `whiten_prior` for LGM/HMoG; $\phi$ by affine composition at its input and output.
2. Start each M-step at $a \cdot \theta_t$ rather than $\theta_t$ (currently approximation 1 in the draft).
3. Two-block MM: generative block on $-\mathbb E_\pi \log p + D$, conjugation block on $D$.
4. Separately investigate the collapse of the recognition precision in `scratch/circuit_whiten/REPORT.md` (failure mode, seeds 4 and 5), which whitening does not address.

References:

- `~/documents/latex/articles/conjugated-graphical-harmoniums/scratch/variational-conjugation/article.tex`: Relation to Directed Variational Autoencoders; Whitened Variational EM.
- `src/goal/models/graphical/hmog.py` (`whiten_prior`): one global map applied to every component.
- `src/goal/models/graphical/mixture.py` (`whiten_prior`): one map per component (MFA).
- `scratch/circuit_whiten/mm.py`, `scratch/teacher/mm5.py`: current whitening, which acts on the target only and is diagonal.
