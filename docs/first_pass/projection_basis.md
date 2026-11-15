# Choosing the 2-D projection basis for the energy landscape

**Question.** The artifact projects every four-letter string onto the top-2
eigenvectors of the P-weighted covariance of its 104 one-hot (position, letter)
indicators. The user reported discontinuities in coverage and isolated points far from
the data mass, and asked whether a better basis exists — nonlinear iterative methods,
combinations of principal components, or an autoencoder/VAE.

**Answer (short).** No candidate we tested beats the current PCA basis on the metrics
that matter for a *viewing angle*; the one alternative with a principled kinetic
argument (TICA / linear VAC, the method BioEmu uses for its free-energy surfaces) is
degenerate when applied to one-hot indicators and, once regularised, reduces to
≈ PCA with no net gain. We therefore recommend **keeping PCA**, and document why.

All results below are exact (26⁴ strings enumerated; the substitution chain is applied
by fibre averages). The one sampled quantity is the trustworthiness estimate
(P-weighted by sampling 2000 strings).

---

## 1. Setup: the substitution kinetics used to judge a basis

The natural dynamics on the landscape is Stephens & Bialek's single-letter move,
realised as the **random-site heat-bath (Gibbs) chain**: pick a position i uniformly,
redraw its letter from P(lᵢ | rest). The update is a P-orthogonal projection, so the
chain is reversible with stationary distribution P and its transition operator T is
self-adjoint and positive semi-definite on L²(P). Time is counted in single-site
updates; L = 4 updates make one sweep. (A deterministic sweep i = 1..L is *not*
reversible, which is why the random-site update is used throughout.)

A linear basis is a pair of loadings V ∈ ℝ^(104×2); its coordinates are
y(s) = Vᵀ (x(s) − μ) with x the one-hot indicator vector. We score each candidate on:

| metric | what it measures |
|---|---|
| VAMP-2 (τ=4) | kinetic content of the 2-D basis under the substitution chain (Wu & Noé 2020) |
| implied timescales | t = −τ/ln λ of the basis's two coordinates |
| subspace variance share | share of total indicator variance in the 2-D plane (Eckart–Young) |
| basin separation | P-weighted fraction of strings whose plane-nearest string is in the same string-level descent basin |
| trustworthiness / continuity | Hamming ↔ plane neighbourhood agreement (Venna & Kaski 2001), P-weighted |
| coverage | P-mass in grid cells outside the largest 8-connected occupied component; empty cells inside the occupied hull |
| outliers | how far the tail of the P-mass sits from the P-weighted centroid, in RMS-radius units |
| time stability | principal angle of the basis fit on a frame one year before the end vs. the final frame |

---

## 2. Candidates

1. **PCA** — top-2 eigenvectors of C(0) (the current basis).
2. **TICA / linear VAC** at lags τ = 1, 4, 8 — generalised eigenvectors C(τ)v = λC(0)v
   (Pérez-Hernández et al. 2013; Schwantes & Pande 2013; Noé & Nüske 2013). The
   slowest modes are the reaction coordinates between wells — the projection BioEmu
   uses for free-energy surfaces (Lewis et al., Science 2025).
3. **TICA, regularised** — the same generalised eigenproblem restricted to the leading
   subspace of C(0) carrying 95% of the variance, which removes near-null (rare-letter)
   directions that the whitening otherwise amplifies.
4. **Fisher LDA** on the string-level descent basins (each support string labelled by
   the strict local minimum its steepest descent reaches), also regularised.
5. **Slow modes (nonlinear)** — the exact maximisers of VAMP over all functions, the
   diffusion-map coordinates (Coifman & Lafon 2006) of the heat-bath kernel.

---

## 3. Evidence

Numbers for the trajectory's final model ("pooled", the fixed-basis model) and for the
sparsest model (Austen). `vamp2` is at τ = 4 (one sweep), scored out of 2.0.

| basis | VAMP-2 | var share | basin NN | trustw. | outlier max (RMS) | mass >5·RMS | last-12 stab. |
|---|---|---|---|---|---|---|---|
| **PCA** | 1.518 | **0.170** | **0.350** | **0.886** | 2.0 | 0 | **9.8°** |
| TICA τ=4 (raw) | **1.821** | 0.062 | 0.206 | 0.877 | 5.7 | 0.014 | 24.3° |
| TICA τ=4 (reg.) | 1.550 | 0.170 | 0.333 | 0.879 | 2.1 | 0 | 9.9° |
| LDA (basins) | 1.552 | 0.169 | 0.323 | 0.887 | 1.9 | 0 | 23.0° |

The same ordering holds for the other three tab-1 models; the Austen model makes the
TICA failure vivid, because its support is sparse (7 059 of 456 976 strings, with
unseen letter pairs forbidden):

| Austen | VAMP-2 | var share | basin NN | outlier max (RMS) | occupied cells (of 1024) |
|---|---|---|---|---|---|
| PCA | 1.731 | **0.197** | **0.478** | 1.9 | 381 |
| TICA τ=4 (raw) | 2.000 | 0.001 | 0.153 | **108.5** | **7** |
| TICA τ=4 (reg.) | 1.759 | 0.196 | 0.444 | 1.9 | 363 |
| LDA | 1.761 | 0.196 | 0.465 | 1.9 | 363 |

The implied timescales show why: the Austen substitution chain is nearly reducible
(slowest timescale ≈ 8.5×10⁸ single-site updates), because forbidden letter pairs
trap the support into rare-letter near-absorbing states; the pooled model mixes far
faster (slowest timescale ≈ 76 updates ≈ 19 sweeps).

---

## 4. Findings

1. **Raw TICA is degenerate on one-hot indicators, and it is exactly the failure mode
   the user is worried about.** Whitening by C(0) divides by the variance of each
   indicator; a letter that is rare at a position gets a loading ~1/√p (e.g. 147 on
   'x' at position 2 in Austen). The top "slow" coordinates are then dominated by
   rare-letter near-conserved directions: they collapse the projection to a handful of
   cells (7 of 1024 for Austen) and push strings 108 RMS-radii away from the data mass
   — isolated points, maximally amplified. Its near-saturated VAMP-2 (2.0) is
   therefore an *artifact*: it is achieved on a subspace carrying 0.1% of the variance.

2. **Once TICA is regularised it becomes ≈ PCA.** Restricting the whitening to the 95%
   subspace removes the blow-up, and the regularised TICA gains ≤ 2% in VAMP-2 while
   losing a little basin separation, neighbourhood preservation and nothing else.
   There is no net win.

3. **No linear basis can beat PCA on explained variance.** Eckart & Young (1936): the
   best rank-2 approximation of C(0) is the top-2 eigenvectors, so the plane's share
   of variance (0.17–0.20 here) is the ceiling for every linear basis — TICA and LDA
   are linear combinations of the indicators, hence of the PCs, and the table confirms
   their subspace shares never exceed PCA's.

4. **The slow modes (nonlinear) are the theoretical optimum but are numerically
   fragile and not the right visual coordinates here.** Their operator is the
   symmetrised T = P^½ T P^-½, whose entries span ~15 decades (P spans ~10⁻³⁰ to
   ~10⁻¹); the near-reducible spectrum defeats both ARPACK and dense eigensolvers, and
   the modes concentrate on the tiny-mass traps rather than the main wells.

5. **PCA is time-stable among the TICA/LDA candidates.** A year before the end of the
   trajectory the PCA plane is already within 9.8° of the final plane, and the angle
   shrinks smoothly to 0; regularised TICA matches this (9.9°), but raw TICA (24.3°) and
   LDA (23.0°, with irregular jumps from basin relabelling) do not. (The later selectable
   bases tempered/coupling/energy-aligned are still more stable, 3.4–6.4°.)

6. **Interpretability.** A linear basis has per-(position, letter) loadings; each
   coordinate is literally "a weighted vote over which letter sits in which position",
   and the squared loadings give the share of each position. The two PCA axes of each
   model annotate cleanly (e.g. Austen axis 1: position 2 = 35% of the squared
   loading, dominated by 'h' at position 2 and 't' at position 1). Nonlinear embeddings
   have no such decomposition — only *post hoc* statements of the form "strings in this
   region share feature Z".

---

## 5. Recommendation

**Keep the current basis: the top-2 principal axes of the P-weighted indicator
covariance C(0).**

- It is the *unique* maximiser of the 2-D explained variance (Eckart–Young), so no
  linear alternative can present more of the landscape in the plane.
- It is best or tied-best on every visual-fidelity metric: basin separation (0.35 vs
  0.21–0.33 for the alternatives on the pooled model), neighbourhood preservation
  (trustworthiness 0.886, tied with LDA's 0.887 within sampling noise), coverage (no
  isolated components, 22% hull emptiness), and outlier control (max 2.0 RMS, zero mass
  beyond 5 RMS).
- It is the most time-stable among the TICA/LDA candidates (9.8° a year out, converging
  smoothly); the later selectable bases are still more stable (tempered 3.4°, coupling
  4.2°, energy-aligned 6.4°).
- It is fully interpretable via per-(position, letter) loadings.

The kinetically-motivated alternative (TICA, as used by BioEmu) is **rejected** as a
drop-in: applied to one-hot indicators it degenerates exactly into the isolated-points
failure the user described, and regularised it is statistically indistinguishable from
PCA. This is not a weakness of TICA — it is a statement that on *these* features the
meaningful slow directions coincide with the high-variance directions, so the
variance-optimal and kinetics-optimal views agree; where they differ (the raw TICA
coordinates) the difference is rare-letter noise.

**Why the isolated points are not a bug of the basis.** PCA already keeps every string
within 2 RMS radii of the P-weighted centroid (zero mass beyond 5 RMS). The apparent
"discontinuities" are intrinsic: the support is sparse (1.5% of strings for Austen;
forbidden pairs everywhere else), and 80% of the indicator variance lives in PC3+,
which any 2-D projection must fold over. No linear basis can recover that mass
(Eckart–Young), and the nonlinear alternatives either lose interpretability (autoencoder/VAE)
or are ill-conditioned (slow modes). The honest statement is that a 2-D viewing angle of a
~100-dimensional landscape will always have holes; PCA makes them the smallest possible.

---

## 6. Implementation

- `langfit/basis.py` — exact heat-bath kinetics, PCA / TICA / LDA axes, VAMP-2 and
  implied timescales, slow modes, and the recommended basis:
  - `basis.pca_basis(P, top=2)` → `{V, eigenvalues, mean, coordinates, annotation}`;
    the annotation is byte-for-byte the artifact's current `axis_report` output, so the
    four tab-1 models and the trajectory's fixed basis can be produced with one call.
  - `basis.ticia_axes`, `basis.lda_axes`, `basis.coords_linear`, `basis.vamp2`,
    `basis.annotate_axes` for the alternatives and for scoring.
- `scripts/basis_study.py` → `data/derived/basis_study.json` — the evidence table and
  the per-model annotation for all four tab-1 models plus time-stability curves.
- `tests/test_basis.py` — exactness and structural tests (stationarity, reversibility,
  C(0) recovery, P-orthonormality, Eckart–Young bound, VAMP limits).

---

## 7. References (verified)

- Eckart & Young, "The approximation of one matrix by another of lower rank",
  *Psychometrika* 1(3), 211–218 (1936).
- Ukkonen, "Approximate string-matching with q-grams and maximal matches",
  *Theoretical Computer Science* 92, 191–211 (1992).
- Venna & Kaski, "Neighborhood preservation in nonlinear projection methods: an
  experimental study", *Proc. ICANN 2001*, 485–491 (2001).
- Coifman & Lafon, "Diffusion maps", *Applied and Computational Harmonic Analysis* 21,
  5–30 (2006).
- Stephens & Bialek, "Statistical mechanics of letters in words", arXiv:0801.0253.
- Noé & Nüske, "A variational approach to modeling slow processes in stochastic
  dynamical systems", *Multiscale Model. Simul.* 11, 635–655 (2013).
- Pérez-Hernández, Paul, Giorgino, De Fabritiis & Noé, "Identification of slow
  molecular order parameters for Markov model construction", *J. Chem. Phys.* 139,
  015102 (2013).
- Schwantes & Pande, "Improvements in Markov state model construction reveal many
  non-native interactions in the folding of NTL9", *J. Chem. Theory Comput.* 9,
  2000–2009 (2013).
- Wu & Noé, "Variational approach for learning Markov processes from time series
  data", *Journal of Nonlinear Science* 30, 23–66 (2020).
- Lewis et al., "Scalable emulation of protein equilibrium ensembles with generative
  deep learning" (BioEmu), *Science* 389, 6761 (2025), doi:10.1126/science.adv9817.

---

## 8. Selectable bases for a smoother surface (extended assignment)

The user's follow-up asked for ~5 selectable bases whose goal is a *locally
consistent* surface — broader peaks and wells instead of the jagged troughs/spikes the
PCA quasi-lattice produces — with PCA kept as the default.

### 8.1 Diagnosis: what drives the roughness (measured, not assumed)

Decomposing `E_bin = -log2 P(cell) + log2 |bin|` on the pooled model at grid 32 shows
the roughness is **not** an occupancy artifact, contrary to the "lattice rows"
hypothesis. `|bin|` is wildly variable (CV 6.35; 17% of occupied cells hold one string)
but the *jumps* between 4-neighbour cells are almost entirely energy jumps:
`mean |dE_bin| = 3.5 bits`, `mean |d E_free| = 3.85 bits`, `mean |d log2|bin|| = 2.09
bits`, and `log2|bin|` explains only 0.9% of the jump variance. The real driver is the
**within-cell energy spread**: strings in the same cell differ by 6.8 bits on average
(up to 13.7 bits), so `E_bin` — the cell's mean log-probability — is a noisy summary.
Bases must therefore *align the plane with energy* or *pull similar strings together*.

### 8.2 The five bases

Implemented in `langfit/basis.py` (`BASES` registry + `compute_basis(name, model, …)`),
all deterministic maps from the fitted model alone, defined for all 26⁴ strings, unit
loadings, largest-|loading| positive (the `projection.py` convention):

| id | construction |
|---|---|
| `pca` | top-2 eigenvectors of the P-weighted indicator covariance (default) |
| `coupling_spectral` | top-2 eigenvectors of the gauge-fixed (double-centred) coupling matrix J; `E = -½ xᵀJx`, so these are the Hopfield-pattern directions of largest quadratic energy variation (forbidden pairs V=−∞ set to 0) |
| `energy_aligned` | two PLS loadings of the P-weighted regression of `E(s)` on the indicators |
| `tempered_pca` | PCA of the covariance weighted by `P^α`, α=0.75 (the default, on the smoother-surface criterion; α=0 = classical MDS/Torgerson, α=1 = PCA) |
| `lda` | Fisher discriminant axes of the string-level descent basins (regularised to the 95%-variance subspace): the directions that best separate the landscape's wells |

### 8.3 Metrics (raw uncapped `E_bin`, pooled model, grid 32)

Roughness is normalised **three ways**: by the P-mass-weighted SD of `E_bin`
(outlier-robust, **primary**), by the mass-weighted 5-95 inter-percentile range, and by
the raw range (outlier-sensitive, kept for reference). A fourth number is the
SD-normalised roughness on the **common-support core** (top-90% P-mass cells), so a
basis cannot win by occupying fewer cells.

| basis | \|dE\|/sd | \|dE\|/range | core \|dE\|/sd | Moran I | R²(E\|cell) | wells 1/2/3 | spurious | continuity | \|bin\| CV | occ. | eff. | comp. | varR | stab. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pca | **0.656** | 0.103 | 0.347 | 0.23 | 0.63 | 53/40/30 | 19% | **0.375** | 6.35 | 0.49 | 117 | 1 | 1.4 | 9.8° |
| coupling_spectral | 0.724 | 0.055 | 0.429 | 0.64 | 0.70 | 42/25/14 | 5% | 0.083 | **1.45** | 0.71 | 160 | 1 | 1.7 | 4.2° |
| energy_aligned | 0.666 | 0.081 | **0.236** | 0.42 | **0.69** | 62/47/38 | 11% | 0.145 | 3.20 | 0.64 | 168 | 1 | 1.0 | 6.4° |
| tempered_pca (α=0.75) | 0.663 | 0.064 | 0.345 | 0.43 | 0.67 | 32/27/20 | 9% | 0.319 | 4.57 | 0.52 | 144 | 1 | 1.5 | 3.4° |
| lda (well-discriminant) | 0.674 | 0.100 | 0.335 | 0.35 | 0.65 | 47/35/30 | 26% | 0.341 | 5.38 | 0.50 | 130 | 1 | 1.4 | 23.0° |

(metrics at grids 16/32/48 agree qualitatively; full table in
`data/derived/basis_study.json` under `smoothness`.)

**Correction after outlier-robust normalisation.** The earlier report called
`coupling_spectral` the smoothest basis, but that used range-normalisation, which is set
by the emptiest cells: `coupling_spectral` has the largest raw range (37 bits, stretched
by a few sparse cells) but a small mass-weighted SD, so its low `|dE|/range` (0.055) was
an artifact. On the robust SD-normalised metric it is the **roughest** on the pooled model (0.724; also
roughest on Austen and Bluesky, but actually the *smoothest* on twitter2023 at 0.580), and
on the common-support core also the roughest (0.429); `energy_aligned` is the smoothest on
the core (0.236) and `pca` the smoothest on the full surface (0.656) — all on the pooled model.

**Collapse guards**: occupied-cell fraction, effective occupied cells `exp(H)` of the
cell-mass entropy, the P-weighted axis-variance ratio, and the number of 4-/8-connected
occupied components (diagonal touches do not count for 4-connectivity). The component
count is resolution-dependent and no basis is fully connected at G = 32 except
`coupling_spectral` on the well-populated models (pooled/twitter 8-connected components
= 1/1); the others fragment (pooled G=32, 4-/8-connected: coupling 1/1, tempered 5/2,
energy_aligned 7/5, pca 11/4, lda 17/7).

**On the "locality" idea (why it was dropped).** The integrator's audit showed the
linear LPP over the *full* uniform Hamming graph is vacuous: every centred linear
function of the one-hot indicators is a level-1 Laplacian eigenfunction (eigenvalue 26,
dimension 4×25 = 100), so the generalised spectrum is flat and the graph term
discriminates nothing — the tie-break alone selects the axes, i.e. it is a
uniformly-whitened PCA, not a locality-preserving projection (verified: all non-null
generalised eigenvalues equal 26). A model-dependent flux-weighted version
(w_st = (P_s P_t)^β/2, Hellinger) is genuinely kinetic but collapses on sparse models
(the near-reducible substitution chain concentrates coverage to 1%). It is therefore
replaced by the honest, well-posed **Fisher LDA on descent basins**.

### 8.4 What each non-default basis is for — no single winner

There is **no basis that beats PCA on both smoothness and neighbour preservation**;
each non-default basis trades one for the other:

- **coupling_spectral** — the most uniform occupancy on three of four models (CV 1.45
  pooled, 1.27 twitter, 1.15 bluesky; energy_aligned is more uniform on Austen at 1.27),
  fewest spurious wells, stable — good for the *well / attractor geometry*. **Corrected:**
  its low range-normalised roughness is a range artifact — on the robust SD metric it is
  the roughest on the pooled model (0.724; smoothest on twitter2023), so it does *not*
  give a smooth energy surface. It destroys substitution continuity (0.083).
- **energy_aligned** — high cell homogeneity (R² 0.69–0.80), among the highest on all
  four models (coupling_spectral is higher on twitter/bluesky/pooled, tempered on Austen),
  so `E_bin` is a faithful summary. This is the explicit x≈E trade: worst continuity
  (0.145), nearly a 1-D energy ramp. Always report a neighbour-preservation metric next
  to it.
- **tempered_pca** — highest substitution continuity among the non-default bases on
  twitter2023 and bluesky2024 (LDA is higher on pooled and Austen: 0.341 vs 0.319 and
  0.077 vs 0.051). α=0.75 is the default, chosen on the *smoother-surface* criterion
  (SD-normalised |ΔE| 0.663 vs 0.913/1.250); **no** α in {0.25,0.5,0.75} gives a single
  connected component (4-/8-connected components at G=32: 0.25 → 3/3, 0.5 → 3/3,
  0.75 → 5/2), so the earlier "connected support" part of the criterion is unsatisfiable
  and is dropped. α=0.5's lower 0.29 occupancy gives a sparser, more fragmented-looking
  surface. It down-weights the mass the landscape is defined over (R² 0.67).
- **lda (well-discriminant)** — genuinely distinct (angle 8–17° from PCA), places
  strings by which well they belong to (continuity 0.341, near PCA), non-degenerate on
  all four models (coverage ≥ 50% of PCA, axis-variance ratio ≤ 1.7). Honest limitation:
  least time-stable (23°, basin relabeling), and it is *not* a locality-preserving
  projection — the LPP idea it replaces was audited and found vacuous (flat Laplacian
  spectrum on the full Hamming graph) or degenerate (flux-weighted, sparse models).

The honest recommendation remains **PCA as the default**; select a non-default basis
for a specific question (well geometry, energy fidelity, rare-word coverage, or
substitution connectivity) with the corresponding caveat shown in the artifact.

---

## 9. Sixth basis: the energy-metric layout (geodesic MDS of the 3-D surface)

The user asked for a layout that organises points by *optimising the continuity of the
3-D surface itself* (energy as z), not by projecting word data.

### 9.1 Degeneracy without word information (the λ→0 control)

With **no** word information the problem is degenerate: sort strings by energy and lay
them along a boustrophedon raster. The surface becomes the quantile function of E —
smooth at every scale — and x,y encode only the energy rank (density of states).
Implemented as `basis.energy_rank_layout` and reported as the explicit degenerate
control. Any "unexplained" organising principle that reduces to energy order is this.

### 9.2 The energy metric

A non-degenerate layout needs one structural input that is **not** a feature projection:
the substitution (move) graph. Edge length between one-edit neighbours i,j is
`l_ij = sqrt(h² + (ΔE_ij/λ)²)` with hop length h = 1 and λ the length scale (bits of
energy per layout unit); the geodesic `D_ij` is the shortest-path length (Dijkstra), and
the layout is landmark-MDS / Isomap (Tenenbaum, de Silva & Langford 2000; de Silva &
Tenenbaum 2004) of `D`, with P-aware farthest-point landmarks. λ→∞ is the pure graph
(Hamming) geometry; λ→0 is energy order; in between, adjacent strings with large energy
differences are pushed apart so slopes are bounded ≈λ by construction (verified: the
realised |dE|/||dp|| edge slopes have their 95th percentile ≤ λ).

This differs from sketch-map (Ceriotti, Tribello & Parrinello, PNAS 2011), which embeds
a *neighbourhood* graph in a *soft* low-dimensional manifold; here the graph is the
discrete single-substitution lattice and the geodesic metric is exact. It is a
coarse-grained cousin of disconnectivity graphs (Becker & Karplus 1997) and of Wales'
energy-landscape visualisations: the layout encodes barriers (energy differences along
substitution paths), not just the well ordering.

### 9.3 Why GP/BO and splines are the wrong tools (stated plainly)

- **GP/Bayesian optimisation**: 2N = 913 952 free coordinates with a cheap
  differentiable objective and a trivial optimum without a constraint; BO is designed
  for expensive black-box objectives in low dimension, so it is the wrong tool.
- **Splines**: a spline layout `x(u), y(u)` with `u` a feature vector is again a
  parametric projection of word data; with `u` = energy rank it is the degenerate
  control. Neither provides a non-trivial organising principle.

### 9.4 Determinism, λ and landmarks

Layouts are deterministic: fixed seed, P-aware farthest-point landmark sampling
(K = 200), and an orientation convention — rotate to the P-weighted principal axes of
the layout, then sign each axis so its P-weighted Spearman correlation with E is
non-negative. λ is the flux-weighted median of |dE| over substitution edges (flux =
sqrt(P_i P_j)), so a typical edge has equal hop and energetic length. Forbidden strings
are capped at the largest finite energy so the graph stays connected. Metrics and the
λ∈{0.25,1,4}×λ_med sweep plus the two controls are in `data/derived/basis_study.json`
under `energy_metric`; figures in `figures/energy_metric_controls_<model>.png`.

### 9.5 Verdict: the energy-driven layout is degenerate (negative result)

We ran the geodesic-MDS layout (K = 150 landmarks, λ = flux-weighted median of |dE|,
grid 32) on all four models and the two controls. The honest verdict is **degenerate**:

| model | λ (bits) | Kruskal stress | Spearman(y, E) | Pearson R²(y, E) | linear R² of x on indicators | occupancy | \|dE\|/sd |
|---|---|---|---|---|---|---|---|---|
| austen | 2.94 | 0.92 | 0.894 | 0.576 | 0.96 | 0.18 | 3.11 |
| twitter2023 | 3.11 | 0.89 | 0.956 | 0.920 | 0.98 | 0.40 | 0.51 |
| bluesky2024 | 3.10 | 0.90 | 0.958 | 0.919 | 0.98 | 0.40 | 0.49 |
| pooled | 3.22 | 0.88 | 0.976 | 0.948 | 0.98 | 0.45 | 0.43 |

(Spearman and Pearson R² are computed over the model's *support* — the strings with
P > 0 — with energies capped at the largest finite value; forbidden strings are excluded.)

Three independent signs of degeneracy:

1. **The y-axis is energy-ranked.** Spearman(y, E) = 0.89–0.98 over the support on all
   four models, so the dominant axis reorders strings by energy (the λ→0 degenerate
   control) despite λ being the *flux median*. The *linear* Pearson R²(y, E) is strong
   only for pooled (0.948) and twitter/bluesky (0.92); for Austen it is moderate (0.576),
   so Austen's ramp is weaker and its degeneracy rests more on (2) and (3).
2. **The x-axis is a linear projection.** The linear R² of x on the 104 one-hot
   indicators is 0.96–0.98: the residual axis is ≈ a linear (PCA-like) basis, not a
   genuinely nonlinear organising principle.
3. **High stress + sparse coverage.** Kruskal stress 0.88–0.92 (the 2-D embedding is
   lossy; stress is flat in K: 0.898/0.880/0.877 for K = 50/100/200), and Austen
   occupancy 0.18 (< 50% of PCA's 0.37).

**Root cause.** H(4,26) has diameter 4, so the substitution graph's geodesic metric —
even energy-weighted — is nearly Euclidean, and classical/landmark MDS of a
near-Euclidean metric reproduces a linear basis; the only energy-dependent coordinate is
the ramp. The "barrier structure" intermediate regime (cheapest path avoiding barriers,
like a disconnectivity graph) is too narrow to survive a 2-D embedding at this scale.
The alternative (an energy-weighted Laplacian eigenmap with edge weight
exp(-ΔE²/ℓ²)) is worse: it kills edges across barriers, localising on the sparse
(near-reducible) Austen graph — the same failure as the raw TICA degeneracy documented in
§8. **A purely energy-driven 2-D layout has no non-trivial organising principle here**;
the controls (`basis.energy_rank_layout`, `basis.hamming_layout`) remain as the honest
demonstration of the two limits, and are not shipped as selectable bases.

### 9.6 Supplementary evidence (D-spread, λ-sweep, Laplacian-eigenmap comparator)

- **Geodesic D-spread (closed form, λ→∞).** The Hamming geodesic distance has mean 3.85,
  SD 0.38, **CV 0.10**, and 85.5% of all 26⁴ string pairs are at distance 4 (the
  diameter). The graph is near-equidistant, so MDS of it is a featureless ball — this is
  the structural cause of the stress ≈ 0.9 and the linear (PCA-like) x-axis.
- **λ-sweep (pooled).** At λ×0.25 the y-axis *is* the energy ramp (Spearman(y,E)=1.000,
  R²=0.999); at λ×1.0 it is still the ramp (0.976 / 0.948); at λ×4.0 it is not
  (0.034 / 0.001) but is then the near-equidistant graph geometry. There is **no**
  intermediate λ that escapes: the sweep is degenerate end to end.
- **Laplacian-eigenmap comparator (run, both the linear indicator-space version on all
  four models and the nonlinear node-space version on pooled).** Edge weight
  w = exp(-(ΔE/ℓ)²) at ℓ ∈ {0.5, 1, 2}×λ_med, generalised problem L y = μ M y.
  With **M = graph degree** the layout collapses: occupancy 0.005–0.089 (linear) / 0.062
  (nonlinear, pooled), and the nonlinear x-axis is 85% the energy ramp (R²(x,E)=0.850).
  With **M = P-weighted mass** the *linear* restriction is non-degenerate (occupancy
  0.37–0.51, distinct from PCA by 13–37°) but is linear by construction (linear R²=1.0),
  while the *nonlinear* node-space version collapses to 0.6% occupancy (severe
  localisation, ill-conditioned P-mass spectrum). The energy weight kills barrier edges
  and the graph localises — the same failure as raw TICA (§8). The comparator does not
  clear the bar, so no `energy_laplacian` basis is promoted.

**The one non-degenerate member — and why it is not promoted.** The *linear* restriction
of the energy-weighted Laplacian eigenmap with P-weighted mass (`basis._energy_locality_axes`,
objective: "put strings one substitution apart *and* of similar energy close together")
is a legitimate, distinct linear basis: coverage 0.37–0.51 (≥ 0.8× PCA), substitution
continuity ≥ 0.7× PCA (it beats PCA on twitter/bluesky/pooled), axis-variance ratio ≤ 2.0,
13.6–37.4° from PCA, and 11–17° time-stable. But it does **not** beat PCA on smoothness —
its robust core roughness is *worse* than PCA on 3 of 4 models — better only on bluesky
(0.304 vs 0.308) — and the same holds on the full-surface metric (0.363/0.341/0.304/0.360
vs PCA's 0.360/0.340/0.308/0.347): it beats PCA on smoothness on 1 of 4 models, below the
required 3 of 4 — and its smoothness is partly circular (energy enters the edge weights). It fails the promotion bar and is kept as a documented non-recommended
variant, not a selectable basis.
