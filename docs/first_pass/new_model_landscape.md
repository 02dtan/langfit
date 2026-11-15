# Landscape of the variable-length (augmented) model

Code: `langfit/nm_landscape.py` (model assembly, projection, estimation, wells),
`scripts/build_nm_landscape.py` (build pipeline), `tests/test_nm_landscape.py`.
Output: `data/derived/nm_landscape.npz` (compact arrays) and
`data/derived/nm_landscape.json` (metadata, basis, diagnostics).  Every choice below
is a **new-model** choice, kept strictly separate from the frozen four-letter
Stephens–Bialek (S&B) study in `langfit.maxent`.

## 1. What is shown

Two views, in the style of the S&B artifact:

1. **Static landscape** of the final fitted model: the grand-canonical energy
   landscape of all words of length 1–8, projected onto the top two principal axes of
   a length-invariant bigram profile, with the bin energy as height, projected wells
   and string-level minima overlaid.
2. **Evolution** over the Twitter timeline: the same surface for the empty-data prior
   frame and each cumulative yearly threshold 2006–2023, in one fixed basis.

## 2. Model

### 2.1 Lengths

The modelled lengths are **L = 1 … 8**.  L = 2 … 8 is the range the Lengths study
validates as trustworthy for *all* quantities (87.9 % of tokens, exact log Z / model
entropy / code lengths; `docs/first_pass/length.md`).  L = 1 is included because it is
trivially 26 states and, with no position pairs, its model is uniform — it costs
nothing and completes the move graph's bottom boundary.  L ≥ 9 is excluded: there the
study finds the fitted models over-weight the most probable words (`length.md` §5.2),
so its landscape would not be trustworthy.

### 2.2 Per-length model

For each length the word distribution is the pairwise maximum-entropy (Potts) model

    P_L(s) = exp( Σ_{i<j} V^L_ij(s_i, s_j) ) / Z_L ,   s ∈ {a..z}^L ,

fitted with a Gaussian prior V ~ N(0, σ²) on every coupling cell (the "map" mode of
the exact study, the same estimator family as the S&B trajectory):

| L | fitter | why |
|---|---|---|
| 1 | uniform (no pairs) | the pairwise model with zero position pairs is uniform |
| 2–5 | `langfit.potts`, exact MAP, dense 26^L enumeration | exact; 26^5 ≈ 1.2·10⁷ is tractable |
| 6–8 | `langfit.longwords`, maximum pseudo-likelihood, exact log Z by split enumeration | 26^L cannot be enumerated at every step; PLM is the standard deterministic estimator; log Z is exact for L ≤ 8 |

σ² is chosen **per length by cross-validation** on the final pooled data (held-out
log-loss for L ≤ 5, held-out pseudo-likelihood for L = 6–8), on the same half-decade
grid as the frozen study, subsampled to N = 5·10⁵ tokens per length for tractability.
The result (reported in `nm_landscape.json` under `prior`) is **σ² = 3.16 for
L = 2, 3, 4, 6, 7, 8 and σ² = 10 for L = 5**; in every case the held-out loss is
essentially flat across the grid (the selection spans a few millibits to 0.1 bit), so
the prior width is weakly identified at this scale — the same behaviour the frozen
study reports at its N.  The chosen σ² is **held fixed across all frames** (as in the
S&B trajectory), so the trajectory varies only through the data.  With no data (the
empty frame) the MAP estimate is V = 0, i.e. the uniform model within every length.

### 2.3 Grand-canonical energies

Lengths are joined on one energy scale exactly as in `langfit.edits`:

    E(w) = −log₂ P(L(w)) − log₂ P_L(w)(w) = H_L(w) − μ_L ,
    H_L(w) = −Σ_{i<j} V^L_ij(w_i, w_j) / ln 2 ,
    μ_L    = log₂ P(L) − log₂ Z_L .

P(L) is the **posterior mean under a symmetric Dirichlet(1, …, 1) prior** over the
eight modelled lengths:

    P(L) = (n_L + 1) / (N + 8),

where n_L is the cumulative count of length-L tokens in the frame.  The +1 pseudo-count
makes P(L) > 0 for every length in every frame, so the empty frame is defined
(P(L) = 1/8) and no word ever has zero grand-canonical probability from a vanishing
P(L).  This is consistent with `langfit.edits`'s `grand_canonical_energy`, which takes
any P(L) normalisation (a constant over lengths only shifts E by a constant).

### 2.4 Move set and wells

Moves are the four elementary edits of `langfit.edits`: **substitution**, **adjacent
transposition**, **insertion**, **deletion**.  The configuration space is the union of
the 26^L grids for L = 1 … 8.  **Boundary treatment**: a length-1 word has no deletion
move (L = 0 is not modelled) and a length-8 word has no insertion move (L = 9 is not
modelled); every other move stays inside the space.  A **string-level minimum** is a
word whose energy is strictly below every modelled neighbour's energy.

## 3. Data and frames

Data: `data/derived/twitter_month_L1-8_all.pkl.gz` (all 41 shards of
`enryu43/twitter100m_tweets`, English-filtered by `langfit.social`, monthly,
2006-03 … 2023-04).  The final (pooled) record is **774,217,636 tokens** across
L = 1–8 (L = 2–8 is 87.9 % of all L = 1–20 tokens).

**Subsampling.**  The pseudo-likelihood fits cost O(n_types) and the full record has
~3·10⁵ types per length (an L = 8 PLM fit at full N does not finish in 30 min), so —
exactly as the Lengths study subsamples every length to a common N — **each length is
subsampled to a common N = 2·10⁶ tokens per frame** (multivariate hypergeometric,
fixed seed).  At N = 2·10⁶ the couplings are essentially converged (the Lengths study
shows the small-N coupling inflation is gone by 10⁶ tokens).  The length distribution
P(L) is always computed from the *full* cumulative counts, never the subsample.

Frames are the empty frame plus **cumulative yearly thresholds** 2006–2023 (19 frames).
The granularity is chosen by measuring the per-frame fit cost: a cold exact L = 5 MAP
fit is ~7 min and a cold L = 8 pseudo-likelihood fit ~5 min, but warm-started fits
cost ~seconds to ~1 min, and the dominant per-frame cost is the exact L = 8 partition
function (split enumeration, ~51 s, independent of N).  A warm frame therefore costs
~1.7 min, so monthly (204 frames) would need ~6 h and quarterly (70) ~2 h, outside the
~3 h / 2-worker budget; **yearly (19 frames) is ~50 min** and is used.  Successive
frames are warm-started from the previous frame's couplings.  (All timings recorded in
`nm_landscape.json` under `measure`.)

## 4. Projection

A word of any length is represented by its **boundary-marked bigram profile**: the
count vector of the 27² bigrams over {a..z, #}, where "#" marks both ends, so a word
w₁…w_L has the L + 1 bigrams "#w₁", "w₁w₂", …, "w_{L−1}w_L", "w_L#".  This profile is
length-invariant, which is what lets one coordinate system describe every length.
The two axes are the top two principal axes of the bigram-profile covariance under the
*final* grand-canonical model P(w) = P(L) P_L(w).  Along an axis u the coordinate is a
sum over the word's L + 1 bigrams,

    x(w) = Σ_{k=0}^{L} u[ w_k, w_{k+1} ]  (w₀ = w_{L+1} = "#"),

a **linear per-position / q-gram score** (the structure that makes the bin counts
below tractable).  The covariance is estimated by importance sampling (unbiased;
the same sampler as for the bin probabilities) because a closed form would need
marginals of order four.  The basis is **fixed across frames** (a viewing angle chosen
from the final data; no tokens after frame t enter the fit at frame t), and per-axis
composition is reported as the largest bigram loadings (the variable-length analogue
of the S&B tab's per-position loadings).  This basis is the one validated by the
projection study (`langfit/basis.qgram_profile`): on the exact grand-canonical
L = 1–4 model the top-2 plane carries ~14 % of the bigram-profile variance, and axis 1
is the interpretable content axis (word-initial *t*, *th*, *he*, word-final *e* — the
"the/he/that" direction, the same signal the S&B one-hot PCA finds).

## 5. Height

The height is the **bin energy relative to the prior (empty-data) model** — the Zwanzig
(1954) exponential average of the string energies, compared cell-by-cell with the same
average under the prior — so the empty frame is exactly flat:

    z_bin = log₂ P_prior(bin) − log₂ P_bin ,
    P_bin    = Σ_L P(L) Σ_{s∈bin, |s|=L} P_L(s) ,                       (model mass)
    P_prior  = Σ_L (1/8) Σ_{s∈bin, |s|=L} 1/26^L  =  Σ_L (1/8) |bin|_L / 26^L .

z_bin is the (negative) log-ratio of the model's bin mass to the prior's: a well
(negative z) is a cell where the model concentrates mass relative to the uniform prior,
a barrier (positive z) a cell it depletes.  Crucially, **z_bin does not need |bin|**:
the |bin| factor cancels between P_bin and P_prior, so it is never estimated on its
own.  Both P_bin and P_prior are estimated from the *same* importance draws:

* **L ≤ 5**: every string is enumerated, so P_bin(bin | L) and P_prior(bin | L) =
  |bin|_L / 26^L are **exact**.
* **L = 6–8** (26⁸ ≈ 2.1·10¹¹ strings): each sampled s ~ q contributes two weights,
  w_model = P_L(s)/q(s) and w_prior = (1/26^L)/q(s), so a cell with model mass
  necessarily has prior mass and vice versa — there is no separate |bin| estimator, no
  count floor, and no "mass under one estimator" gap.  The estimate is unbiased with
  per-cell standard errors.  (The projection scores are continuous reals, so an exact
  transfer-matrix bin count is not used.)

A cell with **no mass under either estimator** is a true gap (z = null).  A cell whose
model mass is dominated by a length with ESS below 100 (the L = 8 sampler has
ESS ≈ 10, the L = 7 sampler ≈ 76) is **flagged low-confidence**: there the delta-method
standard error is only a *lower bound*, not a calibrated error, because the proposal's
tails are thinner than the concentrated model's.

## 6. Wells, barrier, minima

* **Projected wells** are local minima of the z_bin surface on the graph of cells
  linked by the move set (a cell u ≠ v is adjacent iff some string of one reaches some
  string of the other by one move), ranked by 0-dimensional sublevel-set persistence
  (`langfit.projection.persistence`, reused read-only).  Wells with persistence
  ≥ 1 bit (a factor-2 drop in mean string probability) are **named by the most
  probable string of the whole basin**: the union of the basin cells' top-word lists
  (exact L ≤ 5 enumeration + sampled L ≥ 6 draws + observed types), ranked by
  grand-canonical probability P(w) = P(L)P_L(w), with the top three listed and each
  entry carrying `{w, L, p, approx, share}` (share = p / basin_mass).  The minimum
  cell's own top string is stored separately as `min_cell_top` (it is often a
  low-mass single string that names the *cell*, not the *basin* — a length-mixing
  effect).  The main barrier is the highest saddle at which a labelled well merges.
  The adjacency is exact for L ≤ 5 (all strings and their moves) and sampled for
  L = 6–8 (importance samples plus observed types).
* **String-level minima** are strict minima of E on the move graph.  For L ≤ 5 they
  are exact (`edits.landscape` on the L = 1–5 space).  For L = 6–8 they are *found,
  not counted*: substitution minima reached by `longwords.descend` from the observed
  types, then kept only if no transposition / insertion / deletion neighbour is
  lower (a full-move minimum is necessarily a substitution minimum, so this finds all
  full-move minima among the substitution minima reachable from the data).

## 7. Estimation uncertainty (honesty of the L = 6–8 cells)

The L ≤ 5 contributions to P_bin and P_prior are exact.  The L = 6–8 contributions are
Monte-Carlo estimates from the same importance draws; their standard errors are
propagated to z_bin and reported per cell.  The **effective sample size** of the
importance sampler is computed and stored per length (`models.final.ess_by_L`): it is
~1140 for L = 6, ~76 for L = 7 and ~10 for L = 8 (out of 2·10⁶ samples).  The tiny
L = 8 (and marginal L = 7) ESS is the honest consequence of the prescribed mixture
proposal (independent letters + empirical words) against a heavily concentrated L = 8
model (90 % of whose mass is on strings never observed): the proposal's tails are
thinner than the model's, so a few samples carry enormous weights.  Cells whose mass is
mostly from a length with ESS < 100 are therefore flagged low-confidence, and there the
reported standard error is a *lower bound*, not a calibrated error.  This is the
analogue of the S&B tab's "values are exact" caveat, inverted: here the L ≤ 5 cells are
exact, the L = 6 cells are well estimated, and the L = 7–8-dominated cells are
estimates with honest (lower-bound) error bars.

## 8. Results (in `nm_landscape.json` / `.npz`)

The final model's grand-canonical entropy is **12.60 bits** (from 24.15 bits for the
uniform prior).  The top-2 bigram-profile plane carries **9.0 %** of the total
bigram-profile variance under the final model; axis 1 is the content axis
(word-initial *t*, *th*, *he*, word-final *e* — the "the/he/that" direction), axis 2
picks up word-final *e*/*s* and *to*.  The projected surface has **12 wells** of
persistence ≥ 1 bit at G = 32, of which **8 are confident**; the six largest basins are
named "of", "be", "to", "is", "the", "in" (the basin's most probable string, see §6).
There are **10,171 strict string-level minima** across all lengths (L ≤ 5 exact,
L ≥ 6 found); the deepest are "the", "to", "of", "and", "in".
Of the 1024 grid cells, **455 are true gaps** (no mass under either estimator); of the
remaining 569, **372 are flagged low-confidence** (some dominated by the low-ESS
L = 7–8 lengths), and the z range is −15.4 … +37 bits (clipped at +40).  The 32×32
grid is a display parameter; the L = 7–8 sparsity is a property of the proposal's thin
tails (ESS ≈ 10–76), not of the grid — a coarser grid would raise the per-cell sample
count but not the fundamental ESS limit, so instead each well's confidence is
aggregated over its whole basin (a well is shown if most of the model mass draining
into it is confidently estimated).

**Resolution slider.**  The static model and the trajectory are precomputed at eight
resolutions G ∈ {8, 12, 16, 24, 32, 48, 64, 96} bins per axis (same projection and
extent at every level; `summary.levels`, trajectory arrays `*_L{G}` in the npz).  At
each level the per-cell words (top-5, exact L ≤ 5 / approximate L ≥ 6), grid-adjacency
wells, and the confidence summary are reported.  As G grows the median per-cell
standard error **rises** (≈ 0.34 bits at G = 8 to ≈ 1.05 bits at G = 96) and the number
of low-confidence cells grows, because the same importance draws are spread over more,
smaller cells — the honest behaviour, reported rather than hidden.  The move-set
adjacency is kept only for the reference G = 32 wells; other levels use the geometric
8-neighbour grid adjacency (recomputing the move-set adjacency at eight resolutions is
prohibitively expensive); each level carries a `wells_adjacency` field
("move_set" | "grid8"), and the G = 32 level's wells/barrier are byte-identical to the
static tab's move-set wells.  The string-minima overlay is the top 200 by probability
(each with x, y, z, L, E); the artifact shows a user-chosen subset.

## 9. Honest limitations of the height

**z is a length-mixed mass ratio, so heights are NOT comparable across lengths.**
z_bin = log₂ P_prior − log₂ P_bin, and the prior is uniform *within each length*, so a
single-string cell of length L has z = −3 − 4.70·L − log₂ P(w) (the −4.70·L is
L·log₂ 26, the prior's per-length configurational entropy).  Two words with the *same*
probability therefore sit at very different heights: "there" (L = 5, P ≈ 1.3·10⁻³)
gives z ≈ −16.9 bits, while "thinking" (L = 8, P ≈ 4.5·10⁻⁵) gives z ≈ −26.2 bits —
the L = 8 word is ~9 bits lower only because it is compared against a 26⁸-cell uniform
prior, not because it is a deeper attractor.  A "well" is a cell the model concentrates
relative to *its own length's* uniform prior; it is not an absolute-energy minimum.

**At fine resolutions, persistence is driven by length mixing and sparse cells, not
real barriers.**  A single-string cell can be a projected local minimum solely because
its neighbours are cells of a *different* length at very different z.  Example: the
L = 5 string "itoto" (P ≈ 9.8·10⁻⁸, so z ≈ −3.2 bits) appears at G = 96 as a well with
~36 bits of persistence because its neighbours are L = 8-dominated cells at
z = +30 … +40 bits; the "saddle" is the length contrast, not a genuine energy barrier
between attractors.  This is intrinsic to a single length-mixed surface.  A more
defensible default would be a **per-length-conditional height** (one surface per
length, z_L = log₂ P_prior(bin|L) − log₂ P_bin(bin|L)) or a **mass-weighted height**
(the minimum over lengths, or the per-length z weighted by that length's mass share);
these are not computed here (they would multiply the precomputed surfaces by 8 and
triple the build cost), and the current surface should be read with the length-mixing
caveat in mind.  The per-well basin mass (`basin_mass`, the fraction of the model's
total probability in the basin) is the robust signal for "is this an attractor": a well
holding 10⁻⁸ of the mass is not an attractor of the language, which is why the artifact
defaults to showing wells above `min_basin_mass_default` = 10⁻³ (0.1 % of the mass),
filterable by a control.

**Word labels for L ≥ 6 are model extrapolation, not corpus words.**  The per-cell top
words come from exact enumeration for L ≤ 5 but from *sampled draws* (the observed
types plus importance samples) for L = 6–8; those entries are flagged `approx` and may
be low-probability pseudo-likelihood artefacts (e.g. "herehere…" repeats).  An `approx`
label is a string the model assigns high probability to, not necessarily a word that
ever occurred in the corpus — see §8's count split for how much of the L ≥ 6 labelling
is unobserved extrapolation.

## 10. References

* G. J. Stephens & W. Bialek, Statistical mechanics of letters in words, Phys. Rev. E
  81, 066119 (2010); arXiv:0801.0253.
* R. W. Zwanzig, High-temperature equation of state by a perturbation method. I.
  Nonpolar gases, J. Chem. Phys. 22, 1420–1426 (1954).
* J. Besag, Statistical analysis of non-lattice data, The Statistician 24, 179–195
  (1975).  [pseudo-likelihood]
* M. Ekeberg, C. Lövkvist, Y. Lan, M. Weigt & E. Aurell, Improved contact prediction
  in proteins: using pseudolikelihoods to infer Potts models, Phys. Rev. E 87, 012707
  (2013).  [symmetric PLM with Gaussian prior]
* H. Edelsbrunner, D. Letscher & A. Zomorodian, Topological persistence and
  simplification, Discrete Comput. Geom. 28, 511–533 (2002).  [sublevel-set
  persistence]
* R. M. Neal, Annealed importance sampling, Stat. Comput. 11, 125–139 (2001).
  [unbiased normalising-constant estimation; the exact Z_L here is from split
  enumeration, not AIS]
* B. Widom, Some topics in the theory of fluids, J. Chem. Phys. 39, 2808–2812 (1963).
  [grand-canonical chemical potentials]
