# Three-body interactions in four-letter words

Code: `langfit/potts.py` (model, fits, CV), `scripts/ternary_study.py` (study),
`tests/test_potts.py`. Output: `data/derived/ternary_study.json` (all numbers below are
read from it; JSON paths are given with each table), run log
`data/derived/ternary_study.log`, per-job cache `data/derived/ternary_study.jobs.jsonl`.

## 1. Question

Stephens & Bialek (S&B; Phys. Rev. E 81, 066119, 2010 — the paper that introduced
pairwise maximum-entropy models of letters in words) found that a model with only
pairwise letter–letter couplings captures ~92% of the multi-information of four-letter
English words. The frozen four-letter study (`data/derived/maxent_study.json`)
reproduced this on Austen and found, in passing, that the order-3 model already equals
the plug-in entropy of the sample, S_3 = S_full to 5 decimals, at N ≈ 1.4·10^5 tokens.
That equality can be read in two ways: (i) three-body terms complete the description of
four-letter words, or (ii) at that N the 4 triplet marginals nearly determine the
sample (a sample of ~10^3 word types leaves little freedom beyond its triplet
marginals), so the near-equality is a finite-sample effect. This study separates the two readings with up to
1.6·10^8 Twitter tokens (all 41 shards) and with held-out code lengths.

## 2. Formalism

**Model.** A word s = (s_1..s_4), s_i ∈ {a..z}, is a configuration of L = 4 Potts spins
with A = 26 states. For a set C of cliques (subsets of positions) the maximum-entropy
distribution consistent with the empirical clique marginals μ_c is the Boltzmann law

  P_C(s) = Z^{-1} exp Σ_{c∈C} V_c(s_c),   E(s) = −log₂ P_C(s)

(Jaynes, Phys. Rev. 106, 620, 1957 — the maximum-entropy foundation of statistical
mechanics). Order k means C = all k-subsets: k = 1 single-site fields, k = 2 pair
couplings (S&B), k = 3 three-body couplings V_ijk(a,b,c). The triplet terms are the
direct analogue of p-spin (p = 3) interactions in spin glasses: an energy contribution
that depends jointly on three letters and is not a sum of pair energies.

**Hierarchy.** Let S_k be the entropy of the order-k maximum-entropy model and
S_full = S_4 the entropy of the sample itself. Then S_1 ≥ S_2 ≥ S_3 ≥ S_4, and the
multi-information I_N = S_1 − S_4 decomposes exactly into connected informations
I^(k) = S_{k−1} − S_k = D_KL(P_k ‖ P_{k−1}) (Schneidman, Still, Berry & Bialek,
Phys. Rev. Lett. 91, 238701, 2003 — the definition of connected information used
throughout neural-population and sequence studies). Amari (IEEE Trans. Inf. Theory 47,
1701, 2001 — the information-geometric foundation of this decomposition) showed that
the P_k are m-projections onto a nested hierarchy of exponential families, for which
the KL divergence obeys a generalised Pythagorean relation,
D_KL(P_4 ‖ P_1) = Σ_k D_KL(P_k ‖ P_{k−1}). This is an identity between KL divergences,
not a decomposition into Euclidean lengths (Amari relates squared Fisher distance to
KL only infinitesimally). For the specified hierarchy "all marginals of each order"
the split I_N = I^(2) + I^(3) + I^(4) is unique; increments from adding individual
cliques one at a time are not, and depend on which cliques are already included.
The quantity of this study is I^(3) = S_2 − S_3, the
information carried by three-letter correlations not implied by pairwise ones, and its
share I^(3)/I_N of all letter–letter dependence.

**Why 4 triplet cliques suffice to contain all pairs.** For L = 4 the triplets are
{123},{124},{134},{234}. Each pair {i,j} lies in exactly L − 2 = 2 triplets, so
matching the triplet marginals matches every pair and single-site marginal (they are
sums of the triplet marginals); the order-3 maximum-entropy family is therefore
spanned by the 4 triplet tensors alone, since any pair or field term can be written
as a triplet tensor that ignores one argument. The implementation uses exactly these 4
unrestricted, over-complete 26^3 tensors. Every cluster of 3 positions is
a clique, so order 3 is the last level before the saturated model (the order-4 model
is the empirical distribution itself, one parameter per word).

**Parameter counts after gauge fixing.** Clique tensors are over-complete: adding a
function of s_i alone to V_ij and subtracting it from V_i leaves P unchanged. Two
standard gauges remove this redundancy (Ekeberg et al., Phys. Rev. E 87, 012707,
2013 — gauge-fixing standard in direct-coupling analysis): the zero-sum gauge (every
tensor sums to zero over each argument, their Eq. 8) and the lattice-gas gauge (every
entry involving a reference letter is zero, their Eq. 9). In either, a k-body tensor
has (A−1)^k free entries, and the identifiable order-k model is fields + pure pairs +
… + pure k-body terms with Σ_{j≤k} C(L,j)(A−1)^j parameters. For order 3 that is
100 field + 3,750 pair + 62,500 pure-triplet directions = 66,350; the 4 raw triplet
tensors (4·26^3 = 70,304 entries) span the same family, with the remainder gauge:

| order | 1 | 2 | 3 | 4 (full) |
|---|---|---|---|---|
| free parameters | 100 | 3,850 | 66,350 | 456,975 = 26^4 − 1 |

(`meta.params_gauge_fixed`). Going from pairs to triplets multiplies the parameter
count by 17 for an expected gain I^(3) that, at fixed N, the data can only partly
resolve; the in-sample entropies (exact mode, every observed cell free, unseen cells
forbidden) therefore must be read together with held-out code lengths.

**When are three-body terms physically warranted?** Pairwise models are sufficient
for many biological networks (Schneidman, Berry, Segev & Bialek, Nature 440, 1007,
2006 — the result that made pairwise maximum-entropy models standard in neuroscience).
Merchan & Nemenman (J. Stat. Phys. 162, 1294, 2016 — a general argument for pairwise
sufficiency) showed with simulated random Ising networks with p-spin (p > 2)
interactions that pairwise maximum-entropy models fit such densely interacting
networks well in certain regimes, so pairwise sufficiency need not signal a pairwise
Hamiltonian; conversely Ganmor, Segev & Schneidman (PNAS 108, 9679, 2011 — the
counter-example that higher-order terms matter for larger populations) found that
for ~100 retinal neurons pairwise models, accurate for small groups, are no longer
sufficient, and that a sparse set of higher-order interactions, learnable because the
code is sparse, is needed. Spelling is a
sparse, strongly constrained system (≈ 10^4–10^5 used strings out of 4.6·10^5), closer
to the second regime: one expects three-body terms wherever a morpheme or orthographic
rule spans three letters (e.g. "-ght", "-tch", "wh-" + vowel). The test here is
operational: three-body terms are warranted to the extent their information gain
survives cross-validation and grows (rather than vanishes) with N.

**Ranking triplet couplings.** Two complementary measures per clique c:
(a) D_c = D_KL(μ_c ‖ P_2|_c), the divergence of the empirical triplet marginal from
the pairwise model's prediction of it: how much clique c's statistics are not implied
by pairs. Since P_{2+c} (pairs plus clique c) has marginal μ_c on c, data processing
gives D_c ≤ D_KL(P_{2+c} ‖ P_2) = S_2 − S_{2+c}: D_c is a lower bound on (b) the
exact entropy drop when c alone is added to all pairs, the "gain of inducing one
feature" of Della Pietra, Della Pietra & Lafferty
(IEEE Trans. PAMI 19, 380, 1997 — the standard feature-induction criterion for random
fields); and held-out, the drop in nested-CV code length when only c is added. Within
a clique the letter triples are ranked by their contribution μ log₂(μ/P_2) to D_c.

## 3. Methods

**Estimator.** `potts.fit` implements the same estimator and stopping rule
as the frozen `maxent.fit` (S&B exact mode; Gaussian-prior map mode with
λ = 1/(2σ²N)), with dense tensor algebra on the 26^L grid instead of a sparse feature
matrix: log-weights by broadcasting clique tensors, marginals by memoised axis sums,
Hessian–vector products Cov_P[f_c, f·v] by the same two operations. In exact mode,
when the allowed support (strings with no forbidden cell) is small, the same
operations run on the support only (gathers and bincounts). The trust-region Newton–
Krylov solver runs in diagonally rescaled variables (unit Hessian diagonal at the
data), followed by preconditioned, backtracked Newton polish. Stopping rule:
exact mode — every marginal cell within 1% of its sampling SE, and an *estimated*
entropy error (½·Newton decrement² + |g·θ|) within 1% of the Basharin (Theory
Probab. Appl. 4, 333, 1959 — the first-order sampling SD of plug-in entropy) entropy
SE, floored at the Miller (1955) bias scale. The Newton-decrement term estimates the
dual gap from the local quadratic model; it is not a bound and can understate the
true error (in an analytically solvable boundary case, `tests/test_potts.py::
test_entropy_error_estimate_boundary`, the true error is 1.03× the estimate; we
treat a factor 2 as the practical margin). Map mode — the gradient of the penalised
objective, μ_model − μ_emp + V/(σ²N), within 1% of each cell's SE: this is
stationarity of the posterior, not marginal matching (shrinkage deliberately leaves
model and empirical marginals apart; `max_marginal_error` reports that gap
separately). Tests (`tests/test_potts.py`) check `potts.fit` against
`maxent.fit` for orders 1–3 in both modes: to 1e-6 bits on a synthetic interior
problem, and on Austen (couplings at −∞) within the sum of the two fits' estimated
entropy errors — a practical tolerance, not a certificate (observed differences
9.2e-7 and 1.3e-5 bits vs tolerances 8.5e-6 and 4.8e-5 for orders 2 and 3).

**Data.** Twitter: `enryu43/twitter100m_tweets`, all 41 shards, cleaned, fastText
English filter, exact duplicates removed per year (`langfit/social.py`), counted by
`scripts/ingest_social.py --lengths 1-20 --period year`
(`data/derived/twitter_year_L1-20_all.pkl.gz`). Austen, Gutenberg and Bluesky 2024
exactly as in the four-letter study (`maxent_study.load_corpora`). Corpora: Austen,
Gutenberg, Bluesky, pooled Twitter, and every Twitter year with at least the common
N = 142,614 tokens (the Austen token count, the common size of the four-letter study).

**In-sample curves.** For every corpus, S_1, S_2, S_3, S_full in exact mode on a
half-decade grid N = N_c·10^{k/2}, k = −4, −3, … (two decades below the common N up
to the full corpus; the half-decade step is the convention used for σ²), with 5
hypergeometric subsamples per point up to N_c, one above (`meta.curve_reps`: the
subsample SE of I^(3) is ≤ 0.04 bits already at N = 1,426, ≤ 0.003 at N_c, and falls as N^{-1/2};
the machine was shared, so the run used 1 worker), and the full corpus once.

**Held-out.** Nested CV exactly as in the four-letter study (5 outer token folds ×
4 inner folds, σ² on a half-decade grid extended until interior; scored on outer test
tokens never used for fitting or choosing σ²) for orders 1, 2, 3 and for each "pairs +
one triplet" model (at the order-3 σ² of the same outer fold): Austen, Gutenberg, the
first and last eligible Twitter years, Bluesky and pooled Twitter at N_c; pooled
Twitter at 10 N_c = 1,426,140 (a planned 100 N_c layer was dropped when the shared
machine ran out of memory; `meta.cv`).

**Landscape.** Strict local minima under single-letter substitution (S&B's
definition, `maxent.local_minima`) of exact-mode P_2 and P_3 at N_c and at the full
corpus, and of map-mode P_2 and P_3 (each at the median nested-CV σ²) on the CV
subsamples, where every string has finite energy so unseen acronyms have defined
ranks. Probe words: tmrw, tldr, html, lmao, asap, nsfw, imho, omfg, rofl, smth plus the
four-letter study's slang/control/typo probes.
Fits: every fit in the JSON met the stopping rule above (`converged` fields
in `curves`, `detail`, `cv`, `map_landscape`). All nested-CV σ² optima are interior
for orders 2 and 3; for order 1 on Austen and Gutenberg the held-out loss is flat at
large σ² and the grid extension hit its cap in some folds (`cv.<corpus>.1.all_interior
= false`); the independent model has only 100 free parameters and every letter is
observed at every position, so the prior is immaterial there (H_1 equals S_1 to
≤ 0.003 bits).

## 4. Results

### 4.1 The hierarchy at the common N and on whole corpora

Bits per token (`curves.<corpus>`, the entry with `N = 142614` and the last entry).
I^(3) = S_2 − S_3; I_N = S_1 − S_full; S_3 − S_full is the four-body connected
information I^(4) measured in-sample.

| corpus | tokens | S_1 | S_2 | S_3 | S_full | I^(3) | I^(3)/I_N | S_3−S_full | (full corpus) S_3−S_full | I^(3) | I^(3)/I_N |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Austen | 142,614 | 14.385 | 7.651 | 7.045 | 7.045 | 0.606 | 0.083 | 0.0000 | 0.0000 | 0.606 | 0.083 |
| Gutenberg | 401,237 | 14.472 | 8.609 | 7.691 | 7.690 | 0.918 | 0.135 | 0.0010 | 0.0029 | 0.915 | 0.135 |
| Bluesky | 17,904,714 | 14.855 | 9.266 | 8.051 | 8.039 | 1.215 | 0.178 | 0.0128 | 0.0382 | 1.163 | 0.172 |
| Twitter (pooled) | 156,879,957 | 14.991 | 9.452 | 8.176 | 8.157 | 1.277 | 0.187 | 0.0183 | 0.0501 | 1.199 | 0.177 |
| Twitter 2009 | 268,431 | 15.174 | 9.782 | 8.399 | 8.384 | 1.384 | 0.204 | 0.0144 | 0.0206 | 1.370 | 0.202 |
| Twitter 2010 | 488,134 | 15.255 | 9.929 | 8.498 | 8.481 | 1.431 | 0.211 | 0.0175 | 0.0308 | 1.399 | 0.207 |
| Twitter 2011 | 991,269 | 15.199 | 9.887 | 8.478 | 8.458 | 1.409 | 0.209 | 0.0198 | 0.0376 | 1.363 | 0.204 |
| Twitter 2012 | 1,651,284 | 15.160 | 9.819 | 8.432 | 8.411 | 1.386 | 0.205 | 0.0211 | 0.0427 | 1.334 | 0.199 |
| Twitter 2013 | 2,371,721 | 15.146 | 9.769 | 8.413 | 8.390 | 1.356 | 0.201 | 0.0229 | 0.0491 | 1.296 | 0.193 |
| Twitter 2014 | 3,064,869 | 15.124 | 9.686 | 8.352 | 8.332 | 1.334 | 0.196 | 0.0199 | 0.0467 | 1.275 | 0.189 |
| Twitter 2015 | 3,827,766 | 15.104 | 9.621 | 8.299 | 8.281 | 1.322 | 0.194 | 0.0183 | 0.0463 | 1.262 | 0.186 |
| Twitter 2016 | 4,709,006 | 15.098 | 9.617 | 8.292 | 8.273 | 1.325 | 0.194 | 0.0190 | 0.0477 | 1.257 | 0.186 |
| Twitter 2017 | 5,731,737 | 15.077 | 9.557 | 8.252 | 8.233 | 1.305 | 0.191 | 0.0190 | 0.0477 | 1.241 | 0.183 |
| Twitter 2018 | 8,536,338 | 15.018 | 9.426 | 8.158 | 8.141 | 1.268 | 0.184 | 0.0171 | 0.0451 | 1.202 | 0.176 |
| Twitter 2019 | 10,567,367 | 14.990 | 9.380 | 8.128 | 8.112 | 1.252 | 0.182 | 0.0164 | 0.0454 | 1.189 | 0.174 |
| Twitter 2020 | 15,344,933 | 14.944 | 9.265 | 8.047 | 8.033 | 1.218 | 0.176 | 0.0145 | 0.0434 | 1.154 | 0.168 |
| Twitter 2021 | 20,371,235 | 14.949 | 9.314 | 8.072 | 8.057 | 1.242 | 0.180 | 0.0149 | 0.0460 | 1.174 | 0.172 |
| Twitter 2022 | 43,487,839 | 14.952 | 9.374 | 8.112 | 8.096 | 1.263 | 0.184 | 0.0158 | 0.0477 | 1.194 | 0.176 |
| Twitter 2023 | 35,442,252 | 14.944 | 9.443 | 8.158 | 8.141 | 1.285 | 0.189 | 0.0167 | 0.0475 | 1.215 | 0.180 |

Twitter 2006–2008 have fewer than N_c four-letter tokens and are excluded
(`meta.excluded_years`). Sampling SEs of these entries at N_c (5 subsamples) are
≤ 0.003 bits for I^(3) (`I3_se`); the full-corpus entries are single evaluations.

Check against the four-letter study: Austen S_1 = 14.385, S_2 = 7.651 here, identical
to its cross-corpus Austen rows (`maxent_study.json: corpora[name=Austen]`); its
replication (S_ind 14.323, S_2 7.549) uses the dictionary-filtered Austen sample.

### 4.2 Sample-size dependence: near-saturation S_3 ≈ S_full is a small-sample effect

Pooled Twitter, all 41 shards (`curves.Twitter`; 5 subsamples below N_c, mean ± SE):

| N | S_1 | S_2 | S_3 | S_full | I^(3) | I^(3)/I_N | S_3−S_full | Miller bias of S_full | types |
|---|---|---|---|---|---|---|---|---|---|
| 1,426 | 14.978 | 8.542 | 7.579 | 7.578 | 0.964 ± 0.039 | 0.130 | 0.0010 | 0.2125 | 421 |
| 4,510 | 14.988 | 9.051 | 7.832 | 7.832 | 1.218 ± 0.008 | 0.170 | 0.0007 | 0.1244 | 779 |
| 14,261 | 14.983 | 9.325 | 8.013 | 8.012 | 1.312 ± 0.005 | 0.188 | 0.0005 | 0.0720 | 1425 |
| 45,099 | 14.986 | 9.404 | 8.098 | 8.097 | 1.305 ± 0.004 | 0.189 | 0.0019 | 0.0397 | 2482 |
| 142,614 | 14.991 | 9.452 | 8.176 | 8.157 | 1.277 ± 0.002 | 0.187 | 0.0183 | 0.0229 | 4534 |
| 450,985 | 14.984 | 9.464 | 8.216 | 8.186 | 1.247 | 0.183 | 0.0298 | 0.0133 | 8288 |
| 1,426,140 | 14.989 | 9.468 | 8.247 | 8.205 | 1.221 | 0.180 | 0.0427 | 0.0075 | 14882 |
| 4,509,851 | 14.989 | 9.475 | 8.267 | 8.218 | 1.207 | 0.178 | 0.0493 | 0.0042 | 26011 |
| 14,261,400 | 14.990 | 9.478 | 8.275 | 8.224 | 1.202 | 0.178 | 0.0508 | 0.0022 | 43082 |
| 45,098,507 | 14.990 | 9.475 | 8.276 | 8.225 | 1.199 | 0.177 | 0.0504 | 0.0011 | 67556 |
| 142,614,000 | 14.990 | 9.476 | 8.277 | 8.227 | 1.199 | 0.177 | 0.0501 | 0.0005 | 99718 |
| 156,879,957 | 14.990 | 9.476 | 8.277 | 8.227 | 1.199 | 0.177 | 0.0501 | 0.0005 | 102664 |

Three regimes:

1. **N ≲ 5·10^4 (≲ 2.5·10^3 types): S_3 ≈ S_full.** Up to N = 45,099 the four-body
   residual S_3 − S_full is ≤ 0.002 bits for pooled Twitter and ≤ 0.0052 bits in every
   corpus (maximum: Twitter 2010 at 45,099, `curves.<corpus>[].I4`). This is the
   regime of the four-letter study and of S&B. It is not an identity — triplet
   constraints generally leave a positive four-body residual — but with ~10^3
   observed types the 4 triplet marginals (only cells seen in the sample are free)
   come close to determining the empirical distribution: the near-equality is a
   property of samples of this size, not evidence about English.
2. **N ≈ 10^5–5·10^6: the saturation lifts.** S_3 − S_full grows from 0.002 to
   0.049 bits while the plug-in bias scale of S_full falls below it (0.023 at N_c,
   0.004 at 4.5·10^6): the four-body connected information becomes resolvable. The
   same happens in every corpus large enough (full-corpus column of 4.1: 0.038
   Bluesky, 0.043–0.049 for each Twitter year ≥ 1.6·10^6 tokens); Austen
   (1.4·10^5 tokens, 966 types) and Gutenberg (4·10^5, 2,300 types) never leave
   regime 1.
3. **N ≳ 1.4·10^7: apparent plateau of the subsampling curve.** From 1.43·10^7 to
   1.57·10^8 tokens, S_1, S_2, S_3, S_full and I^(3) of pooled Twitter change by
   ≤ 0.003 bits per half decade (between 4.5·10^6 and 1.4·10^7 the changes are
   still 0.008 in S_3, 0.0065 in S_full, −0.005 in I^(3)), even though the number of
   types grows from 43k to 103k. These curves subsample one finite corpus: they show
   stability with respect to N within it, not convergence to a population value,
   and no extrapolation in N was fitted. The Miller term in the table is a bias scale
   for the observed support, not a correction for unseen types.

I^(3) itself rises at small N (S_2 is pulled down when the pairs alone nearly
determine a tiny sample) to a maximum of 1.31 bits near 1.4·10^4 tokens, then falls
as S_3 stops being pinned to the sample, reaching **I^(3) = 1.199 bits = 17.7% of the
multi-information on the full pooled Twitter corpus** (apparent plateau; Bluesky full
corpus 1.163 bits, 17.2%). The full-corpus decomposition of Twitter's I_N = 6.763
bits is I^(2) = 5.514 (81.5%), I^(3) = 1.199 (17.7%), I^(4) = 0.050 (0.7%). S&B's
"fraction captured by pairs" is therefore 0.815 for Twitter, against 0.917 for
Austen at its full size (I^(2)/I_N = 6.734/7.340).

### 4.3 Held-out code lengths: the triplet gain generalises

Nested CV, map mode (`cv.<corpus>.<N>`; mean ± s/√5 over 5 outer folds; differences
are paired by fold). The ± values describe fold-to-fold dispersion only: the five
outer training sets overlap (any two share 75% of their tokens), so s/√5 is not a
calibrated standard error and gives no sampling uncertainty for the corpus or the
population.

| corpus | N | H_1 | H_2 | H_3 | H_1−H_2 | H_2−H_3 | in-sample I^(3) at same N | median σ² order 2 / 3 |
|---|---|---|---|---|---|---|---|---|
| Austen | 142,614 | 14.386 ± 0.008 | 7.678 ± 0.009 | 7.076 ± 0.007 | 6.709 ± 0.007 | 0.602 ± 0.004 | 0.606 | 100 / 10 |
| Gutenberg | 142,614 | 14.475 ± 0.004 | 8.629 ± 0.010 | 7.741 ± 0.005 | 5.846 ± 0.011 | 0.888 ± 0.006 | 0.918 | 31.6 / 10 |
| Twitter 2009 | 142,614 | 15.177 ± 0.010 | 9.840 ± 0.014 | 8.574 ± 0.011 | 5.337 ± 0.015 | 1.266 ± 0.010 | 1.384 | 10 / 3.16 |
| Twitter 2023 | 142,614 | 14.947 ± 0.009 | 9.469 ± 0.015 | 8.326 ± 0.014 | 5.478 ± 0.008 | 1.143 ± 0.008 | 1.285 | 3.16 / 3.16 |
| Bluesky | 142,614 | 14.847 ± 0.003 | 9.312 ± 0.008 | 8.218 ± 0.007 | 5.535 ± 0.009 | 1.095 ± 0.006 | 1.215 | 10 / 3.16 |
| Twitter (pooled) | 142,614 | 14.983 ± 0.012 | 9.493 ± 0.022 | 8.355 ± 0.012 | 5.489 ± 0.023 | 1.138 ± 0.009 | 1.277 | 3.16 / 3.16 |
| Twitter (pooled) | 1,426,140 | 14.989 ± 0.002 | 9.473 ± 0.004 | 8.294 ± 0.003 | 5.516 ± 0.002 | 1.179 ± 0.002 | 1.221 | 3.16 / 3.16 |

- Three-body terms lower the held-out code length in every corpus, by 0.60 (Austen)
  to 1.27 (Twitter 2009) bits per word, and in every one of the 35 outer folds
  (`cv.<corpus>.<N>.gain_2to3.per_fold`); the fold dispersion is small compared with
  the gain, but it is descriptive, not a significance test.
- The held-out gain is 89–99% of the in-sample I^(3) at the same N, and the gap
  closes with data: pooled Twitter 1.138 of 1.277 bits at N_c, 1.179 of 1.221 at
  10 N_c — already 98% of the full-corpus I^(3) = 1.199. With 66,350 gauge-free
  parameters the order-3 model needs ~10^6 tokens to express its information.
- The selected prior is narrower for order 3 than for order 2 in the literary
  corpora (σ² = 10 vs 31.6–100): three-body couplings need more shrinkage, as
  expected for a 17-fold larger parameter count at the same N.
- H_3 at 10 N_c (8.294) is within 0.017 bits of the full-corpus in-sample S_3 = 8.277:
  at a million tokens the map-mode order-3 code is already close to the in-sample
  order-3 entropy of the whole corpus.

### 4.4 Which triplets carry the gain

Held-out gain of adding a single triplet clique to all pairs (`cv.<corpus>.<N>.
single_clique.<c>.gain_vs_pairs`, positions numbered 1–4), and the in-sample
divergence D_c = D_KL(μ_c ‖ P_2|_c) at N_c (`curves.<corpus>[N_c].kl_clique_bits`):

| corpus | N | {1,2,3} | {1,2,4} | {1,3,4} | {2,3,4} | sum | D_c for {123},{124},{134},{234} |
|---|---|---|---|---|---|---|---|
| Austen | 142,614 | 0.298 ± 0.003 | 0.289 ± 0.003 | 0.274 ± 0.003 | 0.139 ± 0.002 | 0.999 | 0.264, 0.232, 0.242, 0.107 |
| Gutenberg | 142,614 | 0.355 ± 0.003 | 0.345 ± 0.004 | 0.405 ± 0.003 | 0.218 ± 0.003 | 1.323 | 0.333, 0.304, 0.381, 0.200 |
| Twitter 2009 | 142,614 | 0.492 ± 0.004 | 0.465 ± 0.004 | 0.530 ± 0.003 | 0.349 ± 0.004 | 1.836 | 0.496, 0.455, 0.527, 0.343 |
| Twitter 2023 | 142,614 | 0.437 ± 0.004 | 0.410 ± 0.002 | 0.482 ± 0.003 | 0.321 ± 0.003 | 1.650 | 0.448, 0.407, 0.486, 0.327 |
| Bluesky | 142,614 | 0.434 ± 0.003 | 0.392 ± 0.004 | 0.457 ± 0.002 | 0.302 ± 0.003 | 1.586 | 0.437, 0.380, 0.454, 0.297 |
| Twitter (pooled) | 142,614 | 0.438 ± 0.004 | 0.417 ± 0.005 | 0.480 ± 0.005 | 0.322 ± 0.003 | 1.657 | 0.447, 0.408, 0.486, 0.322 |
| Twitter (pooled) | 1,426,140 | 0.446 ± 0.001 | 0.426 ± 0.001 | 0.492 ± 0.001 | 0.330 ± 0.001 | 1.694 | 0.433, 0.393, 0.467, 0.310 (at 10 N_c) |

- On the social corpora and Gutenberg the ranking is the same by all three measures
  (held-out single-clique gain, D_c, and the exact in-sample drop S_2 − S_{2+c} in
  `detail.<corpus>.<tag>.triplets[].drop_bits`, which exceeds D_c by 0.02–0.07 bits,
  as the bound D_c ≤ drop requires): {1,3,4} > {1,2,3} > {1,2,4} > {2,3,4}. On
  Austen the held-out gain ranks {1,2,3} > {1,2,4} > {1,3,4} > {2,3,4}, while D_c
  puts {1,3,4} second.
- **The final three letters {2,3,4} carry the least** (0.32 bits on Twitter) and the
  non-contiguous clique {1,3,4} the most (0.48–0.49 bits); the contiguous prefix
  {1,2,3} is second. [Interpretation] The word-final trigram is largely implied by
  its two adjacent pairs (common endings such as -ere, -hat are near the pairwise
  prediction, ratio 1.1–1.4), whereas {1,3,4} has only one adjacent pair and binds
  first letter to ending across the vowel slot (m_et, d_es, w_nt, p_rt).
- **Non-additivity.** The four single-clique held-out gains sum to 1.59–1.84 bits on
  the social corpora, but the model with all four triplets gains only 1.10–1.27 (Austen:
  sum 0.999 vs joint 0.602). This is aggregate non-additivity under the stated fitting
  procedure (each "pairs + c" model uses the order-3 σ² of its fold, not its own
  selected prior); four separately fitted improvements are not a decomposition, and
  no share of the overlap is attributable to a particular clique.
- **The letter triples** with the largest contributions to D_c on pooled Twitter
  (`detail.Twitter.full.triplets[].top_cells`; *marginal* ratio μ_c/P_2|_c in
  brackets — the cell summed over all 26 completions of the free position):
  {1,2,3}: las_ (3.8), tim_ (1.7), gam_ (5.1), sur_ (3.5); {1,2,4}: wa_t (2.5),
  la_t (3.0), ci_y (13.4), we_e (2.9); {1,3,4}: m_et (160), d_es (6.2), w_nt (2.0),
  p_rt (3.1); {2,3,4}: _oes (23.8), _eam (2.7), _eet (10.8). Word-level check
  (`scripts/ternary_cells.py` → `data/derived/ternary_cells.json`): for 11 of these
  15 cells one word holds ≥ 94% of the cell's tokens (last, time, game, sure, city,
  were, meet, does, part, team), while wa_t (want 71%), w_nt (want 82%), _oes (does
  72%) and _eet (meet 84%) are shared among several words. The word-level ratio
  P_emp/P_2 of the dominant word is 1.9 (time) to 244 (meet), 4–6 for most, and need
  not equal the cell ratio (tim_ 1.7 vs time 1.9; ci_y 13.4 vs city 52.8).
  [Interpretation] The largest triplet residuals are therefore mostly carried by
  single frequent words — illustrative of a lexical component, not a measurement of
  the sparsity of the gauge-fixed three-body couplings, which was not computed. The
  same cells head the lists in every Twitter year and in Bluesky; Austen's are
  period vocabulary (sid_ "side", dar_ "dare", hi_t "hint", tu_n "turn", t_rn "turn").

### 4.5 Energy landscape: order 2 → order 3

Exact mode (`detail.<corpus>.<common|full>.landscape`); strict local minima under
single-letter substitution; "unseen mass" = model probability on strings absent from
the sample; residual = token-weighted rms of log₂(P_emp/P_model) over seen words.

| corpus | N | minima P_2 → P_3 | gained / lost | token share in minima | unseen mass | rms residual (bits) |
|---|---|---|---|---|---|---|
| Austen | 142,614 | 152 → 260 | 143 / 35 | 0.662 → 0.724 | 0.1597 → 0.0000 | 1.93 → 0.00 |
| Gutenberg | 401,237 | 142 → 392 | 299 / 49 | 0.535 → 0.666 | 0.2055 → 0.0006 | 2.04 → 0.18 |
| Bluesky | 142,614 | 158 → 761 | 659 / 56 | 0.573 → 0.702 | 0.2708 → 0.0034 | 2.55 → 0.35 |
| Bluesky | 17,904,714 | 154 → 1251 | 1155 / 58 | 0.574 → 0.701 | 0.0691 → 0.0031 | 2.51 → 0.56 |
| Twitter (pooled) | 142,614 | 142 → 884 | 792 / 50 | 0.549 → 0.697 | 0.2880 → 0.0047 | 2.62 → 0.39 |
| Twitter (pooled) | 156,879,957 | 133 → 1061 | 968 / 40 | 0.551 → 0.694 | 0.0203 → 0.0018 | 2.53 → 0.58 |
| Twitter 2009 | 268,431 | 145 → 967 | 871 / 49 | 0.527 → 0.680 | 0.2798 → 0.0047 | 2.71 → 0.43 |
| Twitter 2023 | 35,442,252 | 137 → 1209 | 1125 / 53 | 0.547 → 0.695 | 0.0568 → 0.0035 | 2.55 → 0.59 |

Map mode at the CV-selected σ² (`map_landscape.<corpus>.<N>`), where every string has
finite energy:

| corpus | N | minima P_2 → P_3 | gained / lost | unseen mass |
|---|---|---|---|---|
| Austen | 142,614 | 159 → 332 | 219 / 46 | 0.162 → 0.003 |
| Gutenberg | 142,614 | 127 → 418 | 331 / 40 | 0.227 → 0.007 |
| Twitter 2009 | 142,614 | 141 → 735 | 643 / 49 | 0.307 → 0.030 |
| Twitter 2023 | 142,614 | 117 → 700 | 619 / 36 | 0.293 → 0.031 |
| Bluesky | 142,614 | 141 → 656 | 560 / 45 | 0.273 → 0.029 |
| Twitter (pooled) | 142,614 | 122 → 721 | 632 / 33 | 0.292 → 0.032 |
| Twitter (pooled) | 1,426,140 | 126 → 795 | 704 / 35 | 0.179 → 0.014 |

- **The landscape becomes much more rugged.** The number of minima is nearly
  constant across corpora and N for P_2 (117–159) but rises with triplets by a factor
  1.7–8.8 in the rows displayed above (exact and map), and 1.7–10.7 across all
  analysed corpora and sizes (maximum Twitter 2013 full corpus, 1,336/125,
  `detail.Twitter 2013.full.landscape`); for P_3 it grows with N (pooled Twitter
  884 → 1061, exact). Pairwise wells are broad basins that collect many words;
  triplets carve a separate well for most frequent words.
- **Gained minima** are frequent words the pairwise model routes to a neighbour:
  on Twitter love, last, team, most, game, sure, part, full, does, call, feel, done
  (`gained_minima`). **Lost minima** are of two kinds: non-words the pairwise model
  invents from frequent pairs (hald, plue, endy, yogh, vity, nace) and genuine
  words displaced by a stronger neighbour once its three-body term is added (past
  and lost lose to last via la_t / las_). On Austen: made, like, last, felt, mind,
  love gained; make, came, kind, life lost (`lost_minima` lists).
- **The pairwise model generalises by inventing; the three-body model by
  shrinking.** Exact P_2 puts 29% of its mass on unseen strings at N_c (Twitter),
  P_3 0.5%; with the map prior P_3 still keeps 3% there and is the better code
  held-out (4.3). The fraction of tokens sitting in minima rises from 53–57% to 68–70%
  in every social corpus.
- **Acronyms.** Energy rank among all 26^4 strings, map mode, P_2 → P_3 (* = strict
  local minimum of P_3; `map_landscape.<corpus>.<N>.landscape.probes`); n = count in
  the subsample:

| probe | Twitter N_c (n) | Twitter 10 N_c (n) | Bluesky N_c (n) | Austen N_c (n = 0) |
|---|---|---|---|---|
| tmrw | 377,598 → 1,226* (5) | 400,141 → 1,420* (34) | 332,248 → 1,021* (7) | 449,264 → 269,311 |
| tldr | 441,192 → 6,927 (1) | 428,166 → 2,487* (14) | 422,907 → 77,798 (0) | 448,456 → 187,234 |
| html | 381,466 → 1,797* (3) | 316,650 → 1,020* (61) | 347,317 → 1,748* (3) | 396,692 → 212,721 |
| lmao | 12,506 → 410* (36) | 8,737 → 354* (435) | 219 → 137* (176) | 414,741 → 236,837 |
| asap | 19,017 → 448* (30) | 18,261 → 456* (273) | 15,937 → 523* (22) | 170,542 → 375,217 |
| nsfw | 395,896 → 73,975 (0) | 144,655 → 1,644* (26) | 1,657 → 459* (27) | 386,157 → 75,914 |
| omfg | 113,603 → 1,742* (3) | 122,072 → 1,739* (23) | 63,165 → 941* (8) | 159,173 → 196,134 |
| rofl | 131,463 → 86,179 (0) | 123,012 → 8,604 (4) | 88,429 → 5,518 (1) | 97,261 → 11,809 |
| bruh | 54,707 → 879* (9) | 54,921 → 845* (90) | 42,141 → 911* (9) | 10,679 → 405,438 |
| that | 3 → 3* | 3 → 3* | 1 → 1* | 1 → 1* |

  The *fitted* pairwise models place the consonant-cluster acronyms tmrw, tldr, html
  at ranks 3.2·10^5–4.4·10^5 of 4.6·10^5 strings (E ≈ 40–50 bits), at the observed
  counts. This is a property of the pairwise maximum-entropy fit to these corpora,
  not an expressivity limit: a pairwise Potts model can make any single string the
  global minimum (e.g. V_ij = (10/3)(1[s_i = w_i] + 1[s_j = w_j]) for w = tmrw gives
  P(tmrw) ≈ 0.995). Under the empirical pair marginals, which are dominated by
  ordinary words, the matched pair couplings for t?r?, ?mr?, ??rw stay weak. The
  fitted three-body cells lift these acronyms to ranks 10^3–7·10^3
  (the top 0.2–1.5%), and with a few dozen occurrences they
  become strict local minima. Probes unseen in a sample (Twitter rofl, nsfw at N_c;
  everything in Austen) stay outside the top 10^4 under P_3, and in Austen several
  (asap, imho, smth, bruh) rank lower under P_3 than under P_2 — the triplet model
  spends its probability on what it has seen.
  Transposition typos (waht, jsut) are not minima under either model.

### 4.6 Across corpora and years

At the common N, I^(3) orders the corpora Austen (0.606) < Gutenberg (0.918) <
Bluesky (1.215) < Twitter years (1.218–1.431), and its share of I_N follows
(8.3% → 13.5% → 17.8% → 17.6–21.1%). Within Twitter, I^(3) at N_c falls from
1.431 (2010) to 1.218 (2020) and recovers to 1.285 (2023); S_1 and S_2 move together
with it. These year-to-year differences (≈ 0.2 bits) are ≫ their subsampling SE
(≤ 0.003, from 5 draws of the same corpus) but are confounded with the composition of
the dataset (§6).

## 5. What this says about the fitted system

1. **An order-three effective model describes four-letter words accurately; a
   pairwise one does not.** On the full pooled Twitter corpus the connected-
   information hierarchy is I^(2) : I^(3) : I^(4) = 81.5 : 17.7 : 0.7 %: the order-3
   model captures 99.3% of the empirical multi-information and its 1.2-bit gain over
   pairs is ~98% recovered held-out at 1.4·10^6 tokens. The four-body residual
   (0.050 bits) is small but non-zero, so the empirical distribution is not exactly
   of order three. Fitted interactions are effective ones: Schneidman et al. (2003)
   note that marginalisation and noise generate apparent higher-order terms, so they
   do not identify a microscopic or linguistic mechanism. The near-equality
   S_3 ≈ S_full of the four-letter study (and the saturation implicit in S&B's 92%)
   is a property of samples with ≲ 3·10^3 types: it lifts at N ≈ 10^5, and the
   residual reaches an apparent plateau near 0.05 bits within the corpus by
   ~1.4·10^7 tokens.
2. **Pairwise sufficiency depends on the corpus.** For Austen's 966-word vocabulary,
   pairs capture 92% of I_N and the triplet gain is 0.6 bits; for social media
   (larger vocabulary, more abbreviations and non-standard forms) pairs capture
   82% and the triplet gain is 1.1–1.3 bits. In the language of Merchan & Nemenman
   and Ganmor et al., the literary sample is in the regime where pairwise models
   suffice, the social sample in the one where sparse higher-order terms are needed.
3. **The largest triplet residuals are mostly lexical.** For 11 of the 15 top cells
   one frequent word holds ≥ 94% of the cell's tokens, often across a non-contiguous
   clique (m_et → meet, d_es → does, p_rt → part), and that word is under-predicted
   by the pairwise model by a word-level factor of 1.9–244 (§4.4). [Interpretation]
   This is reminiscent of the sparse higher-order structure reported by Ganmor et
   al.; the sparsity of the gauge-fixed couplings themselves was not measured.
4. **Landscape: from broad basins to many narrow wells.** Adding three-body terms
   multiplies the number of local minima by 1.7–10.7 and makes almost every frequent
   word a minimum, removing the pairwise model's invented attractors. Abbreviations
   whose letter pairs are atypical of the corpus (tmrw, html, tldr) go from the far
   tail of the fitted pairwise distribution to distinct wells of the fitted
   three-body distribution.

## 6. Limitations

- **Finite-sample saturation.** In exact mode S_3 is an in-sample quantity: below
  ~5·10^4 tokens it is within 0.0052 bits of S_full in every corpus (not equal by
  construction, but close), so only the large-N part of the curves and the nested-CV
  gains measure more than the sample. The full-corpus I^(3), I^(4) values rest on
  pooled Twitter and Bluesky; the literary corpora never leave the near-saturated
  regime and their I^(3) (in-sample and held-out alike) is a statement about samples
  of that size.
- **Finite corpus.** All N-dependence comes from subsampling one finite corpus; the
  plateau values are full-corpus estimates, not extrapolated population limits.
- **Fold dispersion.** CV ± values are s/√5 over overlapping outer folds: descriptive,
  not calibrated standard errors.
- **Plug-in S_full.** S_full is the plug-in entropy; its bias scale (Miller) is
  listed in 4.2 and is below S_3 − S_full only for N ≳ 3·10^5.
- **Twitter composition.** The dataset keeps each account's ~1,000 most recent
  tweets as of early 2023, so early years come from few, long-lived accounts
  (effective users, inverse Simpson index: 495 in 2009, 922 in 2010 vs 15,187–22,316
  in 2021–2023; `meta.corpora.<year>.effective_users`). Year-to-year trends in I^(3)
  therefore mix change in usage with change in who is sampled, and are not
  interpreted here.
- **Reduced design.** Above N_c the in-sample curves use one subsample per point
  (SE not estimated there, expected ≤ 0.003 bits from the N_c values and N^{-1/2});
  CV at large N only for pooled Twitter at 10 N_c (the 100 N_c layer was dropped on
  the shared machine); σ² is resolved on a half-decade grid only.
- **Probe ranks** depend on σ² (map mode) and on the subsample. In exact mode an
  unseen word has a defined rank whenever it is allowed (e.g. rofl, count 0, rank_2 =
  82,115 at N_c on Twitter); only strings containing a forbidden (unseen) clique cell
  have infinite energy and `rank` = null (rofl under P_3).
- **Lexical vs structural.** The largest triplet residuals are dominated by single
  frequent words (§4.4); at L = 4 a three-body term covers 3/4 of the word, so
  "three-body" and "lexical" cannot be separated here. Longer words (L ≥ 5, where
  triplets are a smaller fraction of the word) are needed for that.

## References (verified)

- Amari, S. Information geometry on hierarchy of probability distributions. IEEE
  Trans. Inf. Theory 47, 1701–1711 (2001). doi:10.1109/18.930911
- Basharin, G. P. On a statistical estimate for the entropy of a sequence of
  independent random variables. Theory Probab. Appl. 4, 333–336 (1959).
  doi:10.1137/1104033
- Della Pietra, S., Della Pietra, V. & Lafferty, J. Inducing features of random
  fields. IEEE Trans. Pattern Anal. Mach. Intell. 19, 380–393 (1997).
  doi:10.1109/34.588021
- Ekeberg, M., Lövkvist, C., Lan, Y., Weigt, M. & Aurell, E. Improved contact
  prediction in proteins: using pseudolikelihoods to infer Potts models. Phys. Rev.
  E 87, 012707 (2013). doi:10.1103/PhysRevE.87.012707
- Ganmor, E., Segev, R. & Schneidman, E. Sparse low-order interaction network
  underlies a highly correlated and learnable neural population code. PNAS 108,
  9679–9684 (2011). doi:10.1073/pnas.1019641108
- Jaynes, E. T. Information theory and statistical mechanics. Phys. Rev. 106,
  620–630 (1957). doi:10.1103/PhysRev.106.620
- Merchan, L. & Nemenman, I. On the sufficiency of pairwise interactions in maximum
  entropy models of networks. J. Stat. Phys. 162, 1294–1308 (2016).
  doi:10.1007/s10955-016-1456-5
- Miller, G. A. Note on the bias of information estimates. In Information Theory in
  Psychology II-B (ed. Quastler, H.), Free Press (1955).
- Schneidman, E., Still, S., Berry, M. J. & Bialek, W. Network information and
  connected correlations. Phys. Rev. Lett. 91, 238701 (2003).
  doi:10.1103/PhysRevLett.91.238701
- Schneidman, E., Berry, M. J., Segev, R. & Bialek, W. Weak pairwise correlations
  imply strongly correlated network states in a neural population. Nature 440,
  1007–1012 (2006). doi:10.1038/nature04701
- Stephens, G. J. & Bialek, W. Statistical mechanics of letters in words. Phys. Rev.
  E 81, 066119 (2010). doi:10.1103/PhysRevE.81.066119; arXiv:0801.0253
