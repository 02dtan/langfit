# Pairwise maximum-entropy models of long words (L = 2 … 20)

Code: `langfit/longwords.py` (model, pseudo-likelihood fit, exact split enumeration,
AIS, per-word landscape), `scripts/length_study.py` (study), `tests/test_longwords.py`.
Output: `data/derived/length_study.json` (every number below is read from it; JSON
paths are given with each table), run log `data/derived/length_study.log`. The study
resumes from its output: delete the JSON to recompute (≈ 3 h on one core).

## 1. Question

The frozen four-letter study fits Stephens & Bialek's pairwise model (S&B; Phys. Rev. E
81, 066119, 2010, doi:10.1103/PhysRevE.81.066119, arXiv:0801.0253 — the paper that
introduced pairwise maximum-entropy models of letters in words) by enumerating all
26^4 strings at every optimisation step. That is impossible beyond L ≈ 5 (26^6 ≈ 3·10^8
states per step, 26^10 ≈ 1.4·10^14). The question is how to fit the *same* model,

  P(s) = Z⁻¹ exp Σ_{i<j} V_ij(s_i, s_j),  s ∈ {a..z}^L,

to words of every length that carries data, and — the user's emphasis — exactly which
quantities then become approximate and how badly.

## 2. Method comparison

All work in the same **model family** (pairwise Potts models of L letters). They do
not in general share an **estimand**. Maximum likelihood minimises the joint
cross-entropy −Σ_s p_data(s) ln P_θ(s) (equivalently, matches the pair marginals: the
maximum-entropy model, S&B's S_2). The PLM implemented here minimises the sum of
conditional cross-entropies −Σ_s p_data(s) Σ_i ln P_θ(s_i | s_\i). If the data were
drawn from a pairwise model both population optima would coincide with it
(Hyvärinen 2006 proves PLM consistency in that correctly specified case). English words
are not exactly pairwise (S_3 < S_2 in the frozen study), so the two optima are
generically different points of the family — this is possible/generic under
misspecification, not logically forced by S_3 < S_2 alone — and what PLM delivers is a
different projection, not a noisy copy of the ML fit. §4.1 measures how different at
L = 4, 5.

| method | what is approximated | consistency | cost | failure modes |
|---|---|---|---|---|
| Maximum pseudo-likelihood (PLM) | joint likelihood replaced by Π_i P(s_i \| s_\i) | consistent if the data are pairwise; otherwise a different projection | O(n_types · L² · 26) per step, no Z | no sampling, so no mixing failure; biased at small N; gives no Z |
| Boltzmann learning by MCMC / PCD | model marginals estimated by Monte Carlo | consistent for ML only if chains mix | O(chains · sweeps · L² · 26) per step | chains trapped in wells ⇒ biased gradients, silently wrong fits |
| Mean field / TAP | log Z replaced by a weak-coupling expansion | not consistent (truncated expansion) | inversion of a (25L)² correlation matrix | strong couplings, multimodality |
| Adaptive cluster expansion (ACE) | entropy expanded in clusters of variables, adaptively truncated by a threshold | exact in the all-cluster limit; error set by the truncation | grows with cluster size: a cluster of k positions needs a k-position inference (26^k states for exact cluster sums) | truncation/convergence of the expansion; at small L the useful clusters approach the full word |
| exact ML (reference) | nothing | consistent for ML | 26^L per step | infeasible for L ≥ 6 |

**PLM.** Besag (J. R. Stat. Soc. D / The Statistician 24, 179, 1975,
doi:10.2307/2987782 — introduced the pseudo-likelihood for Markov random fields)
replaced the joint likelihood by the product of each variable's conditional given all
others. For Potts models each conditional is a 26-way softmax with logits
η_i(a | s) = Σ_{j≠i} V_ij(a, s_j), so objective and gradient are closed form and contain
no partition function. Hyvärinen (Neural Comput. 18, 2283, 2006,
https://pubmed.ncbi.nlm.nih.gov/16907626 — the consistency proof for fully visible Boltzmann
machines) showed the estimator is consistent; Ravikumar, Wainwright & Lafferty (Ann.
Stat. 38, 1287, 2010, doi:10.1214/09-AOS691 — the high-dimensional theory of
neighbourhood regression for Ising models) gave sample-complexity guarantees for its
regularised node-wise form. In inverse statistical physics, Aurell & Ekeberg (Phys.
Rev. Lett. 108, 090201, 2012, doi:10.1103/PhysRevLett.108.090201) showed PLM stays
accurate for strongly coupled Ising systems where mean-field methods fail, and Ekeberg,
Lövkvist, Lan, Weigt & Aurell (Phys. Rev. E 87, 012707, 2013,
doi:10.1103/PhysRevE.87.012707 — plmDCA, now the standard estimator for protein Potts
models) used the symmetric (all conditionals jointly) Potts form with a Gaussian prior,
which is what is implemented here. Reviews: Nguyen, Zecchina & Berg (Adv. Phys. 66,
197, 2017, doi:10.1080/00018732.2017.1341604) and Cocco, Feinauer, Figliuzzi, Monasson
& Weigt (Rep. Prog. Phys. 81, 032601, 2018, doi:10.1088/1361-6633/aa9965).
*Weaknesses*: (i) consistency holds only if the data really come from a pairwise
model. English words do not (the frozen study's S_3 < S_2), so PLM and maximum
likelihood can converge to *different* pairwise models — the pseudo-likelihood
projection and the maximum-entropy (ML) projection of the true distribution. §4.1
finds a PLM–exact-MAP divergence that stays flat from 10⁴ to 10⁶ tokens at L = 4, which
is *consistent with* such a gap; it is not a proof of an asymptotic gap, and nothing
was measured at L ≥ 6. (ii) At small N its couplings are inflated (§4.1: slope 1.17 at
2 tokens per parameter). (iii) The fitted marginals do not match the data, so the
max-ent/ML duality S_2 = training code length holds only approximately. (iv) It gives
no Z, so absolute energies, entropies and held-out likelihoods need a separate
estimator. (v) An observation, not a property established here: at L ≥ 9 the fitted
models over-weight the most probable words (§5.2); a candidate explanation (weak
identification of the relative depths of isolated wells) is a hypothesis tested only
partially (§5.3).

**Boltzmann-machine learning by MCMC.** Ackley, Hinton & Sejnowski (Cognitive Science
9, 147, 1985, doi:10.1207/s15516709cog0901_7 — the original Boltzmann-machine learning
rule) follow the exact likelihood gradient ⟨f⟩_data − ⟨f⟩_model with the model term
estimated by Gibbs sampling; persistent contrastive divergence (Tieleman, ICML 2008,
doi:10.1145/1390156.1390290) keeps the chains alive between parameter updates. The
estimator is consistent *if* the chains equilibrate. Our words are the opposite of that
regime: the fitted landscapes have thousands of strict local minima (§5) separated by
multi-letter barriers, and a single-site Gibbs chain leaves a well of the form
"congratulations" only by passing through many high-energy strings. Mis-mixing does
not announce itself — the fit converges to the wrong couplings with a small gradient
estimate. Parallel tempering (Hukushima & Nemoto, J. Phys. Soc. Jpn. 65, 1604, 1996,
doi:10.1143/JPSJ.65.1604) helps but multiplies the cost by the number of replicas,
inside every gradient step.

**Mean field and TAP.** Naive mean field and its linear-response correction (Kappen &
Rodríguez, Neural Comput. 10, 1137, 1998, doi:10.1162/089976698300017386) and the TAP
correction (Thouless, Anderson & Palmer, Phil. Mag. 35, 593, 1977,
doi:10.1080/14786437708235992) are fast but are expansions about weak coupling. Letter
couplings are not weak: the RMS of the double-centred exact-MAP couplings is 1.17–1.87
nats in the five L = 4–5 validation fits, 1.69–1.73 at N = 142,614
(`validation[*].exact_reference.couplings.rms_b`).

**Adaptive cluster expansion.** ACE (Cocco & Monasson, Phys. Rev. Lett. 106, 090601,
2011, https://arxiv.org/abs/1102.3260 — the selective expansion of the entropy in
clusters of strongly interacting variables) is *not* a weak-coupling method: it keeps
clusters whose entropy contribution exceeds a threshold, is exact in the all-cluster
limit, and its authors report successful inference in strongly coupled regimes,
including at criticality and in the low-temperature phase. It is also deterministic,
so it shares PLM's immunity to mixing failures. Its costs and risks are the
truncation threshold, the convergence of the expansion, and cluster size: each kept
cluster of k positions needs its own inference over 26^k states, and in words of ≤ 20
letters with strong long-range couplings the important clusters may approach the whole
word. ACE was **not benchmarked here**; it is a credible alternative to PLM that should
be compared on the L = 4, 5 references.

**Log Z.** Absolute energies E = −log₂ P, entropies and held-out code lengths need
ln Z. Options: annealed importance sampling (Neal, Stat. Comput. 11, 125, 2001,
doi:10.1023/A:1008923215028 — the standard unbiased estimator of normalising constants
by annealing from a tractable distribution), thermodynamic integration / path sampling
(Gelman & Meng, Stat. Sci. 13, 163, 1998, https://statistics.fas.harvard.edu/resource/xlmstatsci13-2163-1851998pdf — the review that
unified importance, bridge and path sampling), and bidirectional Monte Carlo (Grosse,
Ghahramani & Adams, arXiv:1511.02543, 2015), which sandwiches ln Z between a forward
AIS lower bound and a reverse AIS upper bound but needs exact samples from the target.
AIS gives an *unbiased* Z, hence a ln Z that is biased *low* in expectation (Jensen),
and its delta-method error bar is itself unreliable when importance weights are
heavy-tailed — exactly what happens when the annealing misses a well. Thermodynamic
integration has the same mixing problem along the path without AIS's unbiasedness.

**Recommendation and what was built.** PLM for the couplings: deterministic (no mixing
failure), closed-form gradients at a cost linear in the number of word types, the
standard estimator in inverse Potts problems, and its deviations from exact ML are
measurable against exact fits at L ≤ 5 (§4.1). ACE is the main deterministic
alternative not tried (above). For ln Z, two tiers:
(a) **exact** for L ≤ 8 by *split enumeration* — the last five positions are summed
densely once (26^5 tensor) and the 26^(L−5) outer configurations are contracted against
it in batches; this is enumeration in a different summation order, exact to rounding,
and the same pass yields ⟨ln w⟩ and hence the model entropy exactly; (b) **AIS** for
L ≥ 9 from the independent-site model along the geometric path, 4 runs × 500 chains ×
4000 temperatures with one systematic Gibbs sweep per temperature, error bar = max of
the pooled delta-method SE and the between-run SE, checked against the exact value at
L = 4–8 (§4.2) and against schedule length and a hard lower-bound certificate at
L = 10, 14, 18 (§4.3). The schedule was fixed by a pilot on Gutenberg PLM fits at
L = 5–8 (1000 chains, σ² = 10, not part of the study output): with 100 temperatures
the error reached 0.59 nats (ESS 17–218 of 1000); with 4000 it was ≤ 0.11 nats and
within ~2 SE at every L, so 4000 was adopted.

Prior width σ² (Gaussian prior V ~ N(0, σ²) on every cell, λ = 1/(2σ²N) as in the
frozen study) is chosen on the frozen study's half-decade grid, extended until the
optimum is interior, by held-out pseudo-likelihood (always computable); at L ≤ 6 the
same choice is also made by exact held-out likelihood to test the proxy. Two phases
(§3): selection fits on 3/5 of the tokens scored on 1/5, then a final refit on 4/5.
Stopping rule: every component of
the PLM gradient below 0.01 (final fit) or 0.1 (selection fits) of the sampling SE of
the corresponding pair frequency, the analogue of the frozen fit's certificate.
Validation of the implementation (tests): finite-difference gradient; recovery of known
couplings from 2·10^6 exact samples (consistency); split enumeration vs an independent
brute-force L = 6 sum (relative error < 10⁻⁹); AIS within 5 SE of exact; per-word
minima/descent identical to `maxent.local_minima` / `maxent.descend` on the dense grid,
including descent from forbidden strings of a masked model (a forbidden start moves
only to a neighbour of finite weight, judged by the neighbour's full log-weight; a
start whose whole neighbourhood is forbidden stays put).

## 3. Data and protocol

Data: `data/derived/twitter_year_L1-20_all.pkl.gz` (all 41 shards of
`enryu43/twitter100m_tweets`, cleaned and English-filtered by `langfit.social`; tokens
are pure-ASCII-letter runs of length 1–20), all years 2006–2023 pooled. Each length is
subsampled (multivariate hypergeometric) to the frozen study's common
N = 142,614 tokens, so finite-sample biases are comparable across lengths; L ≥ 16 has
fewer tokens and uses all of them. Two phases, both on the same token split:
(1) *selection*: σ² chosen by fits on 3/5 of the tokens scored on a validation 1/5;
(2) *final refit* on the 4/5 that excludes only the test fifth (training + validation
tokens). Every reported model, S_1, S_2 quantity, landscape, plug-in entropy and
"tokens per parameter" is computed on that final 4/5 refit (e.g. L = 16: 110,469 of
137,998 tokens); only the test fifth is held out. The tokens-per-parameter threshold of
§5 is computed the same way in the validation fits. Reported per length (`lengths.<L>`):

- S_1: entropy of the exact independent-site model on the training tokens (closed form);
- training code length `S2_train_code_bits`. Every pairwise Gibbs model has *exact*
  training cross-entropy ≥ the sample maximum-entropy S_2 (ML/max-ent duality). For
  L ≤ 8 ln Z is exact, so this is a rigorous upper bound on S&B's S_2 (at L = 4, 5 it
  exceeds the exact S_2 by 0.06–0.18 bits at N = 142,614, §4.1). For L ≥ 9 it uses AIS
  ln Ẑ, which is biased low in expectation and so *lowers* the code length: there it is
  an **estimate** of the fitted model's code length with a numerical SE
  (`S2_train_code_numerical_se_bits`), not a bound (`S2_train_code_is_rigorous_upper_bound`);
- S_model: entropy of the fitted model (exact for L ≤ 8; AIS estimate beyond);
- captured = (S_1 − training code length)/(S_1 − S_plug-in), S&B's multi-information
  fraction: a rigorous lower bound on the in-sample fraction for L ≤ 8, an estimate with
  numerical SE for L ≥ 9 (`captured_is_rigorous_lower_bound`, `captured_numerical_se`);
- held-out gain: mean over test tokens of log₂ P_2 − log₂ P_1 (MAP independent model
  with its own CV-chosen prior), SE from the token variance combined in quadrature
  with the ln Z error;
- landscape: strict local minima (every single-letter substitution raises E) reached
  by steepest descent from every training type (`maxent.descend`'s rule and
  tie-break), split into seen/unseen strings; at L ≤ 5 compared with exhaustive search.

## 4. Validation: what was compared with exact computations

### 4.1 PLM vs exact maximum likelihood (L = 4, 5)

Same training tokens, same Gaussian prior (the PLM-selected σ²); exact MAP and exact
S&B fits by `langfit.potts`. Couplings compared after double-centring each pair tensor
(removes the single-site gauge freedom). The reference distribution for coupling
agreement, KL, entropy, test bits and minima is the **exact MAP fit at the same finite
prior**, not the unregularised maximum-entropy model; only the "S_2 exact" column is
the S&B maximum-entropy fit. Sampled regimes: L = 4 at N = 10⁴, 142,614, 10⁶; L = 5 at
N = 10⁴, 142,614 (no L = 5 fit at 10⁶; nothing at L ≥ 6). JSON:
`validation[*].exact_reference`.

| L | N | tokens/param | Pearson | slope | rel. L2 | S exact MAP | S PLM | KL(exact‖PLM) bits | test bits exact / PLM | S_2 exact / bound | minima exact / PLM / shared (Jaccard) | exact-minima mass shared | descent recall (count / mass) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 10⁴ | 1.97 | 0.982 | 1.17 | 0.28 | 9.785 | 9.735 | 0.056 | 9.702 / 9.749 | 9.262 / 9.404 | 103 / 120 / 92 (0.70) | 0.973 | 0.93 / 0.999 |
| 4 | 142,614 | 28.1 | 0.980 | 1.05 | 0.22 | 9.492 | 9.564 | 0.051 | 9.526 / 9.574 | 9.430 / 9.486 | 119 / 129 / 100 (0.68) | 0.991 | 0.96 / 0.999 |
| 4 | 10⁶ | 197 | 0.981 | 0.99 | 0.20 | 9.485 | 9.585 | 0.051 | 9.479 / 9.529 | 9.475 / 9.526 | 131 / 120 / 104 (0.71) | 0.990 | 0.99 / 1.000 |
| 5 | 10⁴ | 1.17 | 0.976 | 1.16 | 0.30 | 12.680 | 12.699 | 0.173 | 12.575 / 12.738 | 11.835 / 12.163 | 291 / 351 / 208 (0.48) | 0.938 | 0.78 / 0.980 |
| 5 | 142,614 | 16.9 | 0.973 | 1.05 | 0.25 | 12.383 | 12.483 | 0.166 | 12.318 / 12.484 | 12.270 / 12.447 | 346 / 375 / 236 (0.49) | 0.961 | 0.86 / 0.998 |

Reading: (a) couplings agree to Pearson 0.97–0.98. At L = 4 neither this agreement nor
KL(exact MAP‖PLM) (0.056, 0.051, 0.051 bits at N = 10⁴, 1.4·10⁵, 10⁶) improves with N;
at L = 5 the KL is 0.173 and 0.166 bits at N = 10⁴ and 1.4·10⁵. The flat L = 4 curve is
*consistent with* the misspecification gap of §2 (PLM and ML project the non-pairwise
truth differently) rather than sampling noise, but three N values do not prove an
asymptotic gap, and its size at L ≥ 6 is unmeasured. (b) The PLM
inflation of couplings (slope 1.17) is a small-N effect and is gone by 10⁶ tokens at L = 4.
(c) The price of PLM in held-out code length is 0.05 bits/token at L = 4 and
0.16–0.17 at L = 5, against held-out gains over the independent model of 5.4 and 7.0
bits (§5): 1–2.5 % of the effect being measured. (d) Minima: only half to two thirds of
the strict minima are common to both fits, but the shared ones carry 94–99 % of the
exact model's minimum mass — the disagreements are shallow minima. Statements about
individual shallow minima are therefore not robust to the estimator even at L = 4.
(e) Descent from the data words finds 78–99 % of all strict minima, holding ≥ 98 % of
their mass; the missed ones are shallow wells no data word drains into.

### 4.2 AIS vs exact ln Z (L = 4 … 8)

PLM fits at N = 142,614 on fresh subsamples; 4 runs × 500 chains × 4000 temperatures.
JSON: `ais_check`.

| L | AIS − exact ln Z (nats) | SE | error / SE | ESS per run (of 500) | S exact | S AIS ± SE |
|---|---|---|---|---|---|---|
| 4 | +0.0024 | 0.0014 | 1.7 | 498 | 9.611 | 9.732 ± 0.081 |
| 5 | +0.0007 | 0.0033 | 0.2 | 497 | 12.460 | 12.167 ± 0.087 |
| 6 | −0.0003 | 0.0021 | −0.1 | 497 | 15.394 | 15.399 ± 0.091 |
| 7 | −0.0027 | 0.0040 | −0.7 | 494 | 16.698 | 17.067 ± 0.106 |
| 8 | −0.0076 | 0.0051 | −1.5 | 486 | 17.802 | 17.836 ± 0.132 |

ln Z is accurate to < 0.01 nats (0.011 bits) with honest error bars. The **entropy**
from AIS (⟨ln w⟩ from the weighted final samples) is much worse: observed errors up to
0.37 bits in these five checks, −3.4 SE at L = 5 and +3.5 SE at L = 7, so the
delta-method entropy SE is not calibrated even here. This is a short-length
validation error, not an error scale for L ≥ 9, where the models are far more
concentrated (`S_model` 0.2–18 bits) and per-run ESS is lower; there the AIS entropy
uncertainty is **uncalibrated**. §5.1 reports, for every AIS-only length, the spread of
four independent per-run entropy estimates (`entropy_replicates`) as a replicated
numerical diagnostic, which bounds neither bias nor missed wells.

### 4.3 AIS beyond exact enumeration (L = 10, 14, 18)

No exact value exists; two weaker checks on the length-study models (JSON:
`ais_schedule`). ln Ẑ against annealing length (one run of 500 chains each), and the
certificate ln Z ≥ ln Σ w over training types and all descent terminals:

| L | T = 1000 | T = 4000 | T = 16000 | certificate |
|---|---|---|---|---|
| 10 | 80.485 ± 0.063 | 80.474 ± 0.029 | 80.513 ± 0.015 | 79.645 |
| 14 | 80.654 ± 0.049 | 80.719 ± 0.028 | 80.692 ± 0.014 | 80.598 |
| 18 | 92.884 ± 0.047 | 92.933 ± 0.023 | 92.918 ± 0.012 | 92.803 |

No drift beyond ~2 SE as the schedule grows 16-fold, and the certificate is respected.
This rules out gross failures of the annealing *along the path it explores*; it cannot
rule out wells that neither the annealing nor the data visit (§6).

## 5. Results across lengths

### 5.1 Table (JSON: `lengths`, `summary`, `entropy_replicates`)

Bits. N = tokens used; train tokens = the final 4/5 refit; tokens/param = train tokens
per coupling parameter (L(L−1)/2 · 676). "train code" is the fitted model's training
code length: a rigorous upper bound on S_2 for L ≤ 8 (exact ln Z), an **estimate**
(± numerical SE from AIS ln Ẑ, biased low in expectation) for L ≥ 9. "captured" is the
multi-information fraction from it: a rigorous lower bound for L ≤ 8 (≥), an estimate
(est.) for L ≥ 9. S_model: entropy of the fitted model, exact for L ≤ 8; for L ≥ 9 the
length job's pooled AIS estimate, followed by the range of four independent per-run AIS
entropy estimates on a refit of the same model and their per-run ESS (of 500 chains) —
an **uncalibrated** numerical diagnostic (§4.2). D = train code − S_model (duality gap).
Top-well: log₂ of model probability over pooled empirical frequency for the deepest
strict minimum found. Screens: §5.4 (heuristic).

| L | N used | train tokens | train types | tokens/param | σ² | S_1 | train code | S_model | S_model per-run AIS (ESS) | S_plug-in | captured | test bits ind. | test bits pair | held-out gain | minima found (unseen) | tokens on minima | model mass on train types | D | top-well log₂(P/f) | local screen | all-quantity screen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 142,614 | 114,234 | 513 | 168.99 | 1 | 6.38 | 4.57 | 4.60 | exact | 4.56 | ≥ 1.00 | 6.37 | 4.55 | 1.82 ± 0.01 | 9 (0) | 0.47 | 1.00 | -0.03 | +0.00 (to) | pass | pass |
| 3 | 142,614 | 114,023 | 2,536 | 56.22 | 1 | 10.33 | 6.09 | 6.08 | exact | 5.71 | ≥ 0.92 | 10.32 | 6.14 | 4.18 ± 0.02 | 39 (0) | 0.70 | 0.92 | +0.02 | -0.01 (the) | pass | pass |
| 4 | 142,614 | 113,864 | 4,074 | 28.07 | 3.2 | 14.99 | 9.51 | 9.58 | exact | 8.15 | ≥ 0.80 | 14.99 | 9.56 | 5.43 ± 0.02 | 116 (22) | 0.55 | 0.69 | -0.07 | -0.29 (this) | pass | pass |
| 5 | 142,614 | 114,094 | 6,738 | 16.88 | 3.2 | 19.46 | 12.42 | 12.41 | exact | 9.51 | ≥ 0.71 | 19.46 | 12.51 | 6.95 ± 0.03 | 327 (180) | 0.42 | 0.39 | +0.01 | +0.38 (about) | pass | pass |
| 6 | 142,614 | 113,916 | 9,510 | 11.23 | 3.2 | 23.34 | 15.05 | 15.37 | exact | 10.36 | ≥ 0.64 | 23.34 | 15.18 | 8.17 ± 0.03 | 784 (587) | 0.40 | 0.18 | -0.33 | +0.01 (thanks) | pass | pass |
| 7 | 142,614 | 114,084 | 10,154 | 8.04 | 3.2 | 26.95 | 16.87 | 16.76 | exact | 10.61 | ≥ 0.62 | 26.97 | 17.00 | 9.98 ± 0.04 | 1,719 (1,447) | 0.41 | 0.12 | +0.11 | +0.91 (tonight) | pass | pass |
| 8 | 142,614 | 114,219 | 10,979 | 6.03 | 3.2 | 31.17 | 18.37 | 18.00 | exact | 10.73 | ≥ 0.63 | 31.16 | 18.57 | 12.59 ± 0.05 | 3,202 (2,818) | 0.48 | 0.10 | +0.37 | +1.19 (everyone) | pass | pass |
| 9 | 142,614 | 114,476 | 11,055 | 4.70 | 1 | 34.45 | 19.08 ± 0.01 (est.) | 17.78 | 17.79–18.49 (465–472) | 10.56 | 0.644 ± 0.001 (est.) | 34.45 | 19.34 | 15.12 ± 0.05 | 4,342 (3,959) | 0.51 | 0.15 | +1.30 | +2.41 (beautiful) | pass | fail |
| 10 | 142,614 | 113,918 | 10,445 | 3.74 | 3.2 | 38.03 | 19.92 ± 0.02 (est.) | 12.02 | 11.76–12.59 (341–349) | 10.40 | 0.655 ± 0.001 (est.) | 38.01 | 20.19 | 17.81 ± 0.06 | 6,187 (5,661) | 0.61 | 0.38 | +7.91 | +4.15 (experience) | pass | fail |
| 11 | 142,614 | 114,173 | 9,557 | 3.07 | 1 | 40.58 | 19.23 ± 0.02 (est.) | 14.86 | 14.20–15.20 (406–416) | 10.00 | 0.698 ± 0.001 (est.) | 40.44 | 19.53 | 20.91 ± 0.06 | 6,417 (5,845) | 0.69 | 0.31 | +4.37 | +3.35 (development) | pass | fail |
| 12 | 142,614 | 114,022 | 8,952 | 2.56 | 1 | 44.59 | 18.85 ± 0.04 (est.) | 12.30 | 11.64–13.13 (190–242) | 9.68 | 0.737 ± 0.001 (est.) | 44.58 | 19.32 | 25.26 ± 0.08 | 6,684 (6,089) | 0.76 | 0.44 | +6.55 | +3.93 (relationship) | pass | fail |
| 13 | 142,614 | 113,920 | 7,965 | 2.16 | 1 | 46.20 | 16.29 ± 0.01 (est.) | 9.06 | 8.75–9.60 (426–443) | 8.66 | 0.797 ± 0.000 (est.) | 46.17 | 16.64 | 29.53 ± 0.07 | 6,208 (5,662) | 0.83 | 0.66 | +7.23 | +1.49 (international) | pass | fail |
| 14 | 142,614 | 113,843 | 10,114 | 1.85 | 0.32 | 48.62 | 20.00 ± 0.02 (est.) | 2.71 | 2.24–2.73 (345–362) | 8.33 | 0.710 ± 0.000 (est.) | 48.63 | 20.38 | 28.24 ± 0.08 | 6,855 (6,423) | 0.81 | 0.92 | +17.29 | +3.63 (administration) | pass | fail |
| 15 | 142,614 | 114,130 | 9,661 | 1.61 | 0.32 | 38.45 | 19.57 ± 0.00 (est.) | 0.19 | 0.22–0.31 (493–494) | 5.59 | 0.575 ± 0.000 (est.) | 38.52 | 20.11 | 18.41 ± 0.08 | 6,479 (6,111) | 0.85 | 0.99 | +19.38 | +0.99 (congratulations) | pass | fail |
| 16 | 137,998 | 110,469 | 17,470 | 1.36 | 0.32 | 59.54 | 29.59 ± 0.05 (est.) | 1.17 | 0.75–1.69 (145–162) | 9.01 | 0.593 ± 0.001 (est.) | 59.60 | 30.31 | 29.29 ± 0.12 | 11,204 (10,858) | 0.70 | 0.90 | +28.42 | +3.59 (entrepreneurship) | pass | fail |
| 17 | 55,229 | 44,145 | 12,406 | 0.48 | 0.32 | 65.63 | 34.32 ± 0.03 (est.) | 5.17 | 4.55–5.31 (306–326) | 10.29 | 0.566 ± 0.000 (est.) | 65.72 | 36.06 | 29.67 ± 0.20 | 8,613 (8,340) | 0.55 | 0.75 | +29.14 | +3.51 (multidisciplinary) | fail | fail |
| 18 | 31,185 | 24,904 | 9,069 | 0.24 | 0.32 | 70.01 | 43.26 ± 0.02 (est.) | 1.39 | 1.18–1.43 (392–396) | 10.14 | 0.447 ± 0.000 (est.) | 69.95 | 46.58 | 23.36 ± 0.29 | 6,870 (6,633) | 0.49 | 0.89 | +41.88 | +2.67 (disproportionately) | fail | fail |
| 19 | 16,699 | 13,291 | 6,465 | 0.11 | 0.32 | 77.89 | 45.56 ± 0.12 (est.) | 3.13 | 2.15–4.10 (16–27) | 11.07 | 0.484 ± 0.002 (est.) | 77.68 | 51.85 | 25.83 ± 0.45 | 5,199 (4,964) | 0.37 | 0.82 | +42.43 | +4.12 (beavercreekchurchva) | fail | fail |
| 20 | 11,551 | 9,324 | 4,905 | 0.07 | 0.32 | 82.07 | 44.22 ± 0.05 (est.) | 12.26 | 11.63–12.89 (238–262) | 11.05 | 0.533 ± 0.001 (est.) | 81.88 | 52.25 | 29.63 ± 0.56 | 3,896 (3,626) | 0.37 | 0.52 | +31.96 | +3.59 (internationalization) | fail | fail |

At L ≤ 5 exhaustive search finds 126 and 393 strict minima; descent from data recovers
92 % and 83 % of them, carrying > 99.6 % of their mass. The per-run AIS entropies spread
by 0.1–2.0 bits at L ≥ 9 (largest at L = 19, where per-run ESS is only 16–27 of 500); at
L = 10 two further runs inside the prior ablation (§5.3, same σ², fit warm-started from
σ² = 1) gave 12.84 and 13.45, so six runs on (to stopping tolerance) the same model span
11.76–13.45 bits. Entropy-derived quantities at L ≥ 9 (S_model, D) carry uncertainties of
this order or larger, not the ± in the train-code column, which is the ln Ẑ error only.

### 5.2 Observations

1. **L = 2–8 (87.9 % of all 1–20-letter tokens).** ln Z, model entropy and code lengths
   of the fitted model are exact; the S_2 bound and captured fraction are rigorous.
   Duality gap |D| ≤ 0.37 bits, deepest-well calibration within a factor 1.3 at L ≤ 6
   and ~2 at L = 7–8. The pairwise share of in-sample multi-information falls from ≈ 1
   (L = 2, where the pair *is* the word) through ≥ 0.92 (L = 3) and ≥ 0.80 (L = 4; the
   frozen Austen study had 0.92 with exact S_2) to ≥ 0.62 at L = 6–8. The held-out gain
   over independent letters grows from 1.8 to 12.6 bits/token, ≈ 1.5 bits per letter at
   L = 8. *What is validated against exact maximum likelihood is only L = 4, 5 (§4.1)*;
   at L = 6–8 the fitted model is computed exactly but its fidelity to the ML fit is
   not measured.
2. **L = 9–16 (7.6 % of tokens), observed miscalibration.** Held-out gains remain large
   and significant and AIS ln Ẑ is stable (§4.3). But the duality gap is 1.3 bits at
   L = 9 and 4–28 bits beyond (with entropy uncertainties of up to ~2 bits, §5.1), the
   model entropy falls below the plug-in entropy of the sample, and the deepest strict
   minimum is over-weighted relative to its pooled frequency by log₂ ratios of +1.0 to
   +4.2: "experience" (L = 10, σ² = 3.2) gets P = 0.31 against an empirical 0.017,
   "relationship" (L = 12, σ² = 1) +3.9, "administration" (L = 14, σ² = 0.32) +3.6,
   "congratulations" (L = 15, σ² = 0.32; 50 % of all 15-letter tokens) 0.99 against 0.50.
   At the same time the 25 most frequent training types are on average *under*-weighted
   (mean log₂ ratio −2.9 to −6.9 bits in the ablation fits at L = 10, 14): the mass is
   concentrated on a few wells, not spread over the frequent words. The independent
   baseline is very poor at these lengths (S_1 ≈ 35–60 bits), so a large held-out gain
   certifies letter-level structure, not a calibrated distribution over words.
3. **L ≥ 17 (0.014 % of tokens).** Fewer than 0.5 training tokens per parameter, and the
   vocabulary is increasingly concatenated handles, brand names and keyboard runs that
   survived cleaning (`beavercreekchurchva`, `hahahahahahahahahaha`,
   `jjjjjjjjjjjjjjjjjj` among the unseen minima).

### 5.3 Is the miscalibration caused by the prior? (ablation, JSON: `prior_ablation`)

A candidate explanation is that at long L the relative depths of isolated wells are
weakly identified by the conditionals, so the prior width decides them. This is a
hypothesis, not an established property: the PLM objective is weighted by the observed
words and, with full support, compatible conditionals determine the joint. To test the
prior part of it, the L = 10 and L = 14 models were refitted on the same tokens and
split at σ² = 0.32, 1, 3.2, 10, 31.6 (2 AIS runs each):

| L | σ² | test bits | held-out PLL | train code | S (AIS, 2 runs) | D | deepest training minimum, log₂(P/f) | top-25 types mean log₂(P/f) | ESS |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.32 | **19.41** | −8.10 | 19.26 | 16.71 / 16.91 | +2.44 | experience +3.15 | −2.88 | 449 / 441 |
| 10 | 1 | 19.77 | **−7.98** | 19.56 | 13.93 / 14.04 | +5.57 | experience +3.77 | −3.17 | 405 / 394 |
| 10 | 3.2 (selected) | 20.20 | −7.98 | 19.92 | 12.84 / 13.45 | +6.78 | experience +4.15 | −3.51 | 354 / 342 |
| 10 | 10 | 20.84 | −8.04 | 20.51 | 11.30 / 11.85 | +8.93 | experience +4.06 | −4.08 | 289 / 272 |
| 10 | 31.6 | 23.11 | −8.12 | 22.73 | 8.04 / 9.49 | +14.15 | experience +2.08 | −6.30 | 7 / 92 |
| 14 | 0.32 (selected) | **20.45** | **−6.44** | 20.06 | 2.87 / 2.72 | +17.27 | administration +3.56 | −3.41 | 361 / 353 |
| 14 | 1 | 21.56 | −6.48 | 21.04 | 2.20 / 1.98 | +18.94 | administration +3.66 | −4.45 | 330 / 333 |
| 14 | 3.2 | 22.52 | −6.63 | 21.87 | 1.87 / 1.79 | +20.04 | administration +3.74 | −5.31 | 324 / 316 |
| 14 | 10 | 23.48 | −6.82 | 22.71 | 1.71 / 1.79 | +20.96 | administration +3.76 | −6.17 | 312 / 320 |
| 14 | 31.6 | 24.33 | −7.01 | 23.45 | 1.52 / 1.49 | +21.95 | administration +3.75 | −6.94 | 310 / 319 |

(PLL held-out values are at L = 10 within 0.005 bits between σ² = 1 and 3.2; the
selection used the validation fifth, this table the test fifth. At L = 10, σ² = 31.6
the AIS ESS of 7 marks that run as unreliable.)

Findings: (a) the over-weighting of the deepest well is present at **every** prior
width tried (+2.1 to +4.2 at L = 10, +3.6 to +3.8 at L = 14), and the duality gap stays
≥ 2.4 bits even at the smallest σ². Within 0.32–31.6, the prior width is therefore not
what produces the miscalibration; the "small σ² lets the deepest well absorb the mass"
explanation given in the first version of this document is withdrawn. (b) The prior
width selected by held-out *pseudo*-likelihood is not the held-out-*likelihood*
optimum at L = 10: σ² = 0.32 codes the test tokens 0.78 bits/token better than the
selected 3.2, and at both lengths the best held-out likelihood sits at the lower edge of
this grid, so the likelihood optimum may be lower still. Remaining candidate causes,
not separated here: the PLM-vs-ML difference of population objectives (§2), amplified
at long L; weak identification of well depths that no prior width repairs. Evidence
that would separate them: a likelihood-based refinement (e.g. PCD/Boltzmann learning
started from the PLM solution, or ML at L = 9 by split enumeration of marginals) on the
same data, checking whether the calibration of the deepest wells and D return to the
L ≤ 8 range; and σ² selection by held-out likelihood with AIS ln Ẑ on a grid extended
below 0.32.

### 5.4 Heuristic screens and coverage

The summary (`summary`) gives two **heuristic screens**, not validated limits.
Thresholds come from the exact comparisons at L = 4, 5 and are transferred to other
lengths; they say "no warning sign of these kinds", not "this observable has been
validated". Each screen is the contiguous prefix of passing lengths starting at L = 2
(a failing or missing L = 2 gives no cutoff).

- *Local screen*: (a) held-out gain > 2 SE (token noise and ln Ẑ error), (b) converged
  fit, (c) training tokens per parameter (4/5 refit) ≥ the smallest ratio in a
  PLM-vs-exact comparison (1.17, L = 5 at N = 10⁴), (d) the certificate ln Z ≥ ln Σ_train
  w. Passes for L = 2–16 (these lengths are 95.5 % of all 1–20-letter tokens; 99.99 %
  counting single letters).
- *All-quantity screen*: adds |D| ≤ the largest |D| in the compared fits (0.54 bits).
  D = θ·(μ_model − μ_emp)/ln 2 for any Gibbs model: one signed contraction that is small
  at the ML/MAP optimum. It is not a bound on marginal errors, KL, minima or descent.
  Passes for L = 2–8 (87.9 % of all 1–20-letter tokens; 92.4 % counting single letters).

Status of quantities by length:

| lengths | exactly computed for the fitted model | validated against exact ML | screened only |
|---|---|---|---|
| 2, 3 | ln Z, entropy, code lengths, S_2 bound, minima (dense grid) | no (not run) | yes |
| 4, 5 | as above | yes: couplings, KL, entropy, test bits, minima (§4.1) | yes |
| 6–8 | ln Z, entropy, code lengths, S_2 bound; minima found by descent only | no | yes |
| 9–20 | none (AIS ln Ẑ, AIS entropy; minima by descent) | no | 9–16 pass local screen only |

The pass at L = 9–16 of the local screen rests on the transferred tokens-per-parameter
threshold and the held-out gain; within-well structure (which substitutions raise E,
descent targets) at those lengths has **not** been compared with any exact reference.
Observable-specific validation at L ≥ 6 (e.g. exact ML at L = 6 on a reduced alphabet,
or comparison of descent targets with a likelihood-refined fit) is needed before
claiming it.

## 6. Limitations and low-confidence regions

- **Different estimand, not just sampling.** PLM and ML have different population
  optima unless the data are pairwise (§2). In the sampled regimes, KL(exact MAP ‖ PLM)
  is 0.056/0.051/0.051 bits at L = 4 (N = 10⁴, 1.4·10⁵, 10⁶) and 0.173/0.166 bits at L = 5
  (N = 10⁴, 1.4·10⁵), reference = the exact MAP fit at the same prior. Its size at L ≥ 6
  is not measured. S_2 at L ≥ 6 is reported only through the training code length.
- **Bounds only with exact normalisation.** The S_2 upper bound and captured lower bound
  are rigorous for L ≤ 8. For L ≥ 9 they use AIS ln Ẑ (biased low in expectation) and
  are estimates; no upper bound on ln Z was derived.
- **Well depths at L ≥ 9.** Word-frequency calibration, absolute energies, model
  entropy and anything summed over wells (minima mass, "unseen mass") are observed to be
  miscalibrated (§5.2); the cause is not established (§5.3).
- **ln Z beyond L = 8.** AIS agreed with exact ln Z to 0.01 nats up to L = 8 and is
  stable to 16× longer schedules at L = 10, 14, 18, but AIS ln Ẑ is biased low in
  expectation and cannot see wells that annealing from the independent model does
  not reach. The certificate bounds it from below only. A matching upper bound
  (reverse AIS / bidirectional MC, Grosse et al. 2015) would need exact samples, which
  we do not have.
- **Entropies from AIS** had observed errors up to 0.37 bits (3.5 SE) at L = 4–8; at
  L ≥ 9 their uncertainty is uncalibrated; per-run spreads reach 1–2 bits (§5.1).
- **Minima cannot be counted or certified for L ≥ 6.** They are found by descent from
  data words; exhaustive search at L = 4, 5 shows descent misses 8–17 % of minima
  (shallow, < 0.5 % of minimum mass), and minima that no data word drains into —
  "invented words" far from the vocabulary — are invisible by construction. Strictness
  of each reported minimum *is* certified exactly (all L·25 substitutions checked).
- **Shallow minima are estimator-dependent** even at L = 4 (Jaccard 0.5–0.7 between
  PLM and exact fits).
- **Rare letters and unseen pair cells.** Every cell gets a finite coupling through
  the prior; energies of strings with letter pairs never seen in training are set by
  σ², not data. The S&B "exact" mode (unseen cells = −∞) is supported by `fit_plm`
  (`mask`) and by `descend` (forbidden starts handled as in the dense reference), but
  AIS cannot use it (forbidden cells kill chains); it is not used in the study.
- **Prior selection proxy.** Held-out pseudo-likelihood and held-out exact likelihood
  choose σ² one half-decade apart at L = 2–6 (PLL: 1, 1, 3.2, 3.2, 3.2; LL: 3.2, 3.2,
  1, 1, 1), and at L = 10 the PLL choice codes test tokens 0.78 bits worse than the best
  σ² on the ablation grid (§5.3).
- **Composition.** Pooled 2006–2023 Twitter with the dataset's ~1,001-tweet per-user
  truncation; at L ≥ 13 single words dominate (`international` 6.6 %,
  `congratulations` 50 % of their lengths) and handles/brand concatenations leak
  through cleaning at L ≥ 18.
- **One subsample per length.** Sampling variability across subsamples was not
  measured (the frozen study used 5); held-out SEs cover test-token noise only.

## 7. Open questions for discussion

1. Is within-well structure (spelling robustness, descent, edit moves) what downstream
   questions need, or calibrated word probabilities? Either way L ≥ 9 needs work: the
   former has not been compared with any exact reference beyond L = 5, the latter fails
   (§5.2). The ablation (§5.3) rules out prior width as the fix; next candidates are a
   likelihood-based refinement from the PLM solution and σ² selection by held-out
   likelihood (AIS ln Ẑ per grid point) on a grid below 0.32.
2. Should ACE be benchmarked against the L = 4, 5 exact references as a deterministic
   alternative to PLM (§2)?
3. Should S_2 at L ≥ 6 be pursued exactly? Exact ML at L = 6 needs the 26^6 grid per
   step (≈ 2.5 GB per float64 tensor) — out of reach on this machine; model marginals
   by split enumeration at every Newton step are possible but slow.
4. Cross-length comparisons of energies (is a 9-letter word "deeper" than a 4-letter
   one?) need ln Z at both lengths and calibrated fits; at present only L ≤ 8 has exact
   ln Z, and only L = 4, 5 are validated against ML.
5. Larger N per length (up to 10⁶ tokens exist for L ≤ 11) would raise tokens per
   parameter but does not by itself address the estimand difference or the
   miscalibration; worth it together with (1).

## Citations (all verified by web search)

- Stephens & Bialek, Phys. Rev. E 81, 066119 (2010), https://link.aps.org/doi/10.1103/PhysRevE.81.066119; arXiv:0801.0253.
- Besag, The Statistician 24, 179 (1975), https://doi.org/10.2307/2987782.
- Hyvärinen, Neural Comput. 18, 2283 (2006), https://pubmed.ncbi.nlm.nih.gov/16907626.
- Ravikumar, Wainwright & Lafferty, Ann. Stat. 38, 1287 (2010), https://doi.org/10.1214/09-AOS691.
- Aurell & Ekeberg, Phys. Rev. Lett. 108, 090201 (2012), https://link.aps.org/doi/10.1103/PhysRevLett.108.090201.
- Ekeberg, Lövkvist, Lan, Weigt & Aurell, Phys. Rev. E 87, 012707 (2013), https://link.aps.org/doi/10.1103/PhysRevE.87.012707.
- Nguyen, Zecchina & Berg, Adv. Phys. 66, 197 (2017), https://www.tandfonline.com/doi/abs/10.1080/00018732.2017.1341604.
- Cocco, Feinauer, Figliuzzi, Monasson & Weigt, Rep. Prog. Phys. 81, 032601 (2018), https://doi.org/10.1088/1361-6633/aa9965.
- Ackley, Hinton & Sejnowski, Cognitive Science 9, 147 (1985), https://doi.org/10.1207/s15516709cog0901_7.
- Tieleman, ICML 2008, https://dl.acm.org/doi/10.1145/1390156.1390290.
- Hukushima & Nemoto, J. Phys. Soc. Jpn. 65, 1604 (1996), https://journals.jps.jp/doi/10.1143/JPSJ.65.1604.
- Kappen & Rodríguez, Neural Comput. 10, 1137 (1998), https://doi.org/10.1162/089976698300017386.
- Thouless, Anderson & Palmer, Phil. Mag. 35, 593 (1977), https://www.tandfonline.com/doi/abs/10.1080/14786437708235992.
- Cocco & Monasson, Phys. Rev. Lett. 106, 090601 (2011), https://arxiv.org/abs/1102.3260.
- Neal, Stat. Comput. 11, 125 (2001), https://link.springer.com/article/10.1023/A:1008923215028.
- Gelman & Meng, Stat. Sci. 13, 163 (1998), https://statistics.fas.harvard.edu/resource/xlmstatsci13-2163-1851998pdf.
- Grosse, Ghahramani & Adams, arXiv:1511.02543 (2015), https://arxiv.org/abs/1511.02543.
- Liu & Nocedal (L-BFGS, the optimiser used), Math. Program. 45, 503 (1989), https://link.springer.com/article/10.1007/BF01589116.
- Schneidman, Still, Berry & Bialek, Phys. Rev. Lett. 91, 238701 (2003), https://link.aps.org/doi/10.1103/PhysRevLett.91.238701 (the multi-information hierarchy behind "captured").
