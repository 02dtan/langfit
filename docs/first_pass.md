# First pass: extending the Stephens & Bialek four-letter model

This document assembles the first-pass study. The baseline is Stephens & Bialek's (S&B;
Phys. Rev. E 81, 066119, 2010; arXiv:0801.0253) pairwise maximum-entropy model of
four-letter words, reproduced in [`four_letter_reproduction.md`](../four_letter_reproduction.md)
(frozen: `langfit/maxent.py`, `scripts/maxent_study.py`, `data/derived/maxent_study.json`).
Each section below summarises one extension; the full derivations, tables and verified
references are in the linked fragment. Each fragment was independently reviewed
(reviews in [`docs/reviews/`](reviews/)) and the findings applied. The interactive
artifact [`index.html`](../index.html) shows
both the pure S&B model (tabs 1–2) and the new model (tabs 3–4); tab 0 is the theory.

## 1. Edit-move neighbourhoods — [`first_pass/neighbourhood.md`](first_pass/neighbourhood.md)

S&B define wells through single-letter *substitution*. Adding transposition, insertion
and deletion (a grand-canonical energy across lengths) lets the landscape see temporal
and length-changing typos. With the full move set a transposition such as `waht` → `what`
is recovered in its first downhill step; a noisy-channel decoder recovers 7 of 8 probe
typos; move-conditioned energy changes are uphill on average. Physical justification:
the error channel of typing/speech is not substitution-only (§4 of the fragment).
Code `langfit/edits.py`, `scripts/edit_study.py`, `tests/test_edits.py`.

## 2. Three-body interactions — [`first_pass/ternary.md`](first_pass/ternary.md)

Does the pairwise model miss real structure? On pooled Twitter (all 41 shards) the
multi-information is I_N = 6.763 bits = 5.514 (81.5 %) pairwise + 1.199 (17.7 %) triplet
+ 0.050 quadruplet. Held-out code length improves by 1.18 bits at ten times the
tokens-per-cell threshold; the S&B near-saturation S_3 ≈ S_full at N ≈ 1.4·10^5 is largely
a small-sample effect. Triplet terms change the landscape (133 → 1061 strict minima;
`tmrw` rank 377598 → 1226). Code `langfit/potts.py`, `scripts/ternary_study.py`,
`tests/test_potts.py`.

## 3. Words of every length — [`first_pass/length.md`](first_pass/length.md)

Exact enumeration fails beyond L ≈ 5. The new model fits the same pairwise family by
pseudo-likelihood with exact ln Z up to L = 8 (split enumeration) and annealed importance
sampling beyond. Validation against exact fits (L = 4, 5): coupling correlation
0.97–0.98, KL 0.05–0.17 bits; AIS ln Z within 0.008 nats of exact at L = 4–8.
Quantities are trustworthy for L = 2–8 (87.9 % of tokens); local structure to L = 16
(95.5 %); at L ≥ 9 the model strongly over-weights the deepest well (probability of
"experience" 0.31 vs 0.017 observed) and entropy estimates are uncalibrated. The
adequacy cutoffs are heuristic screens, not validation. Code `langfit/longwords.py`,
`scripts/length_study.py`, `tests/test_longwords.py`.

## 4. Projection basis — [`first_pass/projection_basis.md`](first_pass/projection_basis.md)

Tested PCA, TICA (raw and regularised), Fisher LDA and VAMP-2 against nonlinear slow
modes. Raw TICA degenerates on one-hot indicators (whitening blows up loadings);
regularised TICA and LDA are statistically indistinguishable from PCA; PCA is best on
basin separation (0.35), trustworthiness (0.886) and year-to-year stability (9.8°).
Eckart–Young: no linear basis explains more variance than the top-2 principal axes
(≈ 17–20 %). Recommendation: keep PCA for S&B; use a boundary-marked bigram-profile PCA
for the variable-length model. Code `langfit/basis.py`, `scripts/basis_study.py`,
`tests/test_basis.py`.

## 5. Landscape of the new model — [`first_pass/new_model_landscape.md`](first_pass/new_model_landscape.md)

All lengths 1–8, grand-canonical energies, edit moves, 41 shards; exact (Potts MAP) for
L ≤ 5, PLM with exact split-enumeration ln Z for L = 6–8; σ² chosen per length by CV.
Height z is relative to the empty-data prior (not comparable to the S&B E_bin).
Final model S₂ = 12.6 bits, 10,171 string-level minima (deepest: the, to, of, and, in);
19 cumulative yearly frames 2006–2023. At 32×32 bins the projected surface has 12 wells,
8 of them confident (more than half of the basin's model mass in confidently estimated
cells); only wells holding at least 0.1% of the model's mass are shown by default. A well
is named after the most probable strings of its whole basin, ranked by P(w) = P(L)·P_L(w);
the minimum cell's own top string is stored separately. At 32×32 bins the 8 wells above
the mass floor are named `their`, `in`, `the`, `to`, `be`, `and`, `of`, `is`, and their
minimum-cell strings are `thether`, `inining`, `there`, `thinking`, `mathare`, `revered`,
`another`, `this`. For example, the basin named `in` (6.9% of the model's mass) has
minimum-cell string `inining`, whose own probability is 1.6e-7 against 1.7e-2 for `in`,
so the earlier non-word labels reflected the naming rule, not a model preference for
non-words. Approximate (sampled, L ≥ 6) strings are marked `*`. The length 6–8 part of
the surface remains exploratory: the importance sampler's effective sample sizes are
≈ 1139 / 76 / 10 at L = 6 / 7 / 8, cells dominated by L = 7–8 are flagged low-confidence
(372 cells; 455 are true gaps), and their standard errors are lower bounds. Heights z
are a mass ratio to a per-length-uniform prior, so they are not comparable across
lengths and sparse cells give extreme values. At fine resolution most wells are
negligible: at 96×96 bins 61 of 346 wells hold at least 0.1% of the mass. Two earlier
bugs are fixed: wells were named from the lowest-indexed basin cells and later from the
minimum cell's top string (neither follows probability), and the move graph was
symmetrised wrongly (322 spurious components). Code `langfit/nm_landscape.py`,
`scripts/build_nm_landscape.py`, `tests/test_nm_landscape.py`.

## Caveats carried across sections

- Smoothing in the artifact (a Gaussian filter in the Boltzmann/probability domain, live in the browser) is purely visual; grey = no data.
- E_bin can exceed 18.80 bits: 18.80 = log₂26⁴ is the maximum entropy (the P-weighted mean
  of the energies), not a bound on any single bin energy.
- Time evolution of the new model is yearly (monthly warm-start frames cost ~6 h).
