# Review of the variable-length model (langfit/longwords.py, scripts/length_study.py, docs/first_pass/length.md)

## [P1] Restrict rigorous entropy-bound labels to exact normalisation

`/Users/dtan/projects/languagefitness/docs/first_pass/length.md:144-150`

Should-fix: the statement “this is a rigorous upper bound on S&B's S_2” is valid for the true training cross-entropy, but `analyse_length` substitutes AIS ln Ẑ into `S2_upper` for every L≥9 and derives `captured_lower` from it. AIS estimates of ln Z are lower-biased in expectation and are not upper bounds, so an underestimate directly lowers the alleged S₂ upper bound and raises the alleged captured-information lower bound; the training-type certificate supplies a lower bound on Z and cannot restore either guarantee. Restrict the inequality labels to L≤8 and report the longer-length quantities as estimates with numerical uncertainty, or obtain a valid upper bound on ln Z before claiming rigorous inequalities.

## [P1] Label the adequacy cutoffs as heuristic extrapolations

`/Users/dtan/projects/languagefitness/docs/first_pass/length.md:286-293`

Blocker for the scientific verdict: “L_max = 8 for every quantity” and “L_max,local = 16 for conditional/landscape statements” are not validated by these criteria. `summarise` transfers the smallest raw tokens/parameter ratio (1.170710, from L=5,N=10000) and largest scalar duality gap (0.535704 bits, from that same fit) to different-length models, although exact PLM–MAP comparisons stop at L=5 and already give minima Jaccard 0.479–0.707. The identity D=θ·(μ_model−μ_emp)/ln2 is only one signed contraction, not a bound on marginal errors, KL, minima, or descent; exact ln Z validates normalisation of the fitted model, not its fidelity to ML or language. Preserve the computed coverage numbers but label these boundaries as heuristic screens, distinguish exactly computable quantities from empirically validated ones, and require observable-specific validation before claiming all quantities or local structure are trustworthy at the extrapolated lengths.

## [P2] Do not extrapolate a fixed 0.4-bit AIS entropy uncertainty

`/Users/dtan/projects/languagefitness/docs/first_pass/length.md:201-204`

Should-fix: “AIS entropies beyond L = 8 should be read with an uncertainty of ~0.4 bits” turns the largest error in five enumerable-model checks into an uncertainty calibration for different, much more concentrated models. The JSON confirms errors −0.293607/0.087242 = −3.37 SE at L=5 and +0.368209/0.105793 = +3.48 SE at L=7, but supplies no coverage experiment supporting a universal replacement SE; at L=19 the reported chain ESS is only 33–40/500, versus 485–498/500 in those checks. Stable ln Z does not establish convergence of the endpoint energy moment used for entropy, and missed wells need not cause errors bounded by 0.4 bits. Report 0.37 bits as the observed short-length validation error, mark longer-length entropy uncertainty as uncalibrated, and retain per-run energy/entropy diagnostics and covariance-aware or replicated uncertainty estimates rather than assigning the same error scale to every length.

## [P2] Separate observed well miscalibration from its proposed cause

`/Users/dtan/projects/languagefitness/docs/first_pass/length.md:272-277`

Should-fix: the passage attributes the overweighting to relative depths being determined “only through the prior” and says “the small σ² chosen there (0.32) lets the deepest well absorb the mass.” The observed effect is real, but its flagship example has `lengths.10.sigma2 = 3.1622776601683795`, not 0.32, and the stored experiments contain no same-data prior-sensitivity or likelihood-refinement comparison isolating that mechanism. Moreover the shared-coupling PLM objective is weighted by observed word counts, so relative depths are not generally functions of the prior alone; full-support compatible conditionals determine the joint distribution. Describe weak identification under sparse observed contexts as a hypothesis, give each example's actual σ², and distinguish that hypothesis from the measured calibration failure unless an ablation establishes the causal attribution.

## [P2] Limit the asymptotic-gap claim to the sampled validation regimes

`/Users/dtan/projects/languagefitness/docs/first_pass/length.md:297-300`

Should-fix: “Even at 10⁶ tokens ... KL ≈ 0.05 (L = 4) and 0.17 bits (L = 5); the gap is larger at longer L” overstates the evidence. `validation` contains L=4 at N=10,000/142,614/1,000,000, but L=5 only at N=10,000/142,614; its KL values are 0.173008 and 0.165832, and no exact-reference KL is computed at L≥6. In addition `exact_reference_fits` computes KL from the finite-prior exact MAP model, not from the unregularised maximum-entropy fit used for S₂. State the reference distribution and actual N for each KL, describe the nearly flat L=4 curve as evidence consistent with a misspecification gap rather than an asymptotic proof, and remove the unmeasured monotonic-longer-L claim.

## [P2] Keep forbidden-state descent consistent with the dense reference

`/Users/dtan/projects/languagefitness/langfit/longwords.py:509-514`

Should-fix: `descend` promises “maxent.descend's rule,” but local conditional-logit differences are not global energy differences when the starting string has forbidden pairs. For L=4, set every cell to zero except J₀₁(a,a)=J₂₃(a,a)=−∞ and start at `aaaa`: every single-letter neighbour still has a forbidden pair, so dense `maxent.descend` correctly stays at `aaaa`; this code treats changing position 0 to `b` as an infinite improvement and then changes position 2, returning `baba`. This affects the explicitly supported masked/S&B models even though the current length-study fits are finite. For forbidden starting states, evaluate whether the complete neighbour has finite weight and use the dense reference's plateau/tie rules; add this two-disjoint-violations case to the landscape equivalence test.

## [P2] Require an adequate prefix starting at length two

`/Users/dtan/projects/languagefitness/scripts/length_study.py:463-466`

Should-fix: the `not local` / `not glob` clauses allow either adequacy list to start at any passing length, although the exported criterion says “every length 2..L.” On a fresh documented `--lengths 4,5` run with validation enabled, passing L=4 and L=5 therefore yields L_max=5 and coverage including lengths 2–3 even though neither was evaluated; likewise a failed L=2 can be silently skipped when L=3 passes. Build each verdict from an explicit contiguous prefix beginning at 2, stopping at the first absent or failing length, and return an unavailable cutoff if length 2 has not passed.

## [P2] Use the measured coupling RMS in the method comparison

`/Users/dtan/projects/languagefitness/docs/first_pass/length.md:87-90`

Should-fix: “gauge-invariant RMS ≈ 1.5–2.6 nats at L = 4–5, §4.1” is not the statistic stored in the cited validation output. `validation[*].exact_reference.couplings.rms_b`, computed after double-centring, is 1.202884, 1.685901, 1.873094, 1.167350 and 1.728220 nats: a range of 1.17–1.87, or 1.69–1.73 at the common N. This unsupported larger range is used as evidence against expansion-based alternatives. Replace it with the explicitly identified exact-MAP RMS range, or store and identify the different statistic if another one was intended; do not use an unreproducible number to reject a method.

## [P2] Document the four-fifths refit used for every reported model

`/Users/dtan/projects/languagefitness/docs/first_pass/length.md:138-141`

Should-fix: “Tokens are split 3/5 train, 1/5 validation ... 1/5 test” omits that all reported final fits reincorporate the validation tokens. `prepare` defines `train = counts - test` and `inner = train - val`; `analyse_length` selects σ² on `inner`/`val`, then fits on `train`, approximately 4/5 of N. For example the L=16 record has 110,469 training tokens out of 137,998 and reports 1.3618 tokens/parameter; using the stated 3/5 instead gives about 1.02, apparently below the quoted adequacy threshold 1.17. State both phases explicitly and identify S₁, S₂ estimates, landscapes and tokens/parameter as computed on the final four-fifths refit, with only the test fifth held out.

## [added in second pass] Separate adaptive cluster expansion from weak-coupling approximations

Should-fix: the table groups ACE with mean field/TAP as “not consistent (biased expansion),” assigns it a matrix-inversion cost, and says strong couplings are “exactly where expansions break.” The cited [Cocco–Monasson paper](https://arxiv.org/abs/1102.3260) explicitly reports successful recovery “even at criticality and in the low temperature phase”; ACE adaptively truncates clusters and is exact in the all-cluster limit, rather than being intrinsically a weak-coupling or matrix-inversion method. This mischaracterisation is then used to motivate PLM as the only fit whose accuracy does not depend on mixing, although ACE is also deterministic. Give ACE its own row with cluster-size/truncation cost and convergence limitations, and state that it was not benchmarked here rather than claiming that the cited work rules it out.

## [added in second pass] Distinguish a common model family from a common estimand

Should-fix: “All target the same estimand (the pairwise maximum-entropy / maximum-likelihood model)” contradicts the report's later explanation and its validation results. For a misspecified data distribution, ML minimises joint cross-entropy while this implementation minimises the sum of conditional cross-entropies, so they generally define different population optima within the same pairwise family; [Hyvärinen's consistency result](https://pubmed.ncbi.nlm.nih.gov/16907626/) concerns correctly specified fully visible Boltzmann machines, not equality of these misspecified projections. Calling PLM merely a computational approximation to the same estimand obscures what the reported entropy and KL represent. Replace “same estimand” with “same model family,” state the two population objectives and correct-specification condition, and qualify divergence under misspecification as possible/generic rather than logically forced by S₃<S₂ alone.
