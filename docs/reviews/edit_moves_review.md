# Review of the edit-move model (langfit/edits.py, scripts/edit_study.py, docs/first_pass/neighbourhood.md)

## [P1] Distinguish length-conditioned means from deletion energy changes

`/Users/dtan/projects/languagefitness/docs/first_pass/neighbourhood.md:232-236`

Should-fix: “a deletion therefore lowers the energy of a typical word by ~3.5 bits” does not follow from subtracting averages over independently sampled length classes. Deletion parents are not distributed as P(w|L−1). A read-only calculation using the saved exact L=3,4 models gives mean ΔE=+7.5455 bits for a uniformly selected deletion of a model-distributed four-letter word, with only 11.16% downhill; even choosing its lowest-energy deletion gives mean ΔE=+2.6282 bits. The quoted −3.4268 bits is only the difference between separate length-conditioned means. Report that distinction, calculate move-conditioned ΔE if claiming a downhill direction, and remove the unsupported causal explanation repeated in §§6–7 and 10.

## [P1] Correct channel EM for observation-selection probabilities

`/Users/dtan/projects/languagefitness/langfit/edits.py:622-625`

Should-fix: the update “new /= new.sum()” estimates assignments among retained observations, but channel_study retains only non-lexicon four-letter observations with lexicon preimages and then passes these shares to decode as global forward-channel probabilities. Selection has type-dependent probability: insertions originate at L=3, substitutions/transpositions at L=4, and deletions at L=5; excluding real-word outcomes adds further type-dependent selection. An exact expected-count check with source probabilities P(ab)=0.8, P(cde)=0.1, P(fghi)=0.1 and planted shares 0.25 each, observing only interior length 3, returns 0.090909/0.090909/0.727273/0.090909 despite no sampling error. Include each type's probability of entering the retained sample in the conditional likelihood (and update), or restrict the interpretation to selected-sample assignment shares and do not reuse them as an unconditional decoder channel. The current planted-rate test uses only four-letter sources and does not exercise this bias.

## [P2] Separate steepest descent from zero-temperature Metropolis dynamics

`/Users/dtan/projects/languagefitness/docs/first_pass/neighbourhood.md:129-131`

Should-fix: “in the zero-temperature limit ... selects its lowest neighbour: steepest descent” is false for the Metropolis–Hastings chain just defined. As temperature vanishes, every proposed strictly downhill move is accepted, not only the steepest one: with energies 2→{1,0} and equal proposals, both destinations retain probability 1/2; equal-energy moves can also remain possible, unlike this code's strict stopping rule. Thus the computed deterministic basins are not the zero-temperature limit of those kinetics. Retain the valid Glauber/Kawasaki/insertion-deletion move analogies, but identify steepest descent as an additional deterministic landscape convention, or explicitly define a neighbourhood heat-bath rule whose limit selects a minimum. Also distinguish the inverse-temperature convention from the earlier fixed β=ln 2 when writing base-2 Boltzmann factors.

## [P2] Do not label the selected nonword fraction an error-rate upper bound

`/Users/dtan/projects/languagefitness/scripts/edit_study.py:409-413`

Should-fix: “its token share is therefore an upper bound on the per-token error rate” only accounts for false positives such as slang; it ignores false negatives. The numerator excludes real-word errors, strings with multiple errors or no lexicon preimage, and errors outside observed length 4. Even a corpus consisting entirely of real-word substitutions would contribute zero to this numerator despite a nonzero error rate. Consequently 0.032763 is neither an identified error probability nor a general upper bound, yet it is used as ε in every decoder posterior and performance result. Rename it as the observed selected-near-miss fraction, clearly declare using it as an uncalibrated channel assumption and provide sensitivity results, or estimate/bound ε with error-labelled data covering the missing classes.

## [P2] Normalize the channel for words without a valid transposition

`/Users/dtan/projects/languagefitness/langfit/edits.py:533-538`

Should-fix: the claimed “K(o | c)” is not a conditional probability distribution when c has no unequal adjacent letters, for example aaa or any one-letter word. Such a source has zero transposition events, but K(c|c) remains 1−ε and the other types retain ερ_t, so summing over every possible output gives 1−ερ_trans instead of 1. An independent check with ε=0.1 and ρ_trans=0.1 gives total mass 0.99, including outputs outside the modelled length range; this is not a boundary effect. These sources belong to the model and enter decode, making its scores/posteriors inconsistent with the stated generative channel. Explicitly send impossible event attempts to the no-error outcome or renormalize the available error types per source, then use the same convention in estimation and sampling and test row normalization.

## [P2] Describe the clean-token statistic as a corpus-token change rate

`/Users/dtan/projects/languagefitness/docs/first_pass/neighbourhood.md:365-365`

Should-fix: “false correction of a clean token” is measured by sampling all pooled Twitter tokens, not verified clean intended words: channel_study builds words from every by_len entry, performs no lexicon/clean-data filtering, and counts any decoder change as false. This includes the document's own known typo probes—for example waht has 92 observed tokens and decode(waht)=what, which this metric would count as an incorrect correction. The reported 1.31% is therefore an in-sample no-added-corruption change rate, not a measured false-positive rate on clean language. Relabel the table and headline claims accordingly, or evaluate on independently validated clean intended tokens; also state that the synthetic recovery sources and fitted prior use the same corpus.

## [P3] Describe the typing model's inhibition as directional

`/Users/dtan/projects/languagefitness/docs/first_pass/neighbourhood.md:197-200`

Nit: the passage says Rumelhart–Norman impose order by “mutual inhibition”, but their ordering mechanism is directional: earlier letter schemata inhibit later ones, producing an activation gradient; after a letter is typed its inhibition is removed. Logan's detailed account explicitly says “Each letter schema inhibits the schemata for all letters that follow it” (p. 747, https://www.psy.vanderbilt.edu/faculty/logan/Logan_2003_JEPHPP.pdf). Replace “mutual inhibition” with “directional, left-to-right inhibition”; symmetric mutual inhibition alone would not encode the prescribed order, so the distinction matters to the cited temporal-error explanation.

## [P3] Round the saved energy of “like” to 9.1 bits

`/Users/dtan/projects/languagefitness/docs/first_pass/neighbourhood.md:379-379`

Nit: the passage gives “like (9.2 bits)”, but exact.probes.liek.full[1].E_bits and the saved top-well entry both give 9.145327109590376 bits. At one decimal place this is 9.1, not 9.2; use 9.1 bits or 9.145 bits. The decoder outcome and the surrounding likelihood-ratio argument are unchanged.
