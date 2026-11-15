# Edit-move neighbourhoods: transpositions, insertions and deletions

*Code `langfit/edits.py`; study `scripts/edit_study.py` → `data/derived/edit_study.json`
(log `data/derived/edit_study.log`); tests `tests/test_edits.py`; fitted models
`data/derived/edit_models/`.*

## 1. Motivation and what is new on top of Stephens & Bialek

Stephens & Bialek ([S&B 2010](https://doi.org/10.1103/PhysRevE.81.066119),
arXiv:0801.0253) model four-letter words as a pairwise maximum-entropy (Potts) system
and define its landscape by **single-letter substitution**: a word is a well when every
substitution raises its energy, and a misspelling is corrected by relaxing downhill.
Substitution is a *spatial* accident (the wrong key). The probe `waht` → `what` is a
different accident: every key is right and the order is wrong — a *temporal* error. A
substitution neighbourhood cannot see it (`waht` and `what` differ in two positions), and
insertions or deletions change the word length, which a fixed-L model cannot represent
at all.

New here (none of it is in S&B):

1. **One configuration space across lengths.** Words of lengths 3, 4, 5 (and 1–5 for a
   boundary check) live in one grand-canonical space with one chemical potential per
   length class (§2). This makes energies of words of different length commensurable,
   which insertions and deletions require.
2. **Four move types as Markov-chain kinetics** (§3): substitution (Glauber),
   adjacent transposition (Kawasaki), insertion and deletion (grand-canonical Monte
   Carlo). The equilibrium law is the same for every move set; the move set defines
   the graph on which wells, descent and basins are computed.
3. **Exact landscape on the union graph**: every one of the 26³+26⁴+26⁵ = 12,355,928
   states, its best neighbour, strict minima, steepest-descent terminal and basin, by
   dense tensor operations (no sampling).
4. **Typo recovery per error type** by exhaustive enumeration of all single-edit
   corruptions of every observed word, under substitution-only and full move sets.
5. **A noisy-channel decoder** whose per-type error rates are estimated from the
   corpus by the Kernighan–Church–Gale bootstrap with a correction for which error
   types can enter the sample, compared with steepest descent.

## 2. Grand-canonical energy across lengths

**Maximum entropy.** Following Jaynes ([1957](https://doi.org/10.1103/PhysRev.106.620);
the principle that the least-biased distribution consistent with measured averages is
the Boltzmann form, the foundation of every model in this project), constrain, for each
length L, the joint frequencies P(L, wᵢ = a, wⱼ = b) of letter pairs at positions i < j.
These include the length distribution (sum any pair table of length L over a, b). The
entropy splits as S[P] = S[P(L)] + Σ_L P(L) S[P(· | L)], and every constraint is a
constraint on P(L) or on one conditional P(· | L); so the maximum-entropy solution is
P(L) times, for each L, the maximum-entropy conditional with the pairwise marginals of
length L — which is exactly S&B's model fitted to the length-L words alone. In
exponential form,

  P(w) = exp(λ_L + Σ_{i<j} V^L_ij(wᵢ, wⱼ)) / Ξ,  L = |w|,

where λ_L is the Lagrange multiplier of the length-class constraint (it could be
absorbed into the couplings, since the length indicator is a sum of pair indicators,
but keeping it separate makes the couplings S&B's within-length fit). Writing
H_L(w) = −Σ V^L_ij / ln 2 (bits) and Z_L = Σ_{|w|=L} 2^{−H_L(w)},

  **P(w) = P(L) · 2^{−H_L(w)} / Z_L,**

  **E(w) ≡ −log₂ P(w) = H_L(w) + log₂ Z_L − log₂ P(L) = H_L(w) − μ_L,
  μ_L = log₂ P(L) − log₂ Z_L.**

This is the grand-canonical weight exp(βμ_L − βH_L) with β = ln 2 per bit: the word
length plays the role of particle number, and μ_L is the Lagrange multiplier of the
length-class constraint. The textbook form μN is the special case of a *linear* μ_L,
which is what maximum entropy gives if only the mean length is constrained; here the
whole length distribution is constrained, so there is one μ_L per class. μ_L carries
the gauge of H_L (a constant added to block L's couplings moves log₂ Z_L and μ_L
together); only E is physical. In the normalised gauge (Z_L = 1, H_L = −log₂ P(w | L))
μ_L = log₂ P(L). Within one length E differs from S&B's energy by a constant, so the
substitution-only landscape is exactly S&B's (verified in `tests/test_edits.py`: the
substitution-only minima and descent map equal `maxent.local_minima`/`maxent.descend`).

P(L) is the token share of length L among all tokens of length 1–20 in the same pooled
Twitter data (`p_len` in the JSON). Restricting to the modelled lengths only shifts E by
the constant log₂ P(L ∈ {3,4,5}) and leaves every landscape statement unchanged.

**Widom insertion identity.** Widom's test-particle method
([Widom 1963](https://doi.org/10.1063/1.1734110); the potential-distribution theorem
that expresses the chemical potential as an average insertion Boltzmann factor, the
standard route to μ in simulation) has an exact discrete analogue here. For a length-L
word let W(w) = Σ over its 26(L+1) insertion events of 2^{−(E(w′)−E(w))}. Every
(L+1)-word arises from exactly L+1 (parent, event) pairs, so

  ⟨W⟩_L = Σ_w P(w | L) W(w) = (L+1) · P(L+1)/P(L)

provided every deletion parent of an allowed (L+1)-word is allowed. The ratio of
length-class probabilities — equivalently the difference of chemical potentials in the
normalised gauge — is the mean insertion factor. The study evaluates both sides
(`widom` in the JSON); in exact mode the deficit measures (L+1)-letter probability whose
parents carry a never-seen letter pair.

## 3. Moves as Markov-chain kinetics

Each edit type is an elementary move of a classic stochastic dynamics on configuration
space:

| move | edit | statistical-physics counterpart | conserved |
|---|---|---|---|
| `sub` | change one letter | single-spin flip, Glauber ([1963](https://doi.org/10.1063/1.1703954), *J. Math. Phys.* 4, 294; the first stochastic kinetic Ising model, with non-conserved single-site dynamics) | length |
| `trans` | swap adjacent letters i, i+1 | nearest-neighbour spin exchange, Kawasaki ([1966](https://doi.org/10.1103/PhysRev.145.224), *Phys. Rev.* 145, 224; exchange dynamics that conserves magnetisation, i.e. composition — the standard conserved kinetic Ising model) | length and letter composition |
| `del` | remove one letter, L → L−1 | particle removal in grand-canonical Monte Carlo (Norman & Filinov 1969, *High Temp.* 7, 216, one of the two origins of grand-canonical Monte Carlo; Adams [1974](https://doi.org/10.1080/00268977400102551), *Mol. Phys.* 28, 1241, the other, whose insertion/deletion acceptance rules are in standard use) | — |
| `ins` | insert one letter, L → L+1 | particle insertion (same; Widom 1963 test particle) | — |

The Kawasaki correspondence is the physical content of "the spatial aim is right, the
ordering is wrong": a transposition changes no letter, only which position holds it,
exactly as Kawasaki exchange moves no spin value into or out of the system. It is the
only one of the four moves that conserves composition.

**Equilibrium is the same under every move set.** Use any of these move sets as the
proposal of a Metropolis–Hastings chain (Metropolis et al.
[1953](https://doi.org/10.1063/1.1699114), the original Markov-chain Monte Carlo;
Hastings [1970](https://doi.org/10.1093/biomet/57.1.97), its generalisation to
asymmetric proposals) with acceptance min(1, 2^{−ΔE/T} q(w|w′)/q(w′|w)), ΔE = E(w′) − E(w)
in bits. Temperature convention: T is a dimensionless temperature in units of the data
temperature, so T = 1 targets P itself (the β = ln 2 of §2 is the conversion from bits to
nats at T = 1, not a free parameter). Detailed balance then holds with respect to P^{1/T}
for every move set, so P is stationary at T = 1 whenever the chain is irreducible.
Substitutions alone connect each length block (a Hamming graph) but never change L;
transpositions alone are not even irreducible within L (they conserve composition, like
Kawasaki dynamics); insertions/deletions connect the lengths. For the insertion–deletion
pair the proposal asymmetry is explicit: choosing a move class with probabilities
π_ins, π_del and then a uniform event, a word w and its insertion child w′ are linked by
m events in each direction (bijection of events, see `edits.preimages`), so

  q(w|w′)/q(w′|w) = (π_del · m/(L+1)) / (π_ins · m/(26(L+1))) = 26 π_del/π_ins,

playing the role of the volume factor in continuum grand-canonical insertion moves (the
alphabet size is the "volume" a new letter can occupy at a given position; the L+1
positions cancel against the L+1 deletion sites). **The move set changes kinetics and
basins, not the equilibrium distribution.**

**Steepest descent is a landscape convention, not the T → 0 limit of this chain.** As
T → 0 the Metropolis–Hastings chain accepts *every* proposed strictly downhill move (and
equal-energy moves with probability min(1, q ratio)), not only the steepest: from
energies 2 → {1, 0} with equal proposals both destinations keep probability ½. Its
absorbing states are exactly the strict minima defined below (a state with no
lower-or-equal neighbour), so **the set of wells is a property of the move graph alone**,
but its basins are stochastic and overlap. The deterministic basins computed here belong
to a different rule: a neighbourhood heat bath that jumps from w to w′ ∈ N(w) ∪ {w} with
probability ∝ 2^{−E(w′)/T}, whose T → 0 limit is exactly steepest descent (ties
resolved by the stated convention). That rule is not reversible with respect to P on an
irregular graph (the normalisers Σ_{N(w)∪{w}} 2^{−E/T} differ between w and w′), so it
is used only as the standard deterministic partition of a landscape into basins
(Stadler 2002), not as a model of the writer's dynamics.

**Landscape on a graph.** Following Stadler ([2002](https://doi.org/10.1007/3-540-45692-9_10),
"Fitness landscapes", *Lect. Notes Phys.* 585; the standard formalisation of a landscape
as a function on the graph defined by a move set, with basins defined by steepest
descent and the explicit warning that insertions/deletions give irregular graphs and that
ties make basins ambiguous), Reidys & Stadler
([2002](https://doi.org/10.1137/S0036144501395952), *SIAM Rev.* 44, 3; the mathematical
review of combinatorial landscapes) and Wales (*Energy Landscapes*, Cambridge University
Press 2003, ISBN 978-0-521-81415-7; the reference text on minima, basins and kinetics of
energy landscapes), the move set is a graph on configuration space and:

* **strict local minimum (well):** finite E strictly below every neighbour (S&B's
  strict definition; Stadler uses ≤, which would admit plateaus);
* **steepest descent:** move to the lowest-energy neighbour if strictly lower, else
  stop. **Tie rule:** exact ties go to the first in the order sub < trans < del < ins,
  then ascending position, then ascending letter. With substitution only this is
  exactly `maxent.descend`'s rule. Terminal states that are not strict minima
  (plateaus) are counted separately and never reported as wells;
* **basin** of a well: all states whose descent ends there; size reported as number
  of states, model probability mass Σ 2^{−E}, and observed token mass.

Adding moves can only remove wells: a minimum under the full move set is a minimum
under substitution (checked in the JSON field `full_subset_of_sub`).

**Boundary.** The modelled space is L ∈ {3, 4, 5}. Five-letter words have insertion
neighbours at L = 6 and three-letter words deletion neighbours at L = 2 that are not in
the main space. All wells at L = 5 (resp. 3) are therefore *minima with respect to the
modelled moves only*; true wells are a subset. Two checks quantify this: (i) an
empirical proxy for L = 6 (`upper_boundary`: an L = 5 well is "at risk" if some observed
six-letter insertion child has empirical token probability above the well's model
probability); (ii) the lower boundary is closed exactly by rerunning with lengths 1–5
(`closure_exact`), where the L = 1, 2 pairwise models are the exact empirical joint
distributions and there is no shorter word to delete to. Corruptions that leave the
modelled space (insertion into a five-letter word, deletion from a three-letter word)
are excluded from recovery denominators and reported as `null`.

## 4. The error channel: what the literature establishes

* **Damerau 1964** ([CACM 7, 171](https://doi.org/10.1145/363958.363994)) — the founding
  paper of computer spelling correction. It assumes a misspelling contains at most one
  error of four kinds (wrong, missing or extra letter, or one adjacent transposition)
  and reports that over 80% of the misspellings it examined were of this single-error
  kind (figure as summarised by Kukich [1992](https://doi.org/10.1145/146370.146380),
  *ACM Comput. Surv.* 24, 377, the standard survey of the field, and by Gimenes, Roman &
  Carvalho [2015](https://doi.org/10.1162/coli_a_00216), *Comput. Linguist.* 41, 175).
  This is why the neighbourhood here is exactly these four moves.
* **Levenshtein 1966** (*Sov. Phys. Dokl.* 10, 707,
  [ADS 1966SPhD...10..707L](https://ui.adsabs.harvard.edu/abs/1966SPhD...10..707L)) —
  codes correcting deletions, insertions and reversals (bit flips, i.e. substitutions);
  the origin of edit distance. Damerau's transposition added to Levenshtein's three
  operations gives the Damerau–Levenshtein neighbourhood used here.
* **Kernighan, Church & Gale 1990** ([COLING 1990, ACL C90-2036](https://aclanthology.org/C90-2036);
  [doi:10.3115/997939.997975](https://doi.org/10.3115/997939.997975)) — the noisy-channel
  formulation: rank candidate corrections c of a typo t by Pr(c)·Pr(t|c), with Pr(t|c)
  from four confusion matrices (deletion, insertion, substitution, reversal =
  transposition), estimated by a bootstrap — start from uniform confusions, correct the
  words the Unix `spell` program rejects in a large corpus, re-estimate, iterate. Their
  prior was word frequency, (freq(c)+0.5)/N. §8 uses this bootstrap with the
  maximum-entropy model as prior, corrected for which errors can enter the sample.
* **Typing as a serial-order process.** Lashley (1951, "The problem of serial order in
  behavior", in Jeffress (ed.), *Cerebral Mechanisms in Behavior*, Wiley, pp. 112–146;
  [APA record](https://psycnet.apa.org/record/1952-04498-003)) argued that order errors
  in skilled action (typing, speech) show that a sequence is planned in parallel and
  then serialised, not chained element by element — the classic argument that order is
  a separate degree of freedom from content. Rumelhart & Norman
  ([1982](https://doi.org/10.1207/s15516709cog0601_1), *Cogn. Sci.* 6, 1) built a model of
  skilled typing in which all letters of a word are activated at once and order comes
  from directional, left-to-right inhibition (each letter's schema inhibits those of the
  letters after it, and the inhibition is released once it is typed); it reproduces interkeystroke timing and the pattern of
  transposition errors — transpositions are *timing* failures of an otherwise correct
  plan. Grudin ([1983](https://doi.org/10.1007/978-1-4612-5470-6_6), in Cooper (ed.),
  *Cognitive Aspects of Skilled Typewriting*, Springer, pp. 121–143), from transcription
  typing errors: experts' transpositions are ~80% across-hand and almost always swap two
  successive letters; he attributes them to timing (cross-hand interstroke intervals are
  shorter, so the second letter "has more freedom to reach its key early"). His
  substitutions, by contrast, are spatial: immediately adjacent keys of the same row are
  31% (experts) to 59% (novices) of substitutions, same-finger column neighbours 8–16%.
  His Table 6.1 gives the error mix per typist: for six experts insertions 17–75%,
  omissions 5–21%, substitutions 4–69%, transpositions 3–11% of errors; for novices 9,
  4, 75 and 4%. Transpositions are a minority everywhere, but a systematic one.

Together these make the physical reading of §3 precise: a substitution is a spatial
accident (wrong site value — a spin flip), a transposition a temporal one (right
values, wrong order — a composition-conserving exchange), and insertion/deletion change
the number of keystrokes (particle number).

## 5. Data, fits and energy changes per move

Pooled Twitter counts of every year (2006–2023) of `twitter_year_L1-20_all.pkl.gz`
(all 41 shards; 838,106,954 tokens of length 1–20). Pairwise exact (S&B) models per
length, by exact enumeration with `langfit.potts` (L = 1, 2 are exact empirical joints),
all certified converged (marginals within 1% of a standard error, entropy error within
1% of the Basharin SE; `fits` in the JSON):

| L | tokens N | types | P(L) | S₂ (bits) | forbidden states | ⟨E⟩ over tokens of length L (bits) |
|---|---|---|---|---|---|---|
| 3 | 163,788,307 | 15,788 | 0.1954 | 6.111 | 0 | 8.467 |
| 4 | 156,879,957 | 102,664 | 0.1872 | 9.476 | 0 | 11.894 |
| 5 | 97,907,078 | 197,492 | 0.1168 | 12.286 | 0.73% | 15.383 |

At this N every letter pair at every position pair is observed for L ≤ 4, so the exact
model forbids nothing there; at L = 5, 0.73% of the 26⁵ states are forbidden.

**Energy change of a move ≠ difference of length means.** ⟨E⟩_L rises by about 3.4–3.5
bits per letter (`mean_token_energy_bits`), but that difference compares two
independently sampled length classes: the deletion parents of four-letter words are
not distributed as P(w | 3). The energy change of an actual move, ΔE = E(w′) − E(w)
(`move_delta_bits`; uniform over each word's events, token-weighted over observed words,
and also over the model distribution P(w | L) for L ≤ 4):

| from L | move | mean ΔE, uniform event | events downhill | mean ΔE, best event of the word | words with a downhill move |
|---|---|---|---|---|---|
| 3 | sub | +11.62 | 2.7% | +2.16 | 30% |
| 3 | trans | +15.92 | 2.1% | +12.28 | 3.1% |
| 3 | ins | +20.91 | 0.5% | +3.71 | 12% |
| 4 | sub | +12.03 | 3.6% | −0.17 | 45% |
| 4 | trans | +17.40 | 2.0% | +12.31 | 4.4% |
| 4 | del | +7.55 | 13.3% | +2.66 | 31% |
| 4 | ins | +22.15 | 0.5% | +4.32 | 12% |
| 5 | sub | +13.48 | 3.8% | −1.00 | 58% |
| 5 | trans | +15.70 | 2.9% | +8.93 | 8.5% |
| 5 | del | +7.73 | 17.2% | +1.17 | 45% |

(Observed tokens. L = 3 deletions from the 1–5 space: +6.19, 11.2% downhill. The model
average P(w | L) gives the same uniform-event means to the printed precision, which
follows from exact marginal matching: a uniform-event mean of E(w′) is a sum of
single-site and pair functions of w, whose averages the exact model reproduces.) No
move is downhill on average. Deletions are the least uphill and the most often
downhill move, and they are the move that dissolves wells (§6), but a typical word
is not lowered by deleting a letter.

**Widom check.** Both sides of the discrete Widom identity agree to machine precision at
every L (`widom`; e.g. L = 4: ⟨W⟩ = 3.1204 = 5·P(5)/P(4)), as they must here: with no
forbidden parents at L ≤ 4 there is no deficit. It is a consistency check on the
cross-length normalisation, not a test of the data.

## 6. Results: wells and basins on the move graph

Key `exact` in `edit_study.json` (space L = 3, 4, 5, 12,355,928 states).

| L | strict minima, substitution only (S&B) | strict minima, full move set | lost | killed by |
|---|---|---|---|---|
| 3 | 40 | 36 (21 with L = 1, 2 modelled) | 4 (19) | ins 4 (del 17, ins 2) |
| 4 | 133 | 90 | 43 | del 42, ins 1 |
| 5 | 412 | 265 † | 147 | del 140, trans 7 |

† minima with respect to the modelled moves; see boundary below.

* Every full-set minimum is a substitution minimum (`full_subset_of_sub` true at every
  L), as it must be.
* **Most lost wells have a lower one-letter-shorter neighbour**: *your → you,
  they → the, know → now, show → how, days → day, news → new, many → any, away → way,
  once → one* (L = 4); *there → here, years → year, every → very, never → ever,
  black → back, looks → look* (L = 5); and with L = 1, 2 modelled (`closure_exact`)
  *was → as, one → on, off → of, too → to, its → is*. The surviving wells are, by
  definition, words whose every deletion (and insertion, transposition) has higher
  energy: *the, and, for, you, are, all, but* (L = 3); *this, with, that, from, have,
  will, just, more, like, when* (L = 4); *about, today, their, great, would, thank,
  right, after, first, other* (L = 5).
* **Insertions kill a few short wells**: *our → your, own → down, don → down,
  est → best*, and at L = 4 *izza → pizza*.
* **Transpositions kill almost no real word.** The seven wells they remove are all at
  L = 5: six are rare non-word strings (*emath → meath, impah, uppty, othth, vaiwn,
  edqai*) and one is a real word lost to a model artefact (*actor → cator*: the pairwise
  model rates the non-word lower); `all_lost_wells`. So the transposition move barely
  changes which strings are wells; its effect is on kinetics — which typos reach which
  well (§7). A plausible reading [INFERENCE]: a composition-conserving swap of a word
  that is already a substitution well rarely lands on a still-lower string, because the
  swap keeps the letters but breaks the positional pair couplings the well is built on.
* Token share of words sitting at wells drops from 0.70 / 0.55 / 0.42 to 0.67 / 0.46 /
  0.35 (L = 3 / 4 / 5).
* **Basins merge.** 585 → 391 basins on the L = 3–5 space; median basin 7,021 → 10,177
  states, 99th percentile 166,176 → 307,905 (`basins_sub`, `basins_full`). Basins now
  span lengths: *the* drains 447 states under substitution and 109,396 states under
  the full move set; *you* 287 → 155,303; *now* 1,381 → 526,158. Five-letter basins
  shrink (*about* 104,754 → 22,245, *their* 129,276 → 28,818; `basin_comparison`). No
  plateau terminals occur
  (`plateau_terminals_full` = 0), so basins are unambiguous despite Stadler's caveat
  about ties.

**Boundary.** (i) Lower: three-letter wells lose 15 more wells once L = 1, 2 are modelled
(36 → 21), all to deletions; the L = 3 numbers in the main space are therefore upper
bounds, and `closure_exact` gives the closed values for L ≤ 4 (the L = 4 and 5 counts are
identical in both runs). (ii) Upper: L = 5 wells are minima only with respect to modelled
moves. By the empirical proxy (`upper_boundary`), 15 of the 265 (5.7%, 5.4% of their
model mass) have an observed six-letter insertion child whose token probability exceeds
the well's model probability; the true L = 5 count is therefore likely ≈ 250 or fewer.

## 7. Results: typo recovery

Every single-edit corruption of every observed word of each type, exhaustively
(`recovery_all_tokens`, token-weighted over words; `recovery_dictionary_tokens`
restricts to dictionary words and gives the same picture within 2–3 points).
Corruption type = what the typist did; insertion into L = 5 and omission from L = 3
leave the space and are excluded.

| error | substitution-only descent | full-move descent |
|---|---|---|
| substitution | 0.568 | 0.502 |
| adjacent transposition | 0.155 | **0.500** |
| insertion (extra letter) | 0 | **0.567** |
| omission (missing letter) | 0 | **0.261** |

Pooled over L = 3–5. The full move set triples transposition recovery and makes
insertion and omission errors recoverable at all; it *costs* 7 points on substitution
errors, i.e. some substituted strings that substitution-only descent returned to the
intended word now descend through a different move to another
well. Omissions are recovered worst; for them the intended word is one letter longer
than the typo, and insertions are the most uphill move on average (§5).

**Probe typos** (`probes`; energies in bits):

| typo | substitution-only path (S&B) | full move set |
|---|---|---|
| waht (92 tokens) | waht 25.96 → want 11.60 | waht → **what** 8.54 (trans) → that 7.18 (sub) |
| taht (89) | taht → tast → past | taht → **that** (trans) |
| jsut (158) | jsut → jout → jost → **just** | jsut → **just** (trans) |
| ahve (50) | ahve → abve → able | ahve → **have** (trans) |
| wiht (51) | wiht → wint → want | wiht → **with** (trans) |
| thsi (81) | thsi → thai → that | thsi → **this** (trans) |
| liek (93) | liek → lick | liek → **like** (trans) |
| konw (17) | konw → kong → long | konw → **know** 9.41 (trans) → now 8.25 (del) |

With the temporal move, the first and steepest step from every probe is exactly the
transposition back to the intended word. Substitution-only descent recovers only
*jsut* (by three substitutions through non-words). Steepest descent then
overshoots in two cases: *what* is not a well of the pairwise model (a substitution to
*that* is lower), and *know* is not a well once deletions exist (*now* is lower).

## 8. Noisy-channel decoder vs steepest descent

Bayes decoder (Kernighan, Church & Gale 1990): ĉ = argmax_c P(c) K(o | c) over o and
its one-edit preimages, with P(c) = 2^{−E(c)} the grand-canonical model and the
channel: no error with probability 1 − ε; otherwise an attempt of type t with
probability ε ρ_t, realised as one of the M_t(c) events of that type (25L
substitutions, 26(L+1) insertions, L deletions, one per adjacent unequal letter pair
for transpositions), so K(o | c) = ε ρ_t m(c→o)/M_t(c), m the number of events giving
o. A transposition attempt on a word with no unequal adjacent pair (e.g. *aaa*) leaves
it unchanged, so K(c | c) = 1 − ε + ε ρ_trans for such c and every row of K sums to 1
(`edits.channel_log2`; tested on *aaa, ab, a, what, aab*).

**Per-type rates from data, corrected for selection** (`channel`). KCG's bootstrap/EM on
the observed four-letter strings absent from a reference lexicon (system word list ∪
vocabulary of the project's 21 edited Gutenberg texts; 250,597 forms), each treated as
one error from a lexicon word one edit away. L = 4 is the only length whose source
lengths 3, 4, 5 are all modelled. This retained sample does not admit every error type
with the same probability: four-letter insertion errors come from three-letter sources,
omissions from five-letter sources, and an error whose output is itself a lexicon word
is never retained. The probability that an attempt of type t on a lexicon source
c ~ P enters the sample, s_t = Σ_c P(c | lexicon) f_t(c) (f_t = fraction of c's type-t
events giving a non-lexicon four-letter string; `selection_probabilities`), is
sub 0.319, trans 0.356, ins 0.422, del 0.094. Conditioning the likelihood on retention,
P(o | retained) = Σ_{c,t} P(c) ρ_t m/M_t(c) / Σ_t ρ_t s_t, gives the EM fixed point
ρ_t ∝ n_t / s_t, with n_t the expected retained count of type t (derivation in
`em_channel`; the test reproduces the reviewer's bias case: sources P(ab) = 0.8,
P(cde) = P(fghi) = 0.1, planted ρ = ¼ each, retained length 3: uncorrected shares give
ρ_ins = 0.727, corrected EM returns ¼ for every type). Token-weighted (the only
weighting for which this is a likelihood), 69,256 typo types, 26 iterations:

| | sub | trans | ins | del |
|---|---|---|---|---|
| shares among retained near-misses (uncorrected) | 0.652 | 0.013 | 0.218 | 0.117 |
| forward channel ρ_t (selection-corrected; used below) | 0.533 | 0.009 | 0.135 | 0.323 |

(Type-weighted uncorrected shares, for description only: 0.602 / 0.035 / 0.337 / 0.026.)
Caveats: the "typos" are non-lexicon near-misses on Twitter (slang, names, expressive
spellings included); the correction assumes P is the distribution of intended words and
that every retained string is a single error.

**ε is not identified.** The token share of retained near-misses among four-letter
tokens, 0.0328, is neither the error rate nor a bound on it: it counts slang as errors
and misses real-word errors, multi-edit errors and errors with no lexicon neighbour. It
is used only as a working value, with every decoder result reported on the half-decade
grid ε ∈ {0.0033, 0.0104, 0.0328, 0.104, 0.328} (`eps_grid`).

**Paired Monte Carlo** (10,000 synthetic corruptions per type of corpus tokens drawn
token-weighted; same samples for all decoders; SE ≤ 0.005). The corrupted sources, the
fitted prior and the channel estimate all come from the same corpus, so these are
in-sample numbers.

| error | sub-only descent | full descent | decoder ε = 0.0033 | 0.0104 | **0.0328** | 0.104 | 0.328 |
|---|---|---|---|---|---|---|---|
| substitution | 0.570 | 0.509 | 0.239 | 0.318 | 0.410 | 0.489 | 0.564 |
| transposition | 0.154 | 0.505 | 0.408 | 0.509 | 0.573 | 0.627 | 0.669 |
| insertion | 0 | 0.565 | 0.583 | 0.644 | 0.691 | 0.736 | 0.780 |
| omission | 0 | 0.270 | 0.228 | 0.292 | 0.390 | 0.461 | 0.520 |
| change rate, uncorrupted corpus tokens | 0.431 | 0.497 | 0.005 | 0.008 | 0.014 | 0.022 | 0.040 |
| change rate, uncorrupted lexicon-word tokens (95.5% of the sample) | 0.406 | 0.475 | 0.001 | 0.003 | 0.006 | 0.011 | 0.022 |

The change rate is measured on observed corpus tokens with no corruption added. It is
not a false-positive rate on verified clean text: the sample includes real typos (e.g.
*waht*, 92 tokens, which the decoder changes to *what*), so some of the "changes" are
correct; the lexicon-word row is the closer proxy.

Reading: steepest descent ignores both the channel's no-error probability and the
factor 1/M_t(c), which penalises explanations that need one specific event out of many
(an insertion of a given letter at a given place is one of 26(L+1) events). It
therefore moves about half of all uncorrupted tokens, because only about half of tokens
sit at wells of the pairwise landscape. The decoder changes 0.5–4% of them across the ε
grid. It beats full descent on insertions at every ε, on omissions at ε ≥ 0.0104 and
on transpositions at ε ≥ 0.0328 (at 0.0104 the two are level, 0.509 vs 0.505, within one
SE). Recovery rises with ε for every type while the change rate rises too: ε trades
recall against changing correct tokens. Substitutions are where descent does better
unless ε ≥ 0.33. Probes (working ε): the decoder recovers 7 of 8 at every ε
(*waht → what*, posterior 0.51–0.71 across the grid; *taht, jsut, ahve, wiht, thsi,
konw* ≥ 0.93). *liek* is kept as observed at every ε: the pairwise model rates *liek*
(18.4 bits) too plausible next to *like* (9.1 bits) for a transposition probability
ε ρ_trans / M_trans(like) ≈ 0.0328 × 0.0094 / 3 ≈ 10⁻⁴.

## 9. Robustness: map couplings (L = 3, 4 only)

`map_check_L34` compares exact and Gaussian-prior (map) couplings on the L = 3, 4 space;
σ² = 10^{0.5} chosen by 4-fold CV at both lengths, interior of the grid; CV curves are
flat to < 10⁻⁴ bits, well inside one SE of 2.4·10⁻⁴. Every minimum count, recovery rate
and probe path is identical in both modes, as expected when nothing is forbidden and
N ~ 10⁸.

**Not run: map at L = 5.** Choosing σ² at L = 5 takes ~20 fits on the 26⁵ grid; on the
shared machine that was not affordable and the check was dropped. What is lost: at L = 5
the exact model forbids 0.73% of states (E = +∞). Forbidden strings still relax under
descent (every finite neighbour is lower), so every typo has a terminal and every
recovery number is defined; but conclusions that could change with finite energies
there are (i) basin sizes and memberships near forbidden L = 5 regions (a forbidden
string drains to whichever finite neighbour is lowest, not along a graded slope), (ii)
insertion moves from L = 4 into L = 5 and omission typos of five-letter words (the
insertion target or the corrupted string may be forbidden, ruling a move out that a
map model would rank), and (iii) the transposition-killed L = 5 "wells", which are
low-count strings whose energy depends most on unseen-pair cells. The L = 3, 4 results
do not depend on this.

## 10. What is new relative to S&B, and its justification

* S&B's landscape is the substitution graph of one length (Glauber dynamics). Here the
  neighbourhood is Damerau's full single-error set, each move an established
  statistical-physics kinetics (Glauber, Kawasaki, grand-canonical insertion/deletion),
  and words of different length share one maximum-entropy ensemble with one chemical
  potential per length class (Jaynes). The equilibrium law — S&B's pairwise model per
  length, weighted by P(L) — is unchanged by the choice of moves (detailed balance,
  Metropolis/Hastings); only basins, descent and recovery change.
* The temporal error class that motivated this (`waht`) is exactly the composition-
  conserving exchange; in the data it is the steepest first step from every probe, and
  it raises transposition recovery from 15% to 50% (57% with the Bayes channel at the
  working ε, 41–67% across the ε grid).
* Novel findings: (1) deletion moves dissolve about a third of S&B's wells (43 of 133
  at L = 4, 147 of 412 at L = 5), although no move lowers the energy of a typical
  word: a uniform random deletion raises E by ≈ 7.5 bits on average and is downhill
  for 13–17% of events; (2) transpositions change kinetics but essentially not the
  set of wells; (3) a Bayes decoder whose per-type rates are estimated from the data
  with a selection correction, and which includes the channel's no-error and 1/M_t
  factors, changes 0.5–4% of uncorrupted tokens (vs ~50% for descent) and beats descent
  on insertion errors at every ε and on omission and transposition errors once
  ε ≳ 0.01–0.03. The size of ε itself is
  not identified by these data.

## Reproduction

`python scripts/edit_study.py --workers 1` (≈ 35 min for the uncached L = 5 exact fit,
then ≈ 4 min; models cached in `data/derived/edit_models/`, reusable via
`edit_study.load_model(L, mode)`). Tests: `python -m pytest tests/test_edits.py -q`.
