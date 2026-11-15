# Four-letter words as a pairwise Potts system: Stephens & Bialek reproduced and extended

This document covers the four-letter maximum-entropy study in this repository. It has six parts:

1. Stephens & Bialek's (S&B) model as they published it.
2. How word statistics map onto energetics, in their formulation and in ours.
3. Where our implementation follows S&B and where it departs, item by item.
4. Every result, each with its source field.
5. Limitations.
6. How to reproduce the study.

**Sources of numbers.**
- All results come from `data/derived/maxent_study.json`, written by `scripts/maxent_study.py` (log: `data/derived/maxent_study.log`), or from `data/derived/twitter_L4_by_year.json` / `bluesky_L4_by_year.json`.
- Field paths are written as `replication.S_2`. In `corpora[Twitter 2023, rep 0].probes.tmrw`, `corpora` is the list of per-subsample rows, selected by `name` and `rep`.
- "Mean ± sd" is taken over the 5 subsamples (`meta.reps`), with `ddof=1`.
- Equations are checked against `langfit/maxent.py`, and the implementing function is named at each one.
- Claims that are not stored in any artifact are marked as such.

---

## 1. The model as published (S&B, arXiv:0801.0253v1)

**Version used.** We reproduce arXiv:0801.0253v1 (2007), the PDF at `data/refs/stephens_bialek_2007.pdf`. Equation numbers, reference numbers and benchmark values below refer to that version. The journal version, Phys. Rev. E 81, 066119 (2010), differs in three ways (checked against the authors' PDF, https://swh.princeton.edu/~wbialek/our_papers/stephens+bialek_10.pdf):
- **Sign.** Its eq. (2) writes $P^{(2)}=\exp[-\sum_{i>j}V_{ij}]/Z$. Our $V$ (and the preprint's) is therefore $-V$ in the published convention. The implementation needs no sign change.
- **$S_2$.** Its Fig. 2 gives $S_2=7.471\pm0.006$ bits, against the preprint's 7.48.
- **Reference numbering.** Its references are renumbered: the corpus, error-bar and iterative-scaling citations are refs. 17/18/20 instead of the preprint's 16/19/21.

### 1.1 State space
A four-letter word $w=(l_1,l_2,l_3,l_4)$, $l_i\in\{a,\dots,z\}$, is a configuration of $L=4$ Potts spins with $A=26$ states each. The space has $A^L=26^4=456{,}976$ states, and S&B note it is small enough for "relatively simple algorithms". The empirical distribution $\hat P(w)$ is the token frequency of $w$ among the $N$ four-letter tokens of a corpus.

### 1.2 Maximum-entropy problem and the Boltzmann form
S&B approximate $\hat P$ by the distribution of maximum entropy that reproduces the six pairwise marginals ([Jaynes 1957]: the least-biased distribution consistent with given expectations; this is the founding paper of maximum-entropy inference in statistical mechanics):

$$
\max_P\; H[P]=-\sum_w P(w)\ln P(w)\quad\text{s.t.}\quad
\sum_{w:\,(l_i,l_j)=(a,b)}P(w)=\hat\mu_{ij}(a,b)\;\;\forall\, i<j,\;a,b,\qquad \sum_w P(w)=1 .
$$

Introduce one multiplier $V_{ij}(a,b)$ per constraint cell and one multiplier $\lambda_0$ for normalisation:

$$
\mathcal L=H[P]+\sum_{i<j}\sum_{a,b}V_{ij}(a,b)\Big[\sum_{w}P(w)\,\mathbb 1[(l_i,l_j)=(a,b)]-\hat\mu_{ij}(a,b)\Big]+\lambda_0\Big[\sum_wP(w)-1\Big].
$$

Stationarity, $\partial\mathcal L/\partial P(w)=-\ln P(w)-1+\sum_{i<j}V_{ij}(l_i,l_j)+\lambda_0=0$, gives S&B's eq. (2):

$$
P^{(2)}(w)=\frac1Z\exp\Big[\sum_{i<j}V_{ij}(l_i,l_j)\Big],\qquad Z=\sum_w \exp\Big[\sum_{i<j}V_{ij}(l_i,l_j)\Big].
$$

This is a Potts model with pairwise couplings between all positions and unit temperature. It is not a Markov chain: positions 1 and 4 interact directly. The multipliers are fixed by the marginal equations, S&B eq. (3): $\hat\mu_{12}(l,l')=\sum_{l_3,l_4}P^{(2)}(l,l',l_3,l_4)$, and likewise for the other five pairs.

### 1.3 Gauge freedom and parameter count
Two kinds of transformation leave $\sum_{i<j}V_{ij}$ unchanged, up to a constant that $Z$ absorbs:
- a constant added to any $V_{ij}$;
- a function $u(a)$ of one position $i$ added to $V_{ij}(a,\cdot)$ and subtracted from $V_{ik}(a,\cdot)$.

S&B remove only the first kind ("the zero of energy is arbitrary") and quote $6\cdot(26^2-1)=4050$ parameters. The second kind removes more. The functions on $26^4$ that depend on at most two positions span a space of dimension $\sum_{k=0}^{2}\binom4k(A-1)^k=1+100+3750$. After subtracting normalisation, the number of identifiable parameters with full support is $4\cdot25+6\cdot25^2=3850$. This count is ours, not S&B's. In the exact fits below, forbidden cells reduce it further. Our code keeps the over-complete parametrisation, and the Hessian is singular along the gauge directions. The `newton` helper inside `maxent.fit` notes this and stops CG at relative residual $10^{-3}$.

### 1.4 Entropy hierarchy and multi-information
Let $S_k$ be the maximum entropy consistent with all $k$-point marginals. Each added constraint shrinks the feasible set, so

$$
S_{\rm rand}=L\log_2A\;\ge\;S_1\;\ge\;S_2\;\ge\;S_3\;\ge\;S_{\rm full}=H[\hat P].
$$

[Schneidman, Still, Berry & Bialek 2003] define the connected information of order $k$ as $I^{(k)}=S_{k-1}-S_k$. It decomposes the multi-information exactly: $I_N=S_1-S_{\rm full}=\sum_{k=2}^{L}I^{(k)}$. This is the standard hierarchy behind every "pairwise model captures X%" claim since. S&B report the captured share $I^{(2)}/I_N=(S_1-S_2)/(S_1-S_{\rm full})$. The independent model is the product of position-specific letter frequencies (S&B eq. 1), and $S_1=\sum_iH(l_i)$.

### 1.5 Fitting algorithm
S&B solve eq. (3) by iterative scaling, citing [Darroch & Ratcliff 1972]. Generalised iterative scaling is the classical fixed-point algorithm for log-linear models; it converges monotonically but linearly. Their Fig. 1b shows that the fitted marginals match the empirical ones "to high precision". The paper gives no quantitative stopping rule.

### 1.6 Energy landscape and local minima
S&B read eq. (2) as an energy landscape "much in the spirit of Hopfield's energy landscape for the states of neural networks" ([Hopfield 1982]: attractor memories as energy minima). A local minimum is a string for which "any single letter change will result in an increase in the energy". They say such stable words allow "any single letter spelling error [to] be corrected by relaxing to the nearest local minimum".

### 1.7 Published numbers
Source: arXiv:0801.0253 text, Table I, Figs. 2–3. They are mirrored in `meta.sb_published`.

| quantity | S&B value | note |
|---|---|---|
| Austen corpus | 8 texts, 676,302 words, 7,114 unique words | dictionary-checked against 12dicts (their ref. 18) |
| four-letter types / tokens | 763 / 135,441 | |
| $S_{\rm rand}$ | 18.802 bits (Fig. 2: 18.803) | $4\log_2 26$ |
| $S_{\rm ind}=S_1$ | 14.083 ± 0.001 bits | |
| $S_2$ | 7.48 bits (PRE 2010 Fig. 2: 7.471 ± 0.006) | |
| $S_{\rm full}$ | 6.918 ± 0.003 bits (text: 6.92) | error bars per their ref. 19 |
| $I_N=S_1-S_{\rm full}$ | 7.163 bits | |
| captured $I^{(2)}/I_N$ | 92% (ANC: 89%) | |
| $2^{S_1-S_2}$ | ~100 | |
| model mass on unseen strings | 15% | 1/5 of it on dictionary words |
| local minima | 136, of which 118 real words, carrying 63.5% | **ANC**, not Austen; Austen only "similar" |
| Fig. 3 inset, 25 most probable non-words (row order) | mast tome sime hame sade / whem tike liss mike welt / hove mide lave dore wert / mady mone bady sere gove / dees thom whan fent core | read from the rendered PDF page 3 |

---

## 2. Statistics ↔ energetics

Notation follows `langfit/maxent.py`:
- $\phi(w)\in\{0,1\}^D$ is the vector of clique-cell indicators. Every word activates exactly one cell per clique (`_features`).
- $\theta\in\mathbb R^D$ is the packed vector of all $V_c$ over free cells (`_pack`/`_unpack`).
- $\hat\mu$ is the packed vector of empirical clique marginals.

The same formalism covers S&B's pairwise model ($c=\{i,j\}$) and our order-1 and order-3 models (`cliques(order, L)`).

**Energy.** Define the unnormalised energy $\mathcal E_\theta(w)=-\theta\cdot\phi(w)=-\sum_cV_c(w_c)$. Its negative is `MaxEnt.log_weights`. Then $P_\theta=e^{-\mathcal E_\theta}/Z$ at unit temperature. The normalised energy

$$E(w)=-\ln P_\theta(w)=\mathcal E_\theta(w)+\ln Z$$

is `MaxEnt.energy` (nats, $+\infty$ for forbidden strings). The study reports it in bits, $E/\ln2=-\log_2P$ (`Eb = E.ravel() / LN2` in `scripts/maxent_study.py:analyse`).

**Entropy as free energy.** With $F=-\ln Z$ (β = 1),

$$S=\langle E\rangle=\langle\mathcal E\rangle+\ln Z=\langle\mathcal E\rangle-F\quad\text{(nats)},\qquad S_{\rm bits}=S/\ln2 .$$

`MaxEnt.entropy_bits` evaluates $-\sum_wP\ln P/\ln 2$ over $P>0$.

**The dual.** The function minimised by `fit` (inner function `fun`) is

$$f(\theta)=\ln Z(\theta)-\theta\cdot\hat\mu\;\big[+\lambda\|\theta\|^2\ \text{in MAP mode}\big].$$

Since $\ln P_\theta(w)=\theta\cdot\phi(w)-\ln Z$, $f(\theta)=-\frac1N\sum_n\ln P_\theta(w_n)$ is the per-token negative log-likelihood, so exact maximum entropy and maximum likelihood coincide. In the code, `reg = 2.0 * lam` and `fun` adds `0.5 * reg * θ·θ` $=\lambda\|\theta\|^2$.

**Moment matching is the gradient** (inner function `jac`):

$$\nabla f=\langle\phi\rangle_\theta-\hat\mu\;[+2\lambda\theta],$$

that is, model marginals minus empirical marginals. At the exact-mode optimum the six pairwise marginals are matched, which is eq. (3).

**The Hessian is the covariance of the clique indicators** (inner function `hessp`, `FT @ (p * (F @ v)) - m * (m @ v) + reg * v`):

$$\nabla^2 f=\langle\phi\phi^{\top}\rangle_\theta-\langle\phi\rangle_\theta\langle\phi\rangle_\theta^{\top}=\mathrm{Cov}_\theta[\phi]\;[+2\lambda\mathbb 1],$$

which is the per-token Fisher information of the exponential family ([Wainwright & Jordan 2008], §3: cumulant-generating function, moment map, and Fisher information for exponential families). It is positive semidefinite, so $f$ is convex. Its null space is the gauge directions of §1.3.

**The minimum of the dual is the maximum entropy.** Three identities give this:
- $H(P_\theta)=\ln Z-\theta\cdot\langle\phi\rangle_\theta$, so $f(\theta)-H(P_\theta)=\theta\cdot\nabla f$ in exact mode.
- For any $Q$ whose marginals equal $\hat\mu$, $-\sum_wQ\ln P_\theta=\ln Z-\theta\cdot\hat\mu=f(\theta)$. Hence $f(\theta)-H(Q)=D_{\rm KL}(Q\|P_\theta)\ge0$ (weak duality).
- At $\nabla f=0$ the model $P_{\theta^*}$ is itself feasible, so $S^*=\max_QH(Q)=\min_\theta f=H(P_{\theta^*})$.

Taking $Q=\hat P$ gives $S_k-S_{\rm full}=D^{(\ln)}_{\rm KL}(\hat P\|P_k)/\ln 2$ bits. Here $D^{(\ln)}_{\rm KL}$ is the natural-log divergence defined by $f-H(Q)$ above, while $S_k$ and $S_{\rm full}$ are in bits as in §1.4. The uncaptured multi-information is therefore exactly the code length lost by coding real words with the order-$k$ model.

**Why exact matching forbids unseen pairs.** Suppose $\hat\mu_{ij}(a,b)=0$. Every feasible $Q$ is non-negative and sums to zero over the strings with $(l_i,l_j)=(a,b)$, so it vanishes on all of them. The maximum-entropy solution must vanish there too, which in exponential form requires $V_{ij}(a,b)=-\infty$. [Csiszár 1975] gives the general statement: the maximum-entropy (I-projection) solution has the largest support among feasible distributions, which is the classic geometric treatment of minimum-divergence problems. `fit` implements this with `masks = [mu > 0 ...]` and `fill = -np.inf`, and `_features` marks every word containing a masked cell as `forbidden` (log-weight $-\infty$). One consequence: a single unseen letter pair at any of the six position pairs makes a string impossible. This is the main reason exact-fit statistics depend on $N$ (§3.6).

Our code assumes the support is exactly the set of strings whose six cells are all seen. If the constraints force further zeros, the optimum lies on the boundary and some couplings diverge. Such cases are exactly where the stopping rule of §3.3 is least reliable: it uses a local quadratic estimate, not a bound (see the counterexample there). Every fit met the stopping rule, but that does not certify the entropy in those cases.

**Order 1 in closed form** (`_independent`). The maximum-entropy distribution with given single-site marginals is their product, so $V_i=\ln\hat\mu_i$ with $-\infty$ for unused letters, and $S_1=\sum_iH(l_i)$. No optimisation is involved.

**The Gaussian-prior (MAP) variant** (`fit(..., mode="l2")`, `prior_lambda`). Place an independent prior $V\sim\mathcal N(0,\sigma^2)$ on every cell of every clique tensor. In l2 mode all cells are free, including unseen ones (`masks` all True, `fill = 0`). Then

$$-\ln[\text{likelihood}\times\text{prior}]=N f_0(\theta)+\frac{\|\theta\|^2}{2\sigma^2}+{\rm const}=N\big[f_0(\theta)+\lambda\|\theta\|^2\big],\qquad \lambda=\frac{1}{2\sigma^2N},$$

which is `prior_lambda(sigma2, n_tokens)`, with $N$ the number of training tokens. This is the standard Gaussian-prior smoothing of maximum-entropy models ([Chen & Rosenfeld 2000]: a comparison of smoothing methods for ME models in which the Gaussian prior performed best). The MAP model satisfies $\langle\phi\rangle_\theta=\hat\mu-2\lambda\theta$, so it does not match the marginals and its entropy is not $S_2$. It exists for one purpose: an exact model gives held-out words containing an unseen pair zero probability, i.e. infinite code length. The MAP model is used only for held-out code lengths (§3.8). Every in-sample entropy, landscape, minimum and recovery figure below comes from exact fits. The $\lambda\|\theta\|^2$ term also makes $f$ strictly convex and selects the minimum-norm gauge.

---

## 3. Adherence and departures, item by item

### 3.1 Model and constraints: same
We fit eq. (2) with the six pairwise marginal constraints of eq. (3) in exact mode, with $S_1$, $S_2$ and (replication only) $S_3$ as in §1.4. Code: `scripts/maxent_study.py:replication` and `job_corpus`, calling `maxent.fit(wc, order, "exact")`.

### 3.2 Solver: different algorithm, same optimum
S&B used generalised iterative scaling [Darroch & Ratcliff 1972]. We minimise the same convex dual $f$ with SciPy's `trust-krylov`, a trust-region Newton method whose subproblem is solved by the Lanczos/GLTR method of [Gould, Lucidi, Roma & Toint 1999]. Hessian-vector products are `hessp` as in §2. The initial point is $\theta_0=\ln\hat\mu_c/|{\rm cliques}|$ on free cells. The trust region can stall once the objective decrease drops below float64 rounding of $|f|\approx10$ nats, so `fit` follows with up to 30 full Newton steps $d\approx-H^{+}g$ from CG, each accepted only if it lowers the larger of the two stopping ratios.

Because $f$ is convex and its minimiser is unique up to gauge, the maximum-entropy distribution $P^*$ does not depend on the algorithm. Only speed and the stopping rule differ.

### 3.3 Stopping rule: relative to sampling noise (ours; S&B state none)
`fit` stops when, with `resolution = 0.01`, both of the following hold.

(a) Every marginal cell satisfies $|\nabla_c f|<0.01\,{\rm se}_c$, where ${\rm se}_c=\max\big(\sqrt{\hat\mu_c(1-\hat\mu_c)/N},\,1/N\big)$ is the binomial SE of a cell frequency, floored at one token.

(b) In exact mode, a local quadratic estimate of the entropy error is below $0.01\,{\rm se}_S$. The starting point is a valid inequality that involves the true dual gap $f-f^*$:

$$|H(P_\theta)-S^*|\le|f(\theta)-H(P_\theta)|+\big(f(\theta)-f^*\big)=|g\cdot\theta|+\big(f-f^*\big).$$

The code then replaces $f-f^*$ by the local quadratic estimate

$$f-f^*\approx\tfrac12\,g^{\top}H^{+}g=\tfrac12\lambda_N^2 ,$$

where $H^{+}$ is the pseudo-inverse (the Hessian is singular along gauge directions, §1.3), approximated by CG stopped at relative residual $10^{-3}$. $\lambda_N$ is the Newton decrement ([Boyd & Vandenberghe 2004], §9.5.1: the standard affine-invariant estimate of suboptimality, exact only for the local quadratic model). In the code, `newton` returns `dec = -g·d` $\approx\lambda_N^2$, and `ent_se` returns $(\tfrac12|{\rm dec}|+|g\cdot\theta|)/{\rm se}_S$.

**This is an estimate, not a rigorous certificate, particularly for boundary optima.** Here is a counterexample in which every allowed pair cell is observed, yet the constraints force extra zeros:
- Take three binary positions (a fixed fourth letter changes nothing) and put uniform empirical mass on the six non-constant triples.
- The pair marginals force the two constant triples, 000 and 111, to zero, so the optimum is on the boundary.
- Let the potential be $-x$ times the number of equal pairs. Then $P_\theta\propto e^{-x}$ on non-constant triples and $e^{-3x}$ on constant ones.
- The true dual gap is $f-f^*=\ln(1+e^{-2x}/3)$, while half the squared decrement is $e^{-2x}/6$. The quadratic estimate understates the gap by a factor that tends to 2 as $x\to\infty$: at $x=8$ the values are $3.751\times10^{-8}$ and $1.876\times10^{-8}$ nats.
- The resulting entropy-error estimate, $6.19\times10^{-7}$ nats, also understates the actual error of $6.38\times10^{-7}$ nats.

These values come from a direct numerical check with `numpy.linalg.pinv`; it was run once and is not stored. When the optimum is interior, i.e. the maximum-entropy support equals the set of unmasked strings, the quadratic estimate becomes asymptotically exact as $g\to0$. We have not tested which of our exact fits, if any, have boundary optima.

The scale ${\rm se}_S=\max\big(\sqrt{{\rm Var}_{\hat p}[\ln\hat p]/N},\,(K-1)/(2N)\big)$ (nats) combines two pieces:
- the first-order sampling sd of the plug-in entropy, from [Basharin 1959], the delta-method variance of the plug-in entropy estimator;
- a floor at the second-order bias scale $(K-1)/(2N)$ of [Miller 1955], the original plug-in bias correction. The floor is needed because the first-order term vanishes for uniform counts.

Achieved values, as maxima over all 85 corpus fits: marginal error 0.0093 SE and estimated entropy error 0.0027 SE. Sources: `corpora[*].fit_marginal_in_se` and `fit_entropy_error_in_se`, shown per corpus in §4.2. All fits report `converged = True`, including `replication.converged`. A factor-2 understatement of the gap term, as in the counterexample, would still leave every estimated entropy error below 0.01 SE. This is reassurance only, not a bound.

### 3.4 Order 1: closed form
See §2 (`_independent`). $S_1$ is exact to rounding.

### 3.5 Tokenisation and dictionary: different, documented
- **Tokenisation** (`langfit/corpus.py`). Text is lowercased and `_` becomes a space. A token is a maximal run of `[\w']+`, kept only if it is exactly four ASCII letters. "don't" and "café" are therefore dropped. S&B do not specify their tokeniser.
- **Texts.** The same eight Austen works as S&B ref. 16 (`corpus.AUSTEN`: Emma, Lady Susan, Love and Freindship, Mansfield Park, Northanger Abbey, Persuasion, Pride and Prejudice, Sense and Sensibility), downloaded from Project Gutenberg. The Gutenberg editions may differ from S&B's 2007 files.
- **Dictionary.**
  - S&B: 12dicts (their ref. 18).
  - Ours: `/usr/share/dict/words` → `web2`. Its README reads "Webster's Second International … all 234,936 words worth. The 1934 copyright has lapsed"; the file has 235,976 lines.
  - After `corpus.word_list()` lowercasing and the four-letter ASCII filter in `dictionary_mask`, it has 5,006 distinct four-letter entries. This count comes from a read-only run of `corpus.word_list()`; it is not stored in the JSON.
  - web2 contains many archaic and obscure words but lacks many inflected forms (`eyes`, `ones`, `says` are absent; so are `okay`, `blog`, `haha`).
- **Resulting counts.**
  - Dictionary-filtered Austen: 141,198 tokens, 820 types (`replication.tokens`, `replication.types`), against S&B's 135,441 / 763.
  - Unfiltered Austen: 142,614 tokens, 966 types (`meta.corpora.Austen`).

### 3.6 Matched-N rule for cross-corpus comparison (ours)
Exact-fit quantities depend on the sample size $N$ for three reasons:
- every unseen pair cell is forbidden, and the number of seen cells grows with $N$;
- the plug-in $S_{\rm full}$ is biased low by about $(K-1)/(2N\ln2)$ bits [Miller 1955];
- minima counts and unseen mass are $N$-dependent.

Comparing a 0.14M-token novel corpus with a 17M-token year of social media would confound sample size with language. So every corpus is analysed at the same token count $N=142{,}614$ (`meta.N`), the full unfiltered Austen sample. Each corpus gets 5 independent draws without replacement (multivariate hypergeometric, `subsample`). The seeds are `SEED + 1000·i + rep`, with `SEED = 20071231` (`meta.seed`, a fixed arbitrary seed). Corpora with fewer than $N$ tokens are excluded; this removes Twitter 2006–2009 (§5).

The replication (§4.1) instead uses S&B's own protocol: dictionary filter, full sample, no subsampling.

### 3.7 Held-out evaluation: nested cross-validation (ours; S&B report in-sample only)
`nested_cv_fold` and `summarise_nested` work as follows:
1. Tokens, not types, are split into 5 outer folds (`WordCounts.split`, multinomial per type). Tokens are split because the quantity being estimated is the code length of the next token.
2. For each outer fold, $\sigma^2$ is chosen by 4-fold inner CV on the outer-training tokens only (`select_prior`).
3. The MAP model is refit on all outer-training tokens and scored in bits per token on the untouched outer-test tokens.
4. The reported value is the mean over outer folds, with SE = sd/√5.

Two conventions:
- The $\sigma^2$ grid starts at $\{10^{-1},10^{-1/2},1,10^{1/2},10\}$ (`meta.sigma2_start`): one decade either side of unit coupling variance, the natural scale of a log-probability. Whenever an endpoint wins, the grid is extended by half-decades (at most 8 extensions, `SIGMA2_MAX_EXTENSIONS`), so the reported optimum is always bracketed.
- Both orders 1 and 2 go through the l2 path, so the comparison is like for like. Nested CV removes the optimistic bias of reporting the CV loss of the selected model ([Varma & Simon 2006]: the standard demonstration of that bias and of nested CV as the remedy).

### 3.8 Landscape definitions
- **Strict minimum** (`local_minima`). Finite energy, and the unique lowest point on each of the $L$ one-dimensional fibres through the word. Equivalently, all $L(A-1)=100$ single-substitution neighbours are strictly higher. This is S&B's definition. Forbidden strings are never minima.
- **Steepest descent** (`descend`, `descent_path`). From each string, move to the lowest of its 100 neighbours if that is strictly lower. Ties go to the earliest position, then the earliest letter. Forbidden strings relax too, because any finite neighbour is lower.
- **Terminal-state caveat.** Every strict minimum is a terminal state, but descent can also stop on a plateau of tied energies. Wells are therefore always counted with `local_minima`, never as terminal states (`tests/test_maxent.py::test_plateau_terminals_are_not_reported_as_wells`).
- **`substitution_recovery`.** For each observed type $w$: the fraction of its 100 single-letter misspellings whose steepest descent ends exactly on $w$. A word that is not itself a terminal state scores 0. The corpus value `recovery` $=\sum_w\hat p(w)\,r(w)$ is token-weighted decoding accuracy. It turns S&B's error-correction remark into a measurement.
- **Other landscape fields** (`analyse`):
  - `minima_model_mass`: $P_2$-mass on minima.
  - `minima_token_share`: empirical token share of words that are minima.
  - `unseen_model_mass`: $P_2$-mass on strings absent from the sample.
  - `unseen_dict_share`: dictionary fraction of that unseen mass.
  - `n_minima_dict`, `n_minima_seen`.
  - The residual $\log_2(\hat p/P_2)$ per observed word is ranked by the edge, nearest zero, of the exact 95% Poisson interval of its count ([Garwood 1936], the exact chi-square-based Poisson limits). A word therefore ranks as mis-predicted only if the residual survives counting noise (`under_predicted`, `over_predicted`).

### 3.9 Audit findings that were fixed
There is no audit report in the repository and no VCS history (the directory is not a git repository). This list is reconstructed from the docstrings and tests that encode each fix.
1. **Entropy stopping criterion.** Matching marginals to tolerance does not control the entropy error, because that error scales like $|g\cdot\theta|$ and $|\theta|$ can be large. Criterion (b) of §3.3 was added (`fit` docstring; `ent_se`). The code docstring calls the result "bounded"; that is accurate for the true-gap inequality but not for its Newton-decrement estimate (§3.3).
2. **Recovery versus basin agreement.** Counting a misspelling as "recovered" when it relaxes into the same basin as the intended word over-scores. With counts {aaaa: 3, aaab: 1}, every misspelling of either word relaxes to aaaa: basin agreement gives 1, decoding accuracy gives 3/4. `substitution_recovery` now requires the exact intended word (`test_recovery_counts_only_the_intended_word`).
3. **CV selection bias.** Reporting the inner-CV loss of the chosen $\sigma^2$ is optimistic. This was replaced by nested CV (§3.7).
4. **Boundary-selected prior.** A $\sigma^2$ at the grid edge is not an optimum. The grid now extends until the optimum is interior (`select_prior`; `test_prior_selection_is_interior_and_extends_the_grid_when_needed`). All CV runs report `all_interior = True` (`cv.*.*.all_interior`).
5. **Zero-variance tables.** Two degenerate cases make an SE vanish: uniform counts give ${\rm Var}[\ln\hat p]=0$, and a letter that never varies gives $\hat\mu(1-\hat\mu)=0$. The stopping rule then becomes unsatisfiable. Both SEs are now floored, at $1/N$ and $(K-1)/(2N)$ (`test_degenerate_count_tables_converge_without_crashing`).

---

## 4. Results

### 4.1 Replication on Austen with S&B's protocol (`replication.*`)

| quantity | S&B | ours | field |
|---|---|---|---|
| four-letter tokens / types | 135,441 / 763 | 141,198 / 820 | `tokens`, `types` |
| $S_{\rm rand}$ | 18.802 | 18.8018 | `S_rand` |
| $S_1$ | 14.083 ± 0.001 | 14.3230 | `S_ind` |
| $S_2$ | 7.48 | 7.5495 | `S_2` |
| $S_3$ | — | 6.98192 | `S_3` |
| $S_{\rm full}$ | 6.918 ± 0.003 | 6.98190 | `S_full` |
| $I_N=S_1-S_{\rm full}$ | 7.163 | 7.341 | derived |
| $2^{S_1-S_2}$ | ~100 | 109.4 | derived |
| captured $I^{(2)}/I_N$ | 92% | 92.27% | `captured` |
| model mass on unseen strings | 15% | 15.07% | `unseen_model_mass` |
| dictionary share of unseen mass | 1/5 | 51.6% | `unseen_dict_share` |
| strict minima / in dictionary / observed | ANC: 136 / 118 / — | 149 / 138 / 134 | `n_minima`, `n_minima_dict`, `n_minima_seen` |
| model mass on minima | ANC: 63.5% | 62.19% | `minima_model_mass` |
| token share of minima | — | 68.45% | `minima_token_share` |
| exact-word substitution recovery, all observed tokens | not reported by S&B | 0.670 | `recovery` |

How to read the table:
- **Captured share and unseen mass agree within 0.3 percentage points.** Our minima mass is 62.19%, 1.31 points below S&B's 63.5%. S&B's figure is for ANC, so this is an indicative cross-corpus comparison, not a direct replication.
- **Absolute entropies are about 0.06–0.24 bits higher.** We have 57 more types and 5,757 more tokens; position-specific letter entropies, and hence $S_1$, grow with additional rare types. [INFERENCE: this tokenisation and dictionary difference is the likely cause.]
- **The dictionary share of unseen mass disagrees: 52% against 1/5.** It depends on the dictionary. web2 lists obscure entries (`sime`, `hame`, `sade`, `mide` are all flagged as dictionary words), which 12dicts, built from learner's and desk dictionaries, mostly omits [INFERENCE].
- **S&B's minima figures are for ANC**, so the minima comparison is only indicative. The two studies also search for minima differently. S&B find them by steepest descent from each observed corpus word (PRE 2010, Fig. 4 caption), so their set contains only minima reachable from observed words. `local_minima` instead tests all $26^4$ states, which includes minima in other basins. Their reported minima set is not shown to be exhaustive.
- **Recovery is an extension, not a replication.** S&B's qualitative claim that any single-letter error "can always be corrected" concerns only stable words, i.e. local minima (preprint p. 4; PRE 2010 p. 3). Our 0.670 averages over every observed token and scores non-terminal words as 0, so it does not test that claim like for like.

**Invented words against S&B Fig. 3.** Source: `replication.invented`, the top 25 unseen strings by $P_2$, with probability and dictionary flag (✓ = in web2):

mast 3.95e-3 ✓, tome ✓, sime ✓, hame ✓, sade ✓, tike, whem, mike ✓, mide ✓, welt ✓, liss ✓, hove ✓, lave ✓, bady, wert ✓, sere ✓, mady, dore, thom, mone ✓, core ✓, gove ✓, **mord**, whan ✓, fent 1.06e-3 ✓.

- 24 of S&B's 25 inset words are in our top 25. The exception is `dees`, which is in our top 40; the one newcomer is `mord`.
- The first five agree in order: mast, tome, sime, hame, sade.
- 18 of our 25 are web2 words.

**$S_3=S_{\rm full}$ saturation.**
- The gap is $S_3-S_{\rm full}=2.0\times10^{-5}$ bits (`S_3 − S_full`). For comparison, $S_2-S_3=0.568$ bits, and the plug-in bias scale $(K-1)/(2N\ln2)$ is 0.0042 bits for $K=820$, $N=141{,}198$.
- With 820 observed types, the four triplet marginals ($4\times26^3$ cells) leave almost no freedom beyond the empirical distribution itself, so the triplet model reproduces the sample.
- This is a statement about sample size, that triplets saturate at $N\approx1.4\times10^5$. It is **not** evidence that English has no four-body structure. $S_3$ is not reported for the other corpora.

### 4.2 All 17 corpora at matched N = 142,614 (5 subsamples each; `corpora[*]`)

Entropies are in bits; "captured" is $(S_1-S_2)/(S_1-S_{\rm full})$ per subsample, then averaged. Fields: `types`, `S1`, `S2`, `S_full`, `n_minima`, `n_minima_dict`, `minima_model_mass`, `minima_token_share`, `unseen_model_mass`, `recovery`.

| corpus | types in subsample | S_1 | S_2 | S_full | captured (%) | minima | dict. minima | minima model mass | minima token share | unseen model mass | recovery |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Austen | 966 | 14.385 ± 0.000 | 7.651 ± 0.000 | 7.045 ± 0.000 | 91.74 ± 0.00 | 153.0 ± 0.0 | 121.0 ± 0.0 | 0.604 ± 0.000 | 0.662 ± 0.000 | 0.160 ± 0.000 | 0.649 ± 0.000 |
| Gutenberg | 1828 | 14.472 ± 0.003 | 8.609 ± 0.007 | 7.690 ± 0.006 | 86.45 ± 0.05 | 137.4 ± 5.6 | 103.0 ± 2.3 | 0.466 ± 0.001 | 0.535 ± 0.002 | 0.221 ± 0.004 | 0.524 ± 0.002 |
| Bluesky 2024 | 3850 | 14.855 ± 0.008 | 9.266 ± 0.011 | 8.039 ± 0.006 | 81.99 ± 0.08 | 158.4 ± 3.9 | 95.6 ± 1.9 | 0.462 ± 0.002 | 0.574 ± 0.003 | 0.271 ± 0.004 | 0.562 ± 0.003 |
| Twitter 2010 | 3969 | 15.259 ± 0.002 | 9.838 ± 0.009 | 8.411 ± 0.005 | 79.15 ± 0.07 | 152.8 ± 3.1 | 90.0 ± 2.2 | 0.400 ± 0.003 | 0.512 ± 0.004 | 0.319 ± 0.002 | 0.501 ± 0.003 |
| Twitter 2011 | 4246 | 15.202 ± 0.004 | 9.876 ± 0.006 | 8.443 ± 0.004 | 78.81 ± 0.05 | 142.4 ± 3.6 | 86.4 ± 1.8 | 0.398 ± 0.001 | 0.519 ± 0.003 | 0.315 ± 0.004 | 0.507 ± 0.002 |
| Twitter 2012 | 4437 | 15.188 ± 0.005 | 9.852 ± 0.006 | 8.432 ± 0.004 | 78.99 ± 0.07 | 144.0 ± 3.2 | 87.8 ± 1.8 | 0.404 ± 0.001 | 0.528 ± 0.001 | 0.311 ± 0.003 | 0.515 ± 0.001 |
| Twitter 2013 | 4635 | 15.145 ± 0.009 | 9.769 ± 0.008 | 8.389 ± 0.005 | 79.58 ± 0.09 | 143.4 ± 1.3 | 89.8 ± 0.8 | 0.408 ± 0.001 | 0.529 ± 0.001 | 0.304 ± 0.004 | 0.517 ± 0.001 |
| Twitter 2014 | 4341 | 15.109 ± 0.003 | 9.669 ± 0.006 | 8.314 ± 0.003 | 80.06 ± 0.09 | 141.4 ± 5.2 | 90.6 ± 2.7 | 0.418 ± 0.002 | 0.538 ± 0.004 | 0.300 ± 0.003 | 0.526 ± 0.003 |
| Twitter 2015 | 4301 | 15.105 ± 0.001 | 9.614 ± 0.008 | 8.273 ± 0.006 | 80.37 ± 0.08 | 145.6 ± 4.8 | 87.4 ± 1.5 | 0.425 ± 0.001 | 0.544 ± 0.003 | 0.300 ± 0.005 | 0.533 ± 0.003 |
| Twitter 2016 | 4319 | 15.082 ± 0.012 | 9.582 ± 0.007 | 8.249 ± 0.007 | 80.49 ± 0.04 | 141.6 ± 4.2 | 91.0 ± 2.9 | 0.429 ± 0.002 | 0.548 ± 0.004 | 0.295 ± 0.004 | 0.536 ± 0.003 |
| Twitter 2017 | 4329 | 15.069 ± 0.004 | 9.546 ± 0.006 | 8.223 ± 0.004 | 80.68 ± 0.05 | 146.4 ± 2.7 | 91.6 ± 0.9 | 0.435 ± 0.001 | 0.554 ± 0.001 | 0.296 ± 0.006 | 0.542 ± 0.001 |
| Twitter 2018 | 4170 | 15.009 ± 0.009 | 9.420 ± 0.009 | 8.130 ± 0.008 | 81.25 ± 0.08 | 143.6 ± 5.5 | 92.2 ± 2.3 | 0.447 ± 0.002 | 0.560 ± 0.004 | 0.285 ± 0.002 | 0.548 ± 0.003 |
| Twitter 2019 | 4203 | 15.000 ± 0.013 | 9.385 ± 0.013 | 8.112 ± 0.009 | 81.52 ± 0.06 | 141.4 ± 2.3 | 91.6 ± 0.9 | 0.453 ± 0.001 | 0.565 ± 0.002 | 0.281 ± 0.002 | 0.553 ± 0.002 |
| Twitter 2020 | 4062 | 14.946 ± 0.006 | 9.264 ± 0.008 | 8.028 ± 0.005 | 82.14 ± 0.08 | 147.2 ± 7.2 | 90.2 ± 2.5 | 0.463 ± 0.002 | 0.571 ± 0.002 | 0.272 ± 0.005 | 0.559 ± 0.002 |
| Twitter 2021 | 4244 | 14.951 ± 0.006 | 9.314 ± 0.006 | 8.053 ± 0.003 | 81.72 ± 0.06 | 140.8 ± 4.8 | 89.2 ± 2.6 | 0.457 ± 0.001 | 0.564 ± 0.001 | 0.281 ± 0.003 | 0.553 ± 0.001 |
| Twitter 2022 | 4270 | 14.950 ± 0.007 | 9.382 ± 0.008 | 8.098 ± 0.006 | 81.26 ± 0.04 | 146.6 ± 5.2 | 89.2 ± 2.9 | 0.448 ± 0.001 | 0.558 ± 0.001 | 0.281 ± 0.003 | 0.547 ± 0.001 |
| Twitter 2023 | 4307 | 14.939 ± 0.006 | 9.437 ± 0.008 | 8.135 ± 0.003 | 80.85 ± 0.09 | 137.0 ± 7.5 | 88.4 ± 3.9 | 0.440 ± 0.002 | 0.551 ± 0.003 | 0.286 ± 0.004 | 0.540 ± 0.003 |

Supplementary fields (`n_minima_seen`, `unseen_dict_share`, `S_full_bias`, `fit_marginal_in_se`, `fit_entropy_error_in_se`; the last two are maxima over subsamples):

| corpus | seen minima | unseen-mass dictionary share | S_full bias scale (K−1)/(2N ln2), bits | max marginal error / SE | max entropy error / SE |
|---|---|---|---|---|---|
| Austen | 133.0 ± 0.0 | 0.502 ± 0.000 | 0.0049 | 0.0085 | 0.00267 |
| Gutenberg | 111.0 ± 1.6 | 0.354 ± 0.012 | 0.0092 | 0.0080 | 0.00059 |
| Bluesky 2024 | 113.6 ± 4.6 | 0.304 ± 0.015 | 0.0195 | 0.0092 | 0.00011 |
| Twitter 2010 | 112.4 ± 2.1 | 0.297 ± 0.003 | 0.0201 | 0.0070 | 0.00012 |
| Twitter 2011 | 104.2 ± 4.2 | 0.289 ± 0.005 | 0.0215 | 0.0076 | 0.00008 |
| Twitter 2012 | 105.0 ± 1.9 | 0.298 ± 0.008 | 0.0224 | 0.0090 | 0.00010 |
| Twitter 2013 | 106.2 ± 0.8 | 0.303 ± 0.009 | 0.0234 | 0.0056 | 0.00004 |
| Twitter 2014 | 106.2 ± 4.5 | 0.301 ± 0.005 | 0.0220 | 0.0093 | 0.00006 |
| Twitter 2015 | 106.2 ± 2.8 | 0.308 ± 0.012 | 0.0217 | 0.0046 | 0.00002 |
| Twitter 2016 | 106.8 ± 3.8 | 0.306 ± 0.015 | 0.0218 | 0.0077 | 0.00008 |
| Twitter 2017 | 107.4 ± 1.5 | 0.306 ± 0.007 | 0.0219 | 0.0086 | 0.00005 |
| Twitter 2018 | 104.6 ± 3.6 | 0.308 ± 0.011 | 0.0211 | 0.0075 | 0.00005 |
| Twitter 2019 | 104.4 ± 2.7 | 0.304 ± 0.009 | 0.0213 | 0.0035 | 0.00007 |
| Twitter 2020 | 104.6 ± 3.0 | 0.310 ± 0.006 | 0.0205 | 0.0090 | 0.00018 |
| Twitter 2021 | 100.6 ± 1.1 | 0.314 ± 0.007 | 0.0215 | 0.0062 | 0.00003 |
| Twitter 2022 | 101.0 ± 4.6 | 0.298 ± 0.012 | 0.0216 | 0.0053 | 0.00004 |
| Twitter 2023 | 100.8 ± 3.8 | 0.293 ± 0.005 | 0.0218 | 0.0079 | 0.00003 |

Austen's sd is zero by construction, not a precision claim (§5.2). Differences around $10^{-6}$ in its minima mass between "subsamples" reflect solver tolerance on identical data [INFERENCE].

Matched-N Austen keeps non-dictionary tokens, with 966 types against 820 in the replication, so its captured share (91.7%) is not directly comparable to the replication's 92.3%.

**Edited prose against short-form text.** At equal $N$:
- **Captured share:** about 92% for Austen and 86% for Gutenberg, against 79–82% for Twitter and Bluesky.
- **Wells:** the minima carry 60% of model mass in Austen against 40–46% in social media.
- **Unseen mass:** 16% in Austen against 27–32% in social media.
- **Recovery:** 0.65 in Austen against 0.50–0.56 in social media.
- **Dictionary minima:** about 121 in Austen against 86–96 in social media, although the total number of minima is similar (137–158).

### 4.3 Twitter year trend (rows `Twitter 2010`…`Twitter 2023` above)

All three entropies show substantial net decreases from 2010 to 2020, although $S_2$ and $S_{\rm full}$ initially rise in 2011 (9.838→9.876 and 8.411→8.443 bits). Some annual changes are comparable to the subsample spread: $S_1$ moves by 0.003 bits from 2014 to 2015 against a 2014 sd of 0.003, and by 0.013 from 2016 to 2017 against a 2016 sd of 0.012.

| quantity | 2010 | 2020 | 2023 |
|---|---|---|---|
| $S_1$ (bits) | 15.259 | 14.946 | 14.939 |
| $S_2$ (bits) | 9.838 | 9.264 | 9.437 |
| $S_{\rm full}$ (bits) | 8.411 | 8.028 | 8.135 |
| captured share | 79.15% | 82.14% | 80.85% |
| minima model mass | 0.400 | 0.463 | 0.440 |
| recovery | 0.501 | 0.559 | 0.540 |

- The captured share dips to its minimum, 78.81%, in 2011 before rising to the 2020 peak.
- Bluesky 2024 (82.0%) sits at the Twitter-2020 level.
- Read §5.1 before interpreting this as linguistic change: the composition of the sample by year is strongly non-uniform.

### 4.4 Nested-CV held-out code lengths (`cv.<corpus>.<order>`; corpora chosen as first, middle and last Twitter year plus the three references)

| corpus | $H_1$ held-out (bits/token) | $H_2$ held-out | $\sigma^2$ selected, order 2, per outer fold | held-out gain $H_1-H_2$ (paired over folds) | in-sample $S_1-S_2$ (mean of 5 subsamples) | gain / in-sample |
|---|---|---|---|---|---|---|
| Austen | 14.3861 ± 0.0082 | 7.6775 ± 0.0089 | 100 ×5 | 6.709 ± 0.007 | 6.734 | 99.6% |
| Gutenberg | 14.4750 ± 0.0036 | 8.6294 ± 0.0102 | 31.6 ×5 | 5.846 ± 0.011 | 5.864 | 99.7% |
| Twitter 2010 | 15.2599 ± 0.0114 | 9.8778 ± 0.0060 | 10 ×5 | 5.382 ± 0.009 | 5.420 | 99.3% |
| Twitter 2017 | 15.0704 ± 0.0079 | 9.5944 ± 0.0099 | 3.16 ×5 | 5.476 ± 0.009 | 5.522 | 99.2% |
| Twitter 2023 | 14.9438 ± 0.0050 | 9.4809 ± 0.0086 | 3.16 ×5 | 5.463 ± 0.008 | 5.501 | 99.3% |
| Bluesky 2024 | 14.8472 ± 0.0033 | 9.3122 ± 0.0078 | 10, 3.16, 3.16, 10, 10 | 5.535 ± 0.009 | 5.589 | 99.0% |

Sources: `cv.*.{1,2}.bits`, `.se`, `.sigma2`, `.per_fold` (gain = mean ± SE of the per-fold differences), and `corpora[*].S1/S2`. All runs report `all_interior = True` and `all_converged = True`; the log shows the same values.

- **The pairwise structure is real.** 99.0–99.7% of the in-sample pairwise information survives on held-out tokens, so almost none of $S_1-S_2$ is fitted sampling noise at this $N$.
- **Order-1 prior widths.** The selected $\sigma^2$ for order 1 are wide (Austen up to 31,623, `cv.Austen.1.sigma2`), i.e. nearly unregularised, as expected for 100 parameters. All are interior.
- **Edited prose tolerates weaker regularisation.** The order-2 prior width falls from 100 (Austen) to 3.16 (Twitter 2017/2023).

### 4.5 Energy–frequency agreement (fig. 3)
The comparison is between $\log_{10}\hat p$ and $\log_{10}P_{\rm model}$ over the observed types of subsample 0. Source: `corpora[*, rep 0].scatter.{p_emp, p1, p2}`. $r^2$ is computed by `figures/make_figures.py:fig3` as the squared Pearson correlation over **all observed types**. The figure's label "binned mean, r^2" is misleading: the 20-bin means are only drawn, and $r^2$ is not computed from them.

| corpus (rep 0) | observed types | $r^2$ independent | $r^2$ pairwise |
|---|---|---|---|
| Austen | 966 | 0.04 | 0.36 |
| Twitter 2023 | 4,303 | 0.11 | 0.28 |
| Bluesky 2024 | 3,872 | 0.11 (0.105) | 0.30 |
| (Gutenberg, not plotted) | 1,808 | 0.055 | 0.339 |
| (Twitter 2010, not plotted) | 3,961 | 0.092 | 0.270 |

- Pairwise couplings raise the explained variance of log frequency by a factor of about 3–9.
- Most per-word variance stays unexplained. This matches S&B's "agreement only approximately … substantial scatter".

### 4.6 Where the pairwise model fails: acronyms, and the `tmrw` diagnosis
**Austen.** The most under-predicted words (`corpora[Austen, rep 0].under_predicted`, ranked by the conservative edge of the Poisson interval) are rare dictionary words with unusual letter combinations. Examples: `dela`, `slit`, `slip`, `bias` (8 tokens, residual 47.9 bits), `gala`, `elms`, `airy`.

**Social media.** In Twitter 2023 and Bluesky the list is dominated by acronyms and consonant strings:
- Twitter 2023 (`corpora[Twitter 2023, rep 0].under_predicted`): `tmrw` (4 tokens, residual 34.1 bits, lower edge 32.2), `nwsl`, `tldr`, `lvmh`, `html`, `grrr`, `rpdr`, …
- Bluesky: `nwsl`, `jrnl`, `tbnw`, `lsoh`, `tmrw` (7 tokens, 30.1 bits), `dwtn`, `html`, `ptsd`, `ttyl`, `xkcd`, `gdpr`, `blvd`, …

These strings are strongly under-predicted by the pairwise model, which indicates structure beyond pairwise compatibility. The fits do not distinguish three-body from four-body contributions: only orders 1 and 2 were fitted for these samples (`job_corpus`), so the split of the missing information into $S_2-S_3$ and $S_3-S_{\rm full}$ (§1.4) is not available.

**`tmrw` in Twitter 2023, subsample 0.**
- $E=49.24$ bits, so $P_2(\texttt{tmrw})=2^{-49.24}=1.5\times10^{-15}$ (`probes.tmrw.energy_bits`).
- $\hat p=4/142{,}614=2.8\times10^{-5}$ (`probes.tmrw.count = 4`).

The six pair cells of `tmrw` in this subsample were obtained by redrawing the subsample with the study's seed rule. That is sampling only, with no fit; the redrawn subsample reproduces `types = 4303`.

| cell (positions) | tokens in cell | strings carrying it in the data |
|---|---|---|
| t·m (1,2) | 7 | tmrw 4, tmac, tmnt, tmux |
| t·r (1,3) | 362 | term 134, turn 129, thru 24, tory 23, … |
| t·w (1,4) | 6 | tmrw 4, teow, thaw |
| m·r (2,3) | 8 | tmrw 4, hmrc 2, smrt, vmro |
| m·w (2,4) | 4 | tmrw only |
| r·w (3,4) | 4 | tmrw only |

- Five of the six cells are rare. Two, m·w and r·w, are carried by `tmrw` alone.
- The exact fit matches each cell marginal to within 1% of its SE. Yet it gives the conjunction of all five rare cells almost no weight, because it places each rare cell's mass on strings that combine that cell with common letters elsewhere.
- The earlier diagnostic named the model's main carriers: tm→`tmam`, tw→`thew`, mr→`amro`, mw→`vmiw`, rw→`virw`. **These carrier strings are not stored in the JSON**; reproducing them needs a refit of this subsample, which was outside this document's read-only remit.
- The mechanism is the general one: matching marginals constrains sums over strings, not which string carries them. An acronym, whose only shared property is the conjunction itself, is the extreme case. The rank of `tmrw` stays above 95,000 in every year (§4.7).

### 4.7 Probe trajectories (fig. 4; `corpora[*].probes.<form>.{count, rank, is_min, energy_bits, target}`)
The probe list is fixed before any fit (`meta.probes`): 20 slang forms, 8 controls and 8 transposition typos. "rank" is the energy rank among all 456,976 strings. For forbidden strings ($E=+\infty$) the stable sort only reflects index order, so those ranks carry no meaning. The entries affected: Austen and Gutenberg `lmao` and `tmrw`, Austen `thru`, 2 of 5 Gutenberg `thru` draws, and 1 of 5 Twitter 2023 `tmrw` draws (rank 364,833).

| corpus | lmao count (mean) | lmao rank (median) | lmao wells /5 | thru count | thru rank | thru wells /5 | tmrw count | tmrw rank |
|---|---|---|---|---|---|---|---|---|
| Austen | 0.0 | forbidden | 0 | 0.0 | forbidden | 0 | 0.0 | forbidden |
| Gutenberg | 0.0 | forbidden | 0 | 0.0 | 17,623 | 0 | 0.0 | forbidden |
| Bluesky 2024 | 175.4 | 213 | 5 | 23.6 | 2,094 | 0 | 3.4 | 111,955 |
| Twitter 2010 | 29.6 | 43,175 | 0 | 81.0 | 532 | 5 | 7.8 | 122,422 |
| Twitter 2011 | 89.4 | 2,311 | 0 | 73.8 | 482 | 5 | 10.0 | 117,616 |
| Twitter 2012 | 32.4 | 23,260 | 0 | 60.8 | 750 | 5 | 8.2 | 120,631 |
| Twitter 2013 | 22.4 | 35,216 | 0 | 56.0 | 811 | 5 | 7.8 | 132,892 |
| Twitter 2014 | 18.6 | 52,721 | 0 | 51.8 | 1,046 | 4 | 11.4 | 116,391 |
| Twitter 2015 | 24.8 | 27,449 | 0 | 47.2 | 918 | 5 | 8.6 | 125,148 |
| Twitter 2016 | 25.0 | 30,639 | 0 | 45.4 | 1,063 | 5 | 15.0 | 95,665 |
| Twitter 2017 | 21.6 | 41,462 | 0 | 50.0 | 709 | 5 | 11.8 | 97,413 |
| Twitter 2018 | 25.4 | 22,766 | 0 | 29.4 | 1,759 | 4 | 4.2 | 131,870 |
| Twitter 2019 | 23.6 | 43,332 | 0 | 30.8 | 1,510 | 5 | 6.8 | 126,857 |
| Twitter 2020 | 28.4 | 23,676 | 0 | 36.2 | 1,234 | 5 | 5.2 | 124,672 |
| Twitter 2021 | 39.8 | 8,101 | 0 | 25.8 | 1,605 | 4 | 5.6 | 137,485 |
| Twitter 2022 | 46.4 | 8,168 | 0 | 22.8 | 2,136 | 2 | 4.6 | 136,108 |
| Twitter 2023 | 68.4 | 2,088 | 0 | 22.6 | 1,424 | 2 | 2.2 | 143,582 |

- **`thru` is a strict well** in at least 4 of 5 subsamples every Twitter year 2010–2021 (5/5 in 2010–13, 2015–17, 2019–20). It drops to 2/5 in 2022–2023 and 0/5 in Bluesky, where it descends to `they`. Its usage falls from 81 to about 23 tokens per $N$ over the same period.
- **`lmao`.**
  - Median rank 43,175 in 2010. It is never a Twitter well. Its descent targets across all 70 Twitter draws are `best` (36), `read` (17), `lman` (14), `lost` (2010 rep 2), `xmas` (2015 rep 0) and `lmam` (2015 rep 3).
  - It climbs to about 2,100 by 2023 (median 2,088). There is a one-year spike to 2,311 in 2011, when its usage was also highest (89.4 tokens; a composition effect is likely, §5.1).
  - It reaches rank 213 and a strict minimum in 5/5 Bluesky 2024 subsamples, at 175 tokens per $N$. In Austen and Gutenberg it is forbidden.
- **No other slang probe** (rofl, omfg, imho, smth, tmrw, idek, bruh, yeet, yolo, fomo, smol, yall, gurl, noob, stan, simp, vibe, meme) is a well in any subsample.
- **Controls.** `that`, `with`, `have`, `time` and `good` are wells in every corpus. `love` is a well only in Twitter 2011–2019 and Bluesky. `cool` is a well in Twitter 2010–2019, but only in 2 of 5 subsamples by 2017–2019.

### 4.8 Descent paths of typos: transpositions are not corrected
The paths below are from Twitter 2023, rep 0 (`corpora[Twitter 2023, rep 0].probes.<typo>.path`, energies in bits; `null` = forbidden):

| typo | path | ends at |
|---|---|---|
| waht | waht 25.82 → want 9.16 | want |
| taht | taht 32.57 → tast 13.68 → past 8.35 | past |
| konw | konw (forbidden) → kong 12.88 → long 8.75 | long |
| liek | liek 14.74 → lick 10.80 | lick |
| jsut | jsut 32.18 → jout 17.93 → jost 9.69 → just 6.26 | **just** |
| ahve | ahve (forbidden) → abve 15.27 → able 10.20 | able |
| wiht | wiht (forbidden) → wint 11.07 → want 9.16 | want |
| thsi | thsi 25.65 → thai 13.30 → that 4.42 | that |

- A transposition is two substitutions apart from the intended word. Greedy single-substitution descent from it takes the lowest-energy neighbour at each step and ends at the terminal state of that basin. This need not be the nearest well, and it is rarely the intended word.
- `jsut`→`just` is the only success. It goes through a three-step detour (position 2 s→o, position 3 u→s, position 2 o→u). Every displayed rep-0 Twitter path ends at `just`. Across all Twitter draws, 65 of 70 end at `just`, four at `that` and one at `used`; all 5 Bluesky draws end at `just` (`corpora[*].probes.jsut.target`).
- In Austen the same typos end at `will`, `talk`, `life` and `back`, or stay put (`jsut`, `ahve`, `thsi`, `konw` are forbidden with no finite neighbour path).
- Austen's `waht` path (rep 0) is `waht → want 11.86 → went 10.25 → welt 9.32 → well 6.44 → will 6.06` bits. Steepest descent follows a sequence of decreasing-energy states to its basin's terminal state. In this example only the endpoint, `will`, is a well: each intermediate word has a lower-energy neighbour.

### 4.9 Non-dictionary wells (`corpora[*, rep 0].nondict_minima`: string, $P_2$, observed?)
"Non-dictionary" means absent from web2. That set includes ordinary inflected forms (`eyes`, `ones`, `says`) and modern vocabulary.
- **Austen, matched N:** eyes, **mury**, shew, **fuce**, lyme, **doad**, ones, **safa**, mama, owed, **enve**, **ains**, xvii, **veis**, cobb. Bold strings were never observed; they are invented wells.
- **Replication (dictionary-filtered Austen;** `replication.nondict_minima`**, all unobserved):** mury, fuce, shew, doad, safa, pign, liou, knig, ores, owan, atam.
- **Gutenberg:** eyes, says, shew, mury, ahab, eass, blew, badn, kips, ones, …
- **Twitter 2010:** blog, ipad, haha, plue, etsy, pigs, expo, nzch, thru, yssy, juzz, apps, yosh, bazz, hsvp.
- **Twitter 2017:** blog, guys, tips, hald, pigs, plue, haha, vity, endy, okay, thru, eggs, apps, expo, anny.
- **Twitter 2023:** says, hald, vity, plue, blug, okay, arme, haha, aris, pigs, badn, nfts, neto, eggs, apps.
- **Bluesky 2024:** fuck, okay, hald, lmao, eyes, bets, haha, bsky, badn, eggs, gaza, neto, edis, hehh, adha.

Two kinds of entry appear:
- observed platform vocabulary (`blog`, `ipad`, `etsy`, `nfts`, `bsky`, `gaza`, `lmao`);
- unobserved, pronounceable inventions that recur across years (`hald`, `plue`, `endy`, `juzz`, `vity`, `badn`).

This is S&B's "discovery" effect, carried into the social-media landscape.

### 4.10 The four figures (`figures/`, drawn by `figures/make_figures.py` from the JSON only)
- **`fig1_entropy_hierarchy.svg`.** *How much of the structure of four-letter words is pairwise, and does that differ between edited prose and social media?*
  - Left panel: $S_1$, $S_2$ and $S_{\rm full}$ with % captured for S&B (published), Austen, Gutenberg, Twitter 2010 / 2017 / 2023 and Bluesky 2024. The labels are 92, 92, 86, 79, 81, 81, 82%.
  - Right panel: the captured share by Twitter year (mean ± sd over subsamples), with Austen, Gutenberg and Bluesky as bands.
- **`fig2_generalisation.svg`.** *Is the pairwise structure real?*
  - Left panel: held-out code lengths $H_1$, $H_2$ from nested CV.
  - Right panel: held-out gain against in-sample $S_1-S_2$, with the identity line. The labels are 100% for Austen and Gutenberg and 99% elsewhere (§4.4).
- **`fig3_energy_vs_frequency.svg`.** *Does the pairwise energy predict how often each word is used?* Scatter of $\log_{10}P_2$ against $\log_{10}\hat p$ for Austen, Twitter 2023 and Bluesky 2024 (rep 0). It shows 600 random types, 20 equal-width log-frequency bin means (as in S&B Fig. 2), independent-model bin means, and the $r^2$ values of §4.5.
- **`fig4_landscape_by_year.svg`.** *How does the landscape move over time?* Probes × (Twitter 2010–2023, Bluesky).
  - Left panel: usage in log10 tokens per million.
  - Right panel: energy-rank depth $1-\log_{10}{\rm rank}/\log_{10}26^4$, with a dot sized by the fraction of subsamples in which the form is a strict well.
  - The slang forms that become wells are `lmao` (Bluesky) and `thru` (Twitter).

---

## 5. Limitations

### 5.1 Twitter sample composition
`enryu43/twitter100m_tweets` consists of user timelines, each truncated at roughly the account's 1,001 most recent tweets as of the early-2023 scrape (`langfit/social.py` docstring). Early years are therefore reachable only through low-volume accounts, and the composition of the sample by year is not that of the platform. Only 16 of 41 shards were used (`meta.corpora.Twitter.source`).

Effective users are the inverse Simpson index of per-user four-letter token shares ([Simpson 1949], the concentration index, inverted to give an equivalent number of equally active users). Source: `data/derived/twitter_L4_by_year.json` → `years.<y>.effective_users`.

| year | posts | English posts | 4-letter tokens | users | effective users | max user share | analysed |
|---|---|---|---|---|---|---|---|
| 2006 | 867 | 844 | 1,752 | 2 | 1.2 | 0.9058 | no (< N) |
| 2007 | 1,154 | 995 | 1,818 | 20 | 6.5 | 0.2442 | no (< N) |
| 2008 | 4,436 | 4,021 | 9,061 | 116 | 16.0 | 0.1961 | no (< N) |
| 2009 | 55,160 | 46,131 | 110,914 | 940 | 187.9 | 0.0217 | no (< N) |
| 2010 | 117,522 | 82,750 | 192,286 | 1,639 | 362.6 | 0.0155 | yes |
| 2011 | 245,264 | 171,595 | 392,621 | 2,897 | 654.9 | 0.0066 | yes |
| 2012 | 431,979 | 299,849 | 640,014 | 4,249 | 1,139.6 | 0.0055 | yes |
| 2013 | 625,175 | 431,537 | 904,126 | 5,706 | 1,652.6 | 0.0048 | yes |
| 2014 | 806,320 | 570,839 | 1,181,933 | 7,368 | 2,162.2 | 0.0029 | yes |
| 2015 | 1,064,821 | 751,794 | 1,510,733 | 9,188 | 2,717.2 | 0.0022 | yes |
| 2016 | 1,286,391 | 908,231 | 1,816,565 | 11,025 | 3,179.3 | 0.0016 | yes |
| 2017 | 1,508,366 | 1,043,957 | 2,224,121 | 12,945 | 3,773.1 | 0.0013 | yes |
| 2018 | 1,695,605 | 1,146,637 | 3,318,137 | 14,732 | 4,128.1 | 0.0018 | yes |
| 2019 | 2,002,898 | 1,303,941 | 4,105,083 | 16,577 | 4,637.3 | 0.0013 | yes |
| 2020 | 2,945,262 | 1,869,011 | 6,132,888 | 19,276 | 5,589.1 | 0.0010 | yes |
| 2021 | 4,021,396 | 2,441,845 | 8,034,305 | 22,076 | 6,331.6 | 0.0008 | yes |
| 2022 | 8,784,698 | 5,173,919 | 17,059,951 | 28,682 | 8,901.0 | 0.0005 | yes |
| 2023 | 7,591,374 | 4,379,978 | 14,008,668 | 25,275 | 5,964.7 | 0.0006 | yes |

Bluesky 2024 has 866,682 users and 25,546 effective users (`bluesky_L4_by_year.json` → `years.2024`).

The 2010 sample of $N$ tokens comes from about 363 effective users, against about 6,000 in 2023. A year-to-year difference therefore mixes language change with a change in who is sampled. Subsample sd measures token-sampling noise only, not this user-level variance.

### 5.2 Austen's subsamples are identical
$N$ is the Austen corpus size, so every "subsample" of Austen is the whole corpus, and its sd of 0 is not an uncertainty. The sampling sd of a comparable corpus at this $N$ is that of Gutenberg, roughly 0.003–0.007 bits on the entropies.

### 5.3 Plug-in $S_{\rm full}$ bias
$S_{\rm full}$ is the plug-in entropy, biased low by about $(K-1)/(2N\ln2)$: 0.0049 bits for Austen and 0.019–0.023 bits for the social corpora (`S_full_bias`). The captured share $a/b$ with $b=S_1-S_{\rm full}$ is therefore understated by about $a\,\delta/b^2$: roughly 0.06 percentage points for Austen and 0.2 for Twitter. That is small next to the 10-point gap between prose and social media, but comparable to some year-to-year steps. $S_1$ and $S_2$ have their own, smaller, finite-$N$ biases, which are not corrected.

### 5.4 Dictionary choice
Every "dictionary" statistic (`n_minima_dict`, `unseen_dict_share`, `nondict_minima`, and the replication's token filter) depends on web2 (§3.5). It lacks inflections and modern words and includes archaic ones. This explains the 52%-against-1/5 discrepancy and inflates the "non-dictionary" lists.

### 5.5 Fixed $L=4$
Only words of exactly four ASCII letters are modelled, and the results are not claims about the lexicon at large.

### 5.6 Substitution-only neighbourhood
Minima, descent and recovery use single-letter substitutions only. Transpositions, insertions and deletions, the dominant human typing errors [INFERENCE: standard in the spelling-correction literature; not cited here], lie outside the neighbourhood, so §4.8 is a property of the move set as much as of the landscape.

### 5.7 Years below N excluded
Twitter 2006–2009 are excluded because they have fewer than $N$ four-letter tokens; 2009 has 110,914 (`meta.corpora["Twitter 2009"].tokens`). Rather than shrink $N$ for every corpus, the matched-N rule drops these years.

### 5.8 Other limits
- Gutenberg texts are fetched live and cached (`corpus.fetch`), so revised editions could shift the counts.
- English filtering uses the fastText lid.176 argmax ([Joulin et al. 2017], the fastText classifier) and inherits its errors on short, slang-heavy posts.

---

## 6. Reproduction

All commands run from the repository root with `.venv/bin/python`.

```bash
# 0. tests of the estimator, landscape and CV machinery
.venv/bin/python -m pytest tests/test_maxent.py -q

# 1. the study.  CAUTION: overwrites the frozen data/derived/maxent_study.json.
#    Keep the reference copy and compare afterwards.
cp data/derived/maxent_study.json /tmp/maxent_study.reference.json
.venv/bin/python scripts/maxent_study.py | tee /tmp/maxent_study.rerun.log

# 2. figures (reads only the JSON; writes figures/*.svg, which are also frozen)
.venv/bin/python figures/make_figures.py
```

Step 1 needs these inputs:
- the Gutenberg texts, downloaded on first use to `data/<name>.txt`;
- `data/derived/twitter_L4_by_year.json`, built from shards 00–15;
- `data/derived/bluesky_L4_by_year.json`.

The recorded run took 49,968 s (last line of `maxent_study.log`). It uses `WORKERS = os.cpu_count()` processes, all cores, so on a shared machine expect contention.

**The count JSONs are frozen inputs; do not regenerate them casually.**
- They come from the default call of `langfit.social.ingest(source)` (one length, yearly, no output name), which writes `<source>_L4_by_year.json`.
- Called with no file list, it now picks up all 41 downloaded shards and would overwrite the 16-shard input.
- The CLI `scripts/ingest_social.py` always writes the pickle layout, because `--lengths` defaults to `"4"`, so it cannot produce the JSON.
- To rebuild the exact Twitter input, back up the file, then call `social.ingest("twitter", files=[data/tw100m/shard00.parquet … shard15.parquet])` from a script guarded by `if __name__ == "__main__":`, which multiprocessing requires. For Bluesky, call `social.ingest("bluesky")` (`data/bsky/train-*.parquet`).

**Extracting the numbers in this document.** Every table here comes from read-only reads of the JSON, using only the field paths given. Examples:

```bash
.venv/bin/python -c "import json; r=json.load(open('data/derived/maxent_study.json'))['replication']; print(r['S_ind'], r['S_2'], r['S_3'], r['S_full'], r['captured'])"
.venv/bin/python -c "import json; J=json.load(open('data/derived/maxent_study.json')); print([(r['rep'], r['probes']['lmao']['rank']) for r in J['corpora'] if r['name']=='Bluesky'])"
```

### File map

| path | role | status |
|---|---|---|
| `langfit/maxent.py` | model, exact/MAP fits, stopping rule, CV, minima, descent, recovery | frozen |
| `langfit/corpus.py` | Gutenberg download, tokeniser, dictionary | used |
| `langfit/social.py`, `scripts/ingest_social.py` | social-media cleaning, language ID, deduplication, per-period counts | frozen |
| `scripts/maxent_study.py` | replication, matched-N corpora, nested CV → JSON | frozen |
| `data/derived/maxent_study.json` / `.log` | all results / run log | frozen |
| `data/derived/twitter_L4_by_year.json`, `bluesky_L4_by_year.json` | four-letter counts and composition statistics per year | frozen |
| `data/derived/twitter_month_L4_existing16.pkl.gz` | month-binned four-letter counts from shards 00–15 | do not overwrite |
| `figures/make_figures.py`, `figures/plotsvg.py`, `figures/fig[1-4]_*.svg` | figures | frozen |
| `tests/test_maxent.py` | correctness tests (planted models, brute-force minima, audit counterexamples) | |
| `data/refs/stephens_bialek_2007.pdf` | S&B preprint | |

---

## References
Each entry was checked by web search; Basharin's was also checked against the Crossref record.

- **[S&B]** G. J. Stephens & W. Bialek, "Toward a statistical mechanics of four letter words", arXiv:0801.0253 (2007), https://arxiv.org/abs/0801.0253. Published as "Statistical mechanics of letters in words", *Phys. Rev. E* **81**, 066119 (2010), https://doi.org/10.1103/PhysRevE.81.066119.
- **[Jaynes 1957]** E. T. Jaynes, "Information theory and statistical mechanics", *Phys. Rev.* **106**, 620–630 (1957), https://doi.org/10.1103/PhysRev.106.620. S&B's reference list gives the pages as "62–79"; the correct pages are 620–630.
- **[Schneidman, Still, Berry & Bialek 2003]** "Network information and connected correlations", *Phys. Rev. Lett.* **91**, 238701 (2003), https://doi.org/10.1103/PhysRevLett.91.238701.
- **[Darroch & Ratcliff 1972]** J. N. Darroch & D. Ratcliff, "Generalized iterative scaling for log-linear models", *Ann. Math. Stat.* **43**(5), 1470–1480 (1972), https://doi.org/10.1214/aoms/1177692379.
- **[Hopfield 1982]** J. J. Hopfield, "Neural networks and physical systems with emergent collective computational abilities", *PNAS* **79**(8), 2554–2558 (1982), https://doi.org/10.1073/pnas.79.8.2554.
- **[Csiszár 1975]** I. Csiszár, "I-divergence geometry of probability distributions and minimization problems", *Ann. Probab.* **3**(1), 146–158 (1975), https://doi.org/10.1214/aop/1176996454.
- **[Wainwright & Jordan 2008]** M. J. Wainwright & M. I. Jordan, "Graphical models, exponential families, and variational inference", *Found. Trends Mach. Learn.* **1**(1–2), 1–305 (2008), https://doi.org/10.1561/2200000001.
- **[Chen & Rosenfeld 2000]** S. F. Chen & R. Rosenfeld, "A survey of smoothing techniques for ME models", *IEEE Trans. Speech Audio Process.* **8**(1), 37–50 (2000), https://doi.org/10.1109/89.817452.
- **[Gould, Lucidi, Roma & Toint 1999]** N. I. M. Gould, S. Lucidi, M. Roma & Ph. L. Toint, "Solving the trust-region subproblem using the Lanczos method", *SIAM J. Optim.* **9**(2), 504–525 (1999), https://doi.org/10.1137/S1052623497322735.
- **[Boyd & Vandenberghe 2004]** S. Boyd & L. Vandenberghe, *Convex Optimization*, Cambridge University Press (2004), §9.5.1 (Newton decrement), https://web.stanford.edu/~boyd/cvxbook/.
- **[Basharin 1959]** G. P. Basharin, "On a statistical estimate for the entropy of a sequence of independent random variables", *Theory Probab. Appl.* **4**(3), 333–336 (1959), https://doi.org/10.1137/1104033.
- **[Miller 1955]** G. A. Miller, "Note on the bias of information estimates", in H. Quastler (ed.), *Information Theory in Psychology: Problems and Methods*, Free Press, Glencoe IL (1955), pp. 95–100. No DOI; the citation was verified via secondary sources, e.g. Panzeri et al., *J. Neurophysiol.* 2007.
- **[Varma & Simon 2006]** S. Varma & R. Simon, "Bias in error estimation when using cross-validation for model selection", *BMC Bioinformatics* **7**, 91 (2006), https://doi.org/10.1186/1471-2105-7-91.
- **[Garwood 1936]** F. Garwood, "Fiducial limits for the Poisson distribution", *Biometrika* **28**(3–4), 437–442 (1936), https://doi.org/10.2307/2333958.
- **[Simpson 1949]** E. H. Simpson, "Measurement of diversity", *Nature* **163**, 688 (1949), https://doi.org/10.1038/163688a0.
- **[Joulin et al. 2017]** A. Joulin, E. Grave, P. Bojanowski & T. Mikolov, "Bag of tricks for efficient text classification", *Proc. EACL 2017* (vol. 2), 427–431, https://aclanthology.org/E17-2068/.
- **Data and lexica:**
  - Twitter: `enryu43/twitter100m_tweets`, https://huggingface.co/datasets/enryu43/twitter100m_tweets.
  - 12dicts (A. Beale): https://wordlist.aspell.net/12dicts/.
  - web2: Webster's Second International (1934) word list as shipped in BSD `/usr/share/dict` (local README; https://github.com/freebsd/freebsd-src/blob/main/share/dict/web2).
