"""Maximum-entropy models of fixed-length words (Stephens & Bialek, arXiv:0801.0253).

A word of length L over an alphabet of A letters is a configuration of L interacting
"spins" with A states each. The maximum-entropy distribution consistent with the
marginals of a chosen set of cliques (positions) is the Boltzmann distribution

    P(s) = exp( sum_c V_c(s_c) ) / Z,

with one interaction tensor V_c per clique. Cliques:

    singletons {i}           -> the independent model, entropy S_1
    all pairs {i, j}         -> the pairwise model,    entropy S_2   (S&B eq. 2)
    all triples {i, j, k}    -> the third-order model, entropy S_3

and S_1 >= S_2 >= S_3 >= S_N (Schneidman, Still, Berry & Bialek, PRL 91 238701):
each level can only lower the entropy, and the drops are the connected information
carried at each order.

For L = 4, A = 26 the state space has 26^4 = 456,976 configurations, so the partition
function, every marginal, every entropy and every local minimum are computed by
exact enumeration. Nothing is sampled.

Two fitting modes, both convex in V:

  exact      match the empirical clique marginals exactly (S&B protocol). Cells with
             zero empirical marginal get V = -inf: any word containing such a letter
             combination has probability exactly zero. This defines S_k.
  l2         maximise the token log-likelihood minus lam * sum V^2. Every string gets
             finite energy, which is what a comparison of *unseen* strings across
             corpora needs. lam is chosen by held-out log-likelihood
             (``select_l2_by_cv``), not set by hand.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
A = len(ALPHABET)
LOG2 = np.log(2.0)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

@dataclass
class WordCounts:
    """Token counts of fixed-length words, as integer letter arrays."""

    states: np.ndarray          # (n_types, L) int letter indices
    counts: np.ndarray          # (n_types,) token counts
    L: int

    @classmethod
    def from_counter(cls, counter, L: int = 4) -> WordCounts:
        words = sorted(w for w in counter if len(w) == L and set(w) <= set(ALPHABET))
        states = np.array([[ord(c) - 97 for c in w] for w in words], dtype=np.int64)
        counts = np.array([counter[w] for w in words], dtype=np.float64)
        return cls(states=states.reshape(-1, L), counts=counts, L=L)

    @property
    def words(self) -> list[str]:
        return ["".join(ALPHABET[i] for i in s) for s in self.states]

    @property
    def n_tokens(self) -> float:
        return float(self.counts.sum())

    def flat_index(self) -> np.ndarray:
        """Index of each type in the raveled A^L tensor."""
        return np.ravel_multi_index(self.states.T, (A,) * self.L)

    def empirical(self) -> np.ndarray:
        """The empirical distribution as a dense A^L tensor."""
        p = np.zeros(A ** self.L)
        p[self.flat_index()] = self.counts / self.counts.sum()
        return p.reshape((A,) * self.L)

    def split(self, n_folds: int, rng: np.random.Generator) -> list[np.ndarray]:
        """Token-level K-fold split: each fold is a count vector over the same types.

        Tokens, not types, are assigned to folds, because the quantity being
        estimated is the probability of the next token.
        """
        folds = np.zeros((n_folds, self.counts.size))
        for t, c in enumerate(self.counts.astype(np.int64)):
            folds[:, t] = rng.multinomial(c, np.full(n_folds, 1.0 / n_folds))
        return list(folds)


def cliques(order: int, L: int = 4) -> list[tuple[int, ...]]:
    return list(combinations(range(L), order))


def marginal(p: np.ndarray, clique: Sequence[int]) -> np.ndarray:
    """Marginal of a dense A^L tensor on the given positions (kept in order)."""
    other = tuple(ax for ax in range(p.ndim) if ax not in clique)
    return p.sum(axis=other)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

@dataclass
class MaxEnt:
    """Boltzmann distribution over A^L words with one interaction tensor per clique."""

    L: int
    cliques: list[tuple[int, ...]]
    V: list[np.ndarray] = field(default_factory=list)
    lam: float | None = None            # L2 strength if fitted with l2, else None

    def _shape(self, c):
        return tuple(A if ax in c else 1 for ax in range(self.L))

    def log_weights(self) -> np.ndarray:
        """sum_c V_c(s_c) for every word s: the negative energy, unnormalised."""
        lw = np.zeros((A,) * self.L)
        for c, v in zip(self.cliques, self.V):
            lw = lw + v.reshape(self._shape(c))
        return lw

    def log_prob(self) -> np.ndarray:
        lw = self.log_weights()
        return lw - logsumexp(lw)

    def prob(self) -> np.ndarray:
        return np.exp(self.log_prob())

    def energy(self) -> np.ndarray:
        """E(s) = -ln P(s), in nats. +inf for forbidden words."""
        return -self.log_prob()

    def entropy_bits(self) -> float:
        lp = self.log_prob()
        p = np.exp(lp)
        m = p > 0
        return float(-(p[m] * lp[m]).sum() / LOG2)

    def cross_entropy_bits(self, wc: WordCounts, counts: np.ndarray | None = None) -> float:
        """Mean code length (bits per token) of the given tokens under the model."""
        c = wc.counts if counts is None else counts
        lp = self.log_prob().ravel()[wc.flat_index()]
        m = c > 0
        return float(-(c[m] * lp[m]).sum() / c[m].sum() / LOG2)

    def word_energy(self, words: Iterable[str]) -> np.ndarray:
        e = self.energy()
        out = []
        for w in words:
            out.append(float(e[tuple(ord(ch) - 97 for ch in w)]))
        return np.array(out)


def _pack(V, masks):
    return np.concatenate([v[m] for v, m in zip(V, masks)])


def _unpack(theta, masks, fill):
    out, k = [], 0
    for m in masks:
        v = np.full(m.shape, fill)
        n = int(m.sum())
        v[m] = theta[k:k + n]
        k += n
        out.append(v)
    return out


def _features(L: int, cl: list[tuple[int, ...]], masks: list[np.ndarray]):
    """Sparse word-by-feature indicator matrix over the free cells, plus the mask of
    words that contain a forbidden (-inf) cell and therefore have zero probability.

    Every word activates exactly one cell per clique, so the matrix has len(cl)
    nonzeros per row; model marginals are F^T p and Hessian-vector products of the
    dual are two sparse mat-vecs.
    """
    from scipy import sparse

    grid = np.indices((A,) * L).reshape(L, -1)
    n = grid.shape[1]
    rows, cols = [], []
    forbidden = np.zeros(n, dtype=bool)
    offset = 0
    for c, m in zip(cl, masks):
        cell = np.ravel_multi_index(grid[list(c)], (A,) * len(c))
        colmap = np.full(m.size, -1, dtype=np.int64)
        colmap[m.ravel()] = np.arange(int(m.sum())) + offset
        col = colmap[cell]
        forbidden |= col < 0
        ok = col >= 0
        rows.append(np.nonzero(ok)[0])
        cols.append(col[ok])
        offset += int(m.sum())
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    F = sparse.csr_matrix((np.ones(r.size), (r, c)), shape=(n, offset))
    return F, forbidden


def _independent(wc: WordCounts, counts: np.ndarray, mode: str, lam: float):
    """Order 1 in closed form: the maximum-entropy distribution with given single-site
    marginals is their product, so V_i = ln mu_i (-inf for unused letters). No
    optimisation; exact to rounding."""
    keep = counts > 0
    p_emp = WordCounts(wc.states[keep], counts[keep], wc.L).empirical()
    cl = cliques(1, wc.L)
    with np.errstate(divide="ignore"):
        V = [np.log(marginal(p_emp, c)) for c in cl]
    return MaxEnt(L=wc.L, cliques=cl, V=V, lam=None), {
        "converged": True, "iterations": 0, "newton_polish": 0,
        "max_gradient_in_se": 0.0, "entropy_error_in_se": 0.0, "closed_form": True}


def fit(wc: WordCounts, order: int, mode: str = "exact", lam: float = 0.0,
        counts: np.ndarray | None = None, init: MaxEnt | None = None,
        resolution: float = 0.01, maxiter: int = 500) -> tuple[MaxEnt, dict]:
    """Fit the order-k maximum-entropy model.

    ``exact``: S&B's construction. Minimise the convex dual log Z - sum_c <mu_c, V_c>
    over cells with nonzero empirical marginal; the others are fixed at -inf. At the
    optimum the model marginals equal the empirical ones. Order 1 is closed-form.

    ``l2``: the same dual plus lam * sum V^2 over all cells (a Gaussian prior; see
    ``prior_lambda``). Every string then has finite energy.

    Solver: trust-region Newton-Krylov, then Newton steps if the trust region stalls.
    Stopping rule, both parts relative to sampling noise (``resolution`` is the
    tolerated fraction of one standard error):
      (a) every marginal cell within resolution * se_c, se_c = max(sqrt(mu(1-mu)/N), 1/N);
      (b) exact mode: the entropy error within resolution * se_S. The error of
          H(P_theta) relative to the maximum entropy S* = min f is bounded by
          (f(theta) - f*) + |f(theta) - H(P_theta)| = (f - f*) + |g.theta|, and the dual
          suboptimality f - f* is estimated by half the squared Newton decrement
          g' H^-1 g (exact for the local quadratic model). se_S is the first-order
          sampling sd of the plug-in entropy, sqrt(Var_p[ln p]/N) (Basharin 1959),
          floored at the second-order bias scale (K-1)/(2N) (Miller 1955) because the
          first-order term vanishes for uniform counts.
    """
    if mode not in ("exact", "l2"):
        raise ValueError(mode)
    c_used = wc.counts if counts is None else counts
    if order == 1 and mode == "exact":
        return _independent(wc, c_used, mode, lam)
    keep = c_used > 0
    sub = WordCounts(wc.states[keep], c_used[keep], wc.L)
    p_emp = sub.empirical()
    cl = cliques(order, wc.L)
    mus = [marginal(p_emp, c) for c in cl]
    if mode == "exact":
        masks = [mu > 0 for mu in mus]
        fill = -np.inf
    else:
        masks = [np.ones_like(mu, dtype=bool) for mu in mus]
        fill = 0.0
    model = MaxEnt(L=wc.L, cliques=cl, lam=None if mode == "exact" else lam)
    F, forbidden = _features(wc.L, cl, masks)
    FT = F.T.tocsr()
    mu_vec = _pack(mus, masks)
    reg = 2.0 * lam if mode == "l2" else 0.0

    if init is not None and init.cliques == cl:
        theta0 = _pack([np.where(np.isfinite(v), v, 0.0) for v in init.V], masks)
    else:
        theta0 = _pack([np.log(np.where(mu > 0, mu, 1.0)) / len(cl) for mu in mus], masks)

    state = {}

    def _probs(theta):
        key = theta.tobytes()
        if state.get("key") != key:
            lw = F @ theta
            lw[forbidden] = -np.inf
            lz = logsumexp(lw)
            p = np.exp(lw - lz)
            state.update(key=key, lz=lz, p=p, m=FT @ p)
        return state

    def fun(theta):
        s = _probs(theta)
        return s["lz"] - float(mu_vec @ theta) + 0.5 * reg * float(theta @ theta)

    def jac(theta):
        s = _probs(theta)
        return s["m"] - mu_vec + reg * theta

    def hessp(theta, v):
        s = _probs(theta)
        return FT @ (s["p"] * (F @ v)) - s["m"] * float(s["m"] @ v) + reg * v

    from scipy.sparse.linalg import LinearOperator, cg

    def newton(theta):
        """Inexact Newton step d ~ -H^-1 g and the squared Newton decrement -g.d.

        The decrement is compared with a small fraction of a standard error, so it is
        needed only to leading order: CG stops at a 1e-3 relative residual. A tighter
        tolerance stagnates, because the Hessian of the over-complete clique
        parameterisation is singular along its gauge directions."""
        g = jac(theta)
        H = LinearOperator((theta.size,) * 2, matvec=lambda v: hessp(theta, v))
        d, _ = cg(H, -g, rtol=1e-3, maxiter=theta.size)
        return d, float(-g @ d)

    N = sub.n_tokens
    se = np.maximum(np.sqrt(mu_vec * (1.0 - mu_vec) / N), 1.0 / N)
    pw = sub.counts / N
    lnp = np.log(pw)
    var = max(float((pw * lnp ** 2).sum() - (pw * lnp).sum() ** 2), 0.0)
    se_S = max(np.sqrt(var / N), (pw.size - 1) / (2.0 * N))

    def marg_se(theta):
        return float(np.abs(jac(theta) / se).max())

    def ent_se(theta, decrement):
        if mode != "exact":
            return 0.0
        return (0.5 * abs(decrement) + abs(float(jac(theta) @ theta))) / se_S

    def criteria(theta):
        m_ = marg_se(theta)
        if mode != "exact" or m_ >= resolution:
            return m_, (np.inf if mode == "exact" else 0.0), None
        d, dec = newton(theta)
        return m_, ent_se(theta, dec), d

    def stop(intermediate_result):
        m_, e_, _ = criteria(intermediate_result.x)
        if m_ < resolution and e_ < resolution:
            raise StopIteration

    res = minimize(fun, theta0, jac=jac, hessp=hessp, method="trust-krylov",
                   callback=stop, options={"gtol": 0.0, "maxiter": maxiter})
    theta = res.x
    m_, e_, d = criteria(theta)
    # The trust region can stall near the optimum once the objective's decrease falls
    # below float64 rounding of |f| ~ 10 nats while the directly computed gradient is
    # still accurate. Finish with full Newton steps, accepted only if they improve.
    polish = 0
    while not (m_ < resolution and e_ < resolution) and polish < 30:
        if d is None:
            d, _ = newton(theta)
        new = theta + d
        if not np.all(np.isfinite(new)):
            break
        m2_, e2_, d2 = criteria(new)
        if max(m2_, e2_ if np.isfinite(e2_) else m2_) >= max(m_, e_ if np.isfinite(e_) else m_):
            break
        theta, m_, e_, d = new, m2_, e2_, d2
        polish += 1
    model.V = _unpack(theta, masks, fill)
    p = model.prob()
    mism = max(float(np.abs(marginal(p, c) - mu).max()) for c, mu in zip(cl, mus))
    return model, {"converged": bool(m_ < resolution and e_ < resolution),
                   "iterations": int(res.nit), "newton_polish": polish,
                   "max_gradient_in_se": m_, "entropy_error_in_se": float(e_),
                   "resolution": resolution, "max_marginal_error": mism,
                   "message": str(res.message)}


def prior_lambda(sigma2: float, n_tokens: float) -> float:
    """L2 strength of a Gaussian prior V ~ N(0, sigma2) on every coupling.

    The posterior over couplings is  N * NLL_per_token + ||V||^2 / (2 sigma2), and ``fit``
    minimises the per-token form, so lam = 1 / (2 sigma2 N). The prior width is the
    single hyperparameter of the regularised model, and it does not depend on N.
    """
    return 1.0 / (2.0 * sigma2 * n_tokens)


def cv_fold(wc: WordCounts, order: int, sigma2s: Sequence[float], fold: int,
            n_folds: int = 5, seed: int = 0) -> list[tuple[float, float]]:
    """Held-out cross-entropy (bits per token) of one fold, for each prior width.

    Fits run from the narrowest prior to the widest, each warm-started from the last.
    """
    folds = wc.split(n_folds, np.random.default_rng(seed))
    test = folds[fold]
    train = wc.counts - test
    out, init = [], None
    for s2 in sorted(sigma2s):
        m, _ = fit(wc, order, "l2", lam=prior_lambda(s2, train.sum()), counts=train, init=init)
        init = m
        out.append((float(s2), m.cross_entropy_bits(wc, test)))
    return out


def aggregate_cv(per_fold: Sequence[list[tuple[float, float]]]) -> dict:
    """Mean and standard error over folds of the validation loss at each prior width."""
    grid = sorted({s for f in per_fold for s, _ in f})
    M = np.array([[dict(f)[s] for s in grid] for f in per_fold])
    mean = M.mean(axis=0)
    se = M.std(axis=0, ddof=1) / np.sqrt(M.shape[0])
    best = int(np.argmin(mean))
    return {"sigma2": grid, "bits": mean.tolist(), "se": se.tolist(),
            "best_sigma2": grid[best], "best_bits": float(mean[best]),
            "interior": 0 < best < len(grid) - 1}


# Prior widths are searched on a half-decade grid. The grid only discretises the
# search; it is extended outward whenever an endpoint is selected, so the reported
# optimum is always bracketed. Starting range: one decade either side of unit
# coupling variance, the natural scale of a log-probability.
SIGMA2_START = tuple(10.0 ** (k / 2) for k in range(-2, 3))
SIGMA2_STEP = 10.0 ** 0.5
SIGMA2_MAX_EXTENSIONS = 8


def select_prior(wc: WordCounts, order: int, n_folds: int = 4, seed: int = 0) -> dict:
    """Choose the Gaussian prior width by K-fold CV, extending the grid until the
    optimum is interior. ``interior`` is False only if the extension cap was hit."""
    grid = list(SIGMA2_START)
    per_fold = [cv_fold(wc, order, grid, k, n_folds, seed) for k in range(n_folds)]
    agg = aggregate_cv(per_fold)
    for _ in range(SIGMA2_MAX_EXTENSIONS):
        if agg["interior"]:
            break
        lo = agg["best_sigma2"] == agg["sigma2"][0]
        new = agg["sigma2"][0] / SIGMA2_STEP if lo else agg["sigma2"][-1] * SIGMA2_STEP
        for k in range(n_folds):
            per_fold[k] = per_fold[k] + cv_fold(wc, order, [new], k, n_folds, seed)
        agg = aggregate_cv(per_fold)
    return agg


def nested_cv_fold(wc: WordCounts, order: int, outer: int, n_outer: int = 5,
                   n_inner: int = 4, seed: int = 0) -> dict:
    """One outer fold of nested cross-validation.

    The prior width is chosen by inner CV on the outer training tokens only; the model
    is then refit on all outer training tokens and scored on the outer test tokens,
    which played no part in fitting or in choosing the prior. The mean of this score
    over outer folds is an unbiased estimate of held-out code length.
    """
    folds = wc.split(n_outer, np.random.default_rng(seed))
    test = folds[outer]
    train = wc.counts - test
    inner = WordCounts(wc.states, train, wc.L)
    sel = select_prior(inner, order, n_inner, seed + 1 + outer)
    m, d = fit(wc, order, "l2", lam=prior_lambda(sel["best_sigma2"], train.sum()), counts=train)
    return {"outer": outer, "sigma2": sel["best_sigma2"], "interior": sel["interior"],
            "test_bits": m.cross_entropy_bits(wc, test), "converged": d["converged"],
            "inner": sel}


def summarise_nested(folds: Sequence[dict]) -> dict:
    b = np.array([f["test_bits"] for f in folds])
    return {"bits": float(b.mean()), "se": float(b.std(ddof=1) / np.sqrt(b.size)),
            "per_fold": b.tolist(), "sigma2": [f["sigma2"] for f in folds],
            "all_interior": all(f["interior"] for f in folds),
            "all_converged": all(f["converged"] for f in folds)}


# --------------------------------------------------------------------------- #
# landscape
# --------------------------------------------------------------------------- #

def local_minima(E: np.ndarray) -> np.ndarray:
    """Boolean A^L mask of strict local minima under single-letter substitution.

    A word is a local minimum iff it is the unique lowest-energy point of each of the
    L one-dimensional fibres through it (vary one position, hold the others). This is
    S&B's definition: any single-letter change raises the energy.
    """
    is_min = np.isfinite(E)
    for ax in range(E.ndim):
        srt = np.sort(E, axis=ax)
        lowest = np.take(srt, [0], axis=ax)
        second = np.take(srt, [1], axis=ax)
        is_min &= (E == lowest) & (second > lowest)
    return is_min


def descend(E: np.ndarray) -> np.ndarray:
    """Steepest-descent map under single-letter substitution, for every word.

    Returns, for each flat state, the flat index of its terminal state. Each step
    moves to the lowest-energy single-substitution neighbour if that is strictly
    lower; ties go to the earliest position, then the earliest letter. Forbidden
    strings (E = +inf) relax too, since every finite neighbour is lower; a string whose
    whole neighbourhood is forbidden stays put. Every strict local minimum is a
    terminal state, but a terminal state need not be a strict minimum: descent can
    stop on a plateau of tied energies. Wells are therefore always reported via
    ``local_minima`` (S&B's strict definition), never via terminal states.
    """
    shape = E.shape
    flat = E.ravel()
    n = flat.size
    best_e = flat.copy()
    best_i = np.arange(n)
    idx = np.arange(n).reshape(shape)
    for ax in range(E.ndim):
        amin = np.argmin(E, axis=ax)                       # best letter on each fibre
        emin = np.take_along_axis(E, np.expand_dims(amin, ax), axis=ax)
        imin = np.take_along_axis(idx, np.expand_dims(amin, ax), axis=ax)
        emin = np.broadcast_to(emin, shape).ravel()
        imin = np.broadcast_to(imin, shape).ravel()
        better = emin < best_e
        best_e = np.where(better, emin, best_e)
        best_i = np.where(better, imin, best_i)
    target = best_i
    # pointer jumping: each pass doubles the path length followed
    while True:
        nxt = target[target]
        if np.array_equal(nxt, target):
            return target
        target = nxt


def unravel(i: int, L: int = 4) -> str:
    return "".join(ALPHABET[k] for k in np.unravel_index(int(i), (A,) * L))


def ravel(w: str) -> int:
    return int(np.ravel_multi_index(tuple(ord(c) - 97 for c in w), (A,) * len(w)))


def substitution_recovery(E: np.ndarray, wc: WordCounts, target: np.ndarray | None = None
                          ) -> np.ndarray:
    """For each word type, the fraction of its L*(A-1) single-letter misspellings whose
    steepest descent ends exactly on the intended word.

    This is decoding accuracy, the operational meaning of S&B's remark that for stable
    words "any single letter spelling error can always be corrected by relaxing to the
    nearest local minimum". A word that is not itself a terminal state scores zero.
    """
    target = descend(E) if target is None else target
    flat = wc.flat_index()
    strides = A ** np.arange(wc.L - 1, -1, -1)
    hits = np.zeros(flat.size)
    for ax in range(wc.L):
        for a in range(A):
            mut = flat + (a - wc.states[:, ax]) * strides[ax]
            hits += np.where(wc.states[:, ax] != a, target[mut] == flat, 0.0)
    return hits / (wc.L * (A - 1))


def descent_path(E: np.ndarray, word: str) -> list[tuple[str, float]]:
    """The explicit steepest-descent path from ``word`` to its terminal state (same
    rule and tie-break as ``descend``)."""
    cur = tuple(ord(c) - 97 for c in word)
    path = [("".join(ALPHABET[k] for k in cur), float(E[cur]))]
    while True:
        best, best_e = None, E[cur]
        for ax in range(E.ndim):
            for a in range(A):
                if a == cur[ax]:
                    continue
                nb = cur[:ax] + (a,) + cur[ax + 1:]
                if E[nb] < best_e:
                    best, best_e = nb, E[nb]
        if best is None:
            return path
        cur = best
        path.append(("".join(ALPHABET[k] for k in cur), float(best_e)))
