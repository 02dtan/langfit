"""Fixed-length Potts / maximum-entropy models of words, by dense tensor algebra.

A word of length L over A = 26 letters is a configuration s = (s_1..s_L) of L Potts
spins. Given a set of cliques C (tuples of positions), the maximum-entropy distribution
consistent with the empirical marginals mu_c of every c in C is the Boltzmann law

    P(s) = exp( sum_{c in C} V_c(s_c) ) / Z                                  (1)

(Jaynes 1957; for words, Stephens & Bialek, arXiv:0801.0253 / Phys. Rev. E 81, 066119,
2010). ``cliques(k, L)`` gives all k-subsets: k = 1 independent sites, k = 2 pairwise
couplings, k = 3 three-body interactions. The entropies S_1 >= S_2 >= S_3 >= S_L of
the successive levels form the maximum-entropy hierarchy of Schneidman, Still, Berry
& Bialek (PRL 91, 238701, 2003) and Amari (IEEE Trans. Inf. Theory 47, 1701, 2001).

This module is the dense counterpart of the frozen ``langfit.maxent``: same estimator
semantics, same initial point, same solver and the same stopping rule, but
every quantity is a broadcast or an axis reduction on the A^L grid, never a sparse
feature matrix:

  log-weights          sum_c V_c broadcast to A^L               (one pass per clique)
  model marginals      axis sums of P, memoised over supersets  (a few passes of A^L)
  Hessian-vector Hv    Cov_P[f_c, f.v] = marg_c(P * (f.v)) - m_c (m.v), where f.v is
                       itself the broadcast sum of the direction tensors

so L = 5 (26^5 ~ 1.19e7 states, ~95 MB per float64 grid) is tractable at order 3.

Fitting modes (both convex in V):

  exact  S&B: minimise the dual log Z - sum_c <mu_c, V_c> over cells with mu_c > 0; the
         other cells are fixed at -inf (those letter combinations have probability 0).
         At the optimum the model marginals equal the empirical ones.
  map    Gaussian prior V ~ N(0, sigma2) on every cell: the same dual + lam * |V|^2 with
         lam = 1 / (2 sigma2 N) (posterior per token; ``maxent.prior_lambda``). Every
         string has finite energy. sigma2 is chosen by K-fold CV (``select_prior``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, cg

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
# Newton polish: a rejected step is halved at most this many times (down to ~1e-3 of
# the full step) before polishing stops. Numerical tolerance.
BACKTRACK = 10
A = len(ALPHABET)
LOG2 = np.log(2.0)
_ALPHA = frozenset(ALPHABET)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def cliques(order: int, L: int) -> list[tuple[int, ...]]:
    """All position subsets of size ``order``, in lexicographic order."""
    return list(combinations(range(L), order))


def counts_to_arrays(counter, L: int) -> tuple[np.ndarray, np.ndarray]:
    """{word: count} -> (states (n, L) int64 letter indices, counts (n,) float64).

    Only pure a-z words of length L are kept; types are sorted alphabetically (the same
    order as ``maxent.WordCounts.from_counter``, so token splits coincide)."""
    words = sorted(w for w in counter if len(w) == L and set(w) <= _ALPHA)
    buf = np.frombuffer("".join(words).encode("ascii"), dtype=np.uint8)
    states = (buf.astype(np.int64) - 97).reshape(-1, L)
    counts = np.array([counter[w] for w in words], dtype=np.float64)
    return states, counts


def flat_index(states: np.ndarray, L: int) -> np.ndarray:
    """Index of each state in the raveled A^L grid."""
    return np.ravel_multi_index(np.asarray(states).T, (A,) * L)


def ravel(word: str) -> int:
    return int(np.ravel_multi_index(tuple(ord(c) - 97 for c in word), (A,) * len(word)))


def unravel(i: int, L: int) -> str:
    return "".join(ALPHABET[k] for k in np.unravel_index(int(i), (A,) * L))


def empirical(states: np.ndarray, counts: np.ndarray, L: int) -> np.ndarray:
    """Empirical distribution as a dense A^L tensor."""
    p = np.zeros(A ** L)
    np.add.at(p, flat_index(states, L), counts / counts.sum())
    return p.reshape((A,) * L)


def marginals(p: np.ndarray, cl: Sequence[tuple[int, ...]]) -> list[np.ndarray]:
    """Marginals of the dense tensor ``p`` on each clique (axes kept in order).

    Each marginal is reduced from the memoised marginal of its superset obtained by
    adding the smallest missing position, so the full grid is swept at most L - 1
    times however many cliques there are."""
    L = p.ndim
    memo: dict[tuple[int, ...], np.ndarray] = {tuple(range(L)): p}

    def get(c: tuple[int, ...]) -> np.ndarray:
        if c not in memo:
            j = min(ax for ax in range(L) if ax not in c)
            parent = tuple(sorted(c + (j,)))
            memo[c] = get(parent).sum(axis=parent.index(j))
        return memo[c]

    return [get(tuple(c)) for c in cl]


def split(counts: np.ndarray, n_folds: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Token-level K-fold split: each fold is a count vector over the same types.

    Tokens, not types, are assigned to folds (the target is the probability of the next
    token). Draws the same random stream as ``maxent.WordCounts.split``."""
    folds = rng.multinomial(counts.astype(np.int64), np.full(n_folds, 1.0 / n_folds))
    return list(folds.T.astype(np.float64))


def subsample(states: np.ndarray, counts: np.ndarray, N: int, seed: int
              ) -> tuple[np.ndarray, np.ndarray]:
    """N tokens drawn without replacement (multivariate hypergeometric)."""
    rng = np.random.default_rng(seed)
    if N >= counts.sum():
        return states, counts
    c = rng.multivariate_hypergeometric(counts.astype(np.int64), int(N)).astype(np.float64)
    keep = c > 0
    return states[keep], c[keep]


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

def _broadcast_sum(L: int, cl, V, out: np.ndarray | None = None) -> np.ndarray:
    """sum_c V_c(s_c) on the A^L grid.

    Built position by position: the partial sum over cliques whose last position is
    < k lives on A^k and is extended by one axis before the cliques ending at k are
    added, so only the cliques that contain the last position touch the full grid."""
    acc = np.zeros(())
    for k in range(L):
        group = [(c, v) for c, v in zip(cl, V) if max(c) == k]
        shape = (A,) * (k + 1)
        nxt = out if (k == L - 1 and out is not None) else np.empty(shape)
        nxt[...] = acc[..., None]
        for c, v in group:
            nxt += v.reshape(tuple(A if ax in c else 1 for ax in range(k + 1)))
        acc = nxt
    return acc


def _normalise(lw: np.ndarray) -> tuple[np.ndarray, float]:
    """In place: lw -> P = exp(lw) / Z; returns (P, ln Z)."""
    mx = float(lw.max())
    lw -= mx
    np.exp(lw, out=lw)
    z = float(lw.sum())
    lw /= z
    return lw, mx + np.log(z)


class Potts:
    """Boltzmann distribution (1) over A^L words, one interaction tensor per clique."""

    def __init__(self, L: int, cliques: list[tuple[int, ...]], V: list[np.ndarray],
                 sigma2: float | None = None):
        self.L = L
        self.cliques = [tuple(c) for c in cliques]
        self.V = V
        self.sigma2 = sigma2            # prior width if fitted in map mode, else None

    def log_weights(self) -> np.ndarray:
        """sum_c V_c(s_c): the negative energy in nats, unnormalised."""
        return _broadcast_sum(self.L, self.cliques, self.V)

    def log_prob(self) -> np.ndarray:
        lw = self.log_weights()
        mx = float(lw.max())
        lw -= mx
        lz = np.log(np.exp(lw).sum())
        lw -= lz
        return lw

    def prob(self) -> np.ndarray:
        return _normalise(self.log_weights())[0]

    def energy_bits(self) -> np.ndarray:
        """E(s) = -log2 P(s); +inf for forbidden words."""
        lp = self.log_prob()
        lp *= -1.0 / LOG2
        return lp

    def entropy_bits(self) -> float:
        lp = self.log_prob()
        p = np.exp(lp)
        m = p > 0
        return float(-(p[m] * lp[m]).sum() / LOG2)

    def cross_entropy_bits(self, states: np.ndarray, counts: np.ndarray) -> float:
        """Mean code length (bits per token) of the given tokens under the model."""
        lp = self.log_prob().ravel()[flat_index(states, self.L)]
        m = counts > 0
        return float(-(counts[m] * lp[m]).sum() / counts[m].sum() / LOG2)

    def marginals(self, cl: Sequence[tuple[int, ...]] | None = None) -> list[np.ndarray]:
        return marginals(self.prob(), self.cliques if cl is None else cl)

    def word_energy_bits(self, words: Iterable[str]) -> np.ndarray:
        e = self.energy_bits()
        return np.array([float(e[tuple(ord(ch) - 97 for ch in w)]) for w in words])


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #

def prior_lambda(sigma2: float, n_tokens: float) -> float:
    """Per-token L2 strength of a Gaussian prior V ~ N(0, sigma2): 1 / (2 sigma2 N)."""
    return 1.0 / (2.0 * sigma2 * n_tokens)


def _init_vector(init: Potts, cl, masks, L: int) -> np.ndarray:
    """Starting couplings from ``init``. Same cliques: copied. Otherwise each clique of
    ``init`` must lie inside some clique of ``cl`` and its tensor is added, broadcast,
    to the first such clique (e.g. an order-2 fit warm-starts order 3 at the same
    distribution). Non-finite cells start at 0."""
    V = [np.zeros((A,) * len(c)) for c in cl]
    for ci, v in zip(init.cliques, init.V):
        k = next((k for k, c in enumerate(cl) if set(ci) <= set(c)), None)
        if k is None:
            raise ValueError(f"init clique {ci} is not contained in any clique of {cl}")
        shape = tuple(A if ax in ci else 1 for ax in cl[k])
        V[k] = V[k] + np.where(np.isfinite(v), v, 0.0).reshape(shape)
    return np.concatenate([v[m] for v, m in zip(V, masks)])


def _independent(states, counts, L):
    """Order 1, exact, in closed form: the maximum-entropy distribution with given
    single-site marginals is their product, V_i = ln mu_i (-inf for unused letters)."""
    cl = cliques(1, L)
    V = []
    for i in range(L):
        mu = np.bincount(states[:, i], weights=counts, minlength=A) / counts.sum()
        with np.errstate(divide="ignore"):
            V.append(np.log(mu))
    return Potts(L, cl, V), {
        "converged": True, "iterations": 0, "newton_polish": 0,
        "max_gradient_in_se": 0.0, "entropy_error_in_se": 0.0, "closed_form": True}


def fit(states: np.ndarray, counts: np.ndarray, L: int, order: int, mode: str = "exact",
        sigma2: float | None = None, init: Potts | None = None, resolution: float = 0.01,
        maxiter: int = 500, cliques_: list[tuple[int, ...]] | None = None
        ) -> tuple[Potts, dict]:
    """Fit the maximum-entropy model on the cliques of size ``order`` (or on an explicit
    clique list ``cliques_``, e.g. all pairs plus one triplet; ``order`` is then ignored).

    ``exact``: minimise log Z - sum_c <mu_c, V_c> over cells with mu_c > 0, the rest at
    -inf. Order 1 is closed-form. ``map``: the same dual + lam |V|^2 over all cells,
    lam = 1 / (2 sigma2 N) with N the token count of ``counts``.

    Solver: trust-region Newton-Krylov, then full Newton steps if the trust region
    stalls. Stopping rule (``resolution`` = tolerated fraction of one standard error):
      (a) the gradient in every cell within resolution * se_c,
          se_c = max(sqrt(mu(1-mu)/N), 1/N). In exact mode the gradient is
          mu_model - mu_emp (marginal matching); in map mode it is
          mu_model - mu_emp + V/(sigma2 N), i.e. stationarity of the posterior, and the
          marginal gap itself is reported separately as ``max_marginal_error``;
      (b) exact mode: an estimated entropy error (f - f*) + |g.theta| with f - f*
          estimated by half the squared Newton decrement (local quadratic model; an
          estimate, not a bound), within resolution * se_S;
          se_S = sqrt(Var_p[ln p] / N) (Basharin 1959) floored at the second-order bias
          scale (K - 1)/(2N) (Miller 1955).
    Identical semantics to ``maxent.fit`` (whose ``l2`` mode is ``map`` here).
    """
    if mode not in ("exact", "map"):
        raise ValueError(mode)
    if mode == "map" and (sigma2 is None or not sigma2 > 0):
        raise ValueError("map mode needs a prior width sigma2 > 0")
    states = np.asarray(states, dtype=np.int64).reshape(-1, L)
    counts = np.asarray(counts, dtype=np.float64)
    keep = counts > 0
    states, counts = states[keep], counts[keep]
    cl = [tuple(c) for c in cliques_] if cliques_ is not None else cliques(order, L)
    if cliques_ is None and order == 1 and mode == "exact":
        return _independent(states, counts, L)

    N = float(counts.sum())
    p_emp = empirical(states, counts, L)
    mus = marginals(p_emp, cl)
    del p_emp
    if mode == "exact":
        masks = [mu > 0 for mu in mus]
        fill = -np.inf
    else:
        masks = [np.ones(mu.shape, dtype=bool) for mu in mus]
        fill = 0.0
    idx = [np.flatnonzero(m.ravel()) for m in masks]
    sizes = np.array([i.size for i in idx])
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    mu_vec = np.concatenate([mu.ravel()[i] for mu, i in zip(mus, idx)])
    reg = 2.0 * prior_lambda(sigma2, N) if mode == "map" else 0.0

    def unpack(theta, fill_value):
        out = []
        for k, (c, i) in enumerate(zip(cl, idx)):
            v = np.full(A ** len(c), fill_value)
            v[i] = theta[bounds[k]:bounds[k + 1]]
            out.append(v.reshape((A,) * len(c)))
        return out

    def pack(tensors):
        return np.concatenate([t.ravel()[i] for t, i in zip(tensors, idx)])

    if init is not None:
        theta0 = _init_vector(init, cl, masks, L)
    else:
        theta0 = pack([np.log(np.where(mu > 0, mu, 1.0)) / len(cl) for mu in mus])

    # Exact mode: every word containing a forbidden cell has P = 0, so the dual only
    # involves the allowed support. When the support is small enough that gathering
    # one coupling per clique for each allowed word costs less than one broadcast over
    # the grid (n_allowed * |C| < A^L), work on the support: J[k][w] is the index of
    # word w's cell among clique k's free cells; log-weights are sums of gathered
    # couplings, and marginals / Hessian-vector products are bincounts over J[k].
    # Otherwise dense.
    J = None
    if mode == "exact":
        allowed = _broadcast_sum(L, cl, [np.where(m, 0.0, -np.inf) for m in masks])
        support = np.flatnonzero(np.isfinite(allowed.ravel()))
        del allowed
        if support.size * len(cl) < A ** L:
            coords = np.unravel_index(support, (A,) * L)
            # J[k]: local index, within clique k's free cells, of each allowed word
            J = []
            for c, i in zip(cl, idx):
                colmap = np.full(A ** len(c), -1, dtype=np.int64)
                colmap[i] = np.arange(i.size)
                J.append(colmap[np.ravel_multi_index(tuple(coords[a] for a in c),
                                                     (A,) * len(c))])
            del coords

    state: dict = {}
    if J is not None:
        def _gather(theta):
            out = theta[bounds[0]:bounds[1]][J[0]]
            for k in range(1, len(cl)):
                out += theta[bounds[k]:bounds[k + 1]][J[k]]
            return out

        def _marg(w):
            return np.concatenate([np.bincount(j, weights=w, minlength=sizes[k])
                                   for k, j in enumerate(J)])

        def probs(theta):
            key = theta.tobytes()
            if state.get("key") != key:
                p, lz = _normalise(_gather(theta))
                state.update(key=key, lz=lz, p=p, m=_marg(p))
            return state

        def hv_moment(theta, v):
            s = probs(theta)
            return _marg(s["p"] * _gather(v))
    else:
        grid = np.empty((A,) * L)          # P at the current theta
        work = np.empty((A,) * L)          # scratch for Hessian-vector products

        def probs(theta):
            key = theta.tobytes()
            if state.get("key") != key:
                _broadcast_sum(L, cl, unpack(theta, fill), out=grid)
                _, lz = _normalise(grid)
                state.update(key=key, lz=lz, m=pack(marginals(grid, cl)))
            return state

        def hv_moment(theta, v):
            probs(theta)
            _broadcast_sum(L, cl, unpack(v, 0.0), out=work)
            np.multiply(work, grid, out=work)
            return pack(marginals(work, cl))

    def fun(theta):
        s = probs(theta)
        return s["lz"] - float(mu_vec @ theta) + 0.5 * reg * float(theta @ theta)

    def jac(theta):
        return probs(theta)["m"] - mu_vec + reg * theta

    def hessp(theta, v):
        """Cov_P[f, f.v] + 2 lam v, f the clique-cell indicators."""
        m = probs(theta)["m"]
        return hv_moment(theta, v) - m * float(m @ v) + reg * v

    def newton(theta):
        """Inexact Newton step d ~ -H^-1 g and squared Newton decrement -g.d.

        CG is Jacobi-preconditioned with the exact Hessian diagonal m(1-m) + 2 lam
        (floored at 1/N^2, the scale of one token squared, numerical guard), so cells
        with tiny marginals are resolved relative to their own sampling error, which is
        what the stopping rule measures. CG stops at a 1e-3 relative residual
        (numerical tolerance: the Hessian is singular along gauge directions, so
        tighter CG stagnates; the decrement is only needed to leading order)."""
        g = jac(theta)
        m = probs(theta)["m"]
        diag = np.maximum(m * (1.0 - m) + reg, 1.0 / N ** 2)
        H = LinearOperator((theta.size,) * 2, matvec=lambda v: hessp(theta, v))
        M = LinearOperator((theta.size,) * 2, matvec=lambda v: v / diag)
        d, _ = cg(H, -g, rtol=1e-3, maxiter=theta.size, M=M)
        return d, float(-g @ d)

    se = np.maximum(np.sqrt(mu_vec * (1.0 - mu_vec) / N), 1.0 / N)
    pw = counts / N
    lnp = np.log(pw)
    var = max(float((pw * lnp ** 2).sum() - (pw * lnp).sum() ** 2), 0.0)
    se_S = max(np.sqrt(var / N), (pw.size - 1) / (2.0 * N))

    def criteria(theta):
        m_ = float(np.abs(jac(theta) / se).max())
        if mode != "exact" or m_ >= resolution:
            return m_, (np.inf if mode == "exact" else 0.0), None
        d, dec = newton(theta)
        return m_, (0.5 * abs(dec) + abs(float(jac(theta) @ theta))) / se_S, d

    def stop(intermediate_result):
        m_, e_, _ = criteria(scale * intermediate_result.x)
        if m_ < resolution and e_ < resolution:
            raise StopIteration

    # The trust region works in rescaled variables phi = theta / scale with scale =
    # diag(H)^(-1/2) at the data, diag(H) ~ mu(1 - mu) + 2 lam (cells never observed,
    # map mode only, take the variance 1/N of a once-seen cell). Rare cells have tiny
    # curvature; in phi the Hessian has unit diagonal, so the Krylov subproblems need
    # far fewer products. The optimum and the stopping rule (in theta) are unchanged.
    scale = 1.0 / np.sqrt(np.maximum(mu_vec * (1.0 - mu_vec), 1.0 / N) + reg)
    res = minimize(lambda x: fun(scale * x), theta0 / scale,
                   jac=lambda x: scale * jac(scale * x),
                   hessp=lambda x, v: scale * hessp(scale * x, scale * v),
                   method="trust-krylov", callback=stop,
                   options={"gtol": 0.0, "maxiter": maxiter})
    theta = scale * res.x
    m_, e_, d = criteria(theta)
    # The trust region can stall once the objective's decrease falls below float64
    # rounding of |f| while the gradient is still accurate: finish with Newton steps,
    # backtracked by halving (at most BACKTRACK times) and accepted only if they
    # improve the worse of the two criteria.
    polish = 0
    while not (m_ < resolution and e_ < resolution) and polish < 30:
        if d is None:
            d, _ = newton(theta)
        worse = max(m_, e_ if np.isfinite(e_) else m_)
        for _ in range(BACKTRACK):
            new = theta + d
            if np.all(np.isfinite(new)):
                m2_, e2_, d2 = criteria(new)
                if max(m2_, e2_ if np.isfinite(e2_) else m2_) < worse:
                    break
            d = 0.5 * d
        else:
            break
        theta, m_, e_, d = new, m2_, e2_, d2
        polish += 1
    model = Potts(L, cl, unpack(theta, fill), sigma2 if mode == "map" else None)
    # forbidden cells carry zero mass in both, so free cells give the full mismatch
    mism = float(np.abs(probs(theta)["m"] - mu_vec).max())
    return model, {"converged": bool(m_ < resolution and e_ < resolution),
                   "iterations": int(res.nit), "newton_polish": polish,
                   "max_gradient_in_se": m_, "entropy_error_in_se": float(e_),
                   "resolution": resolution, "max_marginal_error": mism,
                   "n_params": int(theta.size), "message": str(res.message),
                   "backend": "dense" if J is None else "support",
                   "support": int(A ** L if J is None else J[0].size)}


# --------------------------------------------------------------------------- #
# prior selection and nested cross-validation (semantics of maxent.py)
# --------------------------------------------------------------------------- #

# Half-decade grid of prior widths, extended outward whenever an endpoint is selected
# so the optimum is always bracketed (the same convention as maxent.py).
SIGMA2_START = tuple(10.0 ** (k / 2) for k in range(-2, 3))
SIGMA2_STEP = 10.0 ** 0.5
SIGMA2_MAX_EXTENSIONS = 8


def cv_fold(states, counts, L, order, sigma2s, fold, n_folds=5, seed=0,
            cliques_=None) -> list[tuple[float, float]]:
    """Held-out bits per token of one fold, for each prior width (narrowest first,
    each fit warm-started from the previous)."""
    folds = split(counts, n_folds, np.random.default_rng(seed))
    test = folds[fold]
    train = counts - test
    out, init = [], None
    for s2 in sorted(sigma2s):
        m, _ = fit(states, train, L, order, "map", sigma2=s2, init=init, cliques_=cliques_)
        init = m
        out.append((float(s2), m.cross_entropy_bits(states, test)))
    return out


def aggregate_cv(per_fold) -> dict:
    """Mean and standard error over folds of the validation loss at each prior width."""
    grid = sorted({s for f in per_fold for s, _ in f})
    M = np.array([[dict(f)[s] for s in grid] for f in per_fold])
    mean = M.mean(axis=0)
    se = M.std(axis=0, ddof=1) / np.sqrt(M.shape[0])
    best = int(np.argmin(mean))
    return {"sigma2": grid, "bits": mean.tolist(), "se": se.tolist(),
            "best_sigma2": grid[best], "best_bits": float(mean[best]),
            "interior": 0 < best < len(grid) - 1}


def select_prior(states, counts, L, order, n_folds=4, seed=0, cliques_=None) -> dict:
    """K-fold CV choice of sigma2, extending the half-decade grid until interior."""
    grid = list(SIGMA2_START)
    per_fold = [cv_fold(states, counts, L, order, grid, k, n_folds, seed, cliques_)
                for k in range(n_folds)]
    agg = aggregate_cv(per_fold)
    for _ in range(SIGMA2_MAX_EXTENSIONS):
        if agg["interior"]:
            break
        lo = agg["best_sigma2"] == agg["sigma2"][0]
        new = agg["sigma2"][0] / SIGMA2_STEP if lo else agg["sigma2"][-1] * SIGMA2_STEP
        for k in range(n_folds):
            per_fold[k] = per_fold[k] + cv_fold(states, counts, L, order, [new], k,
                                                n_folds, seed, cliques_)
        agg = aggregate_cv(per_fold)
    return agg


def nested_cv_fold(states, counts, L, order, outer, n_outer=5, n_inner=4, seed=0,
                   cliques_=None) -> dict:
    """One outer fold of nested CV: sigma2 chosen by inner CV on the outer training
    tokens only, refit on them, scored on the outer test tokens."""
    folds = split(counts, n_outer, np.random.default_rng(seed))
    test = folds[outer]
    train = counts - test
    sel = select_prior(states, train, L, order, n_inner, seed + 1 + outer, cliques_)
    m, d = fit(states, train, L, order, "map", sigma2=sel["best_sigma2"], cliques_=cliques_)
    return {"outer": outer, "sigma2": sel["best_sigma2"], "interior": sel["interior"],
            "test_bits": m.cross_entropy_bits(states, test), "converged": d["converged"],
            "inner": sel}


def summarise_nested(folds: Sequence[dict]) -> dict:
    b = np.array([f["test_bits"] for f in folds])
    return {"bits": float(b.mean()), "se": float(b.std(ddof=1) / np.sqrt(b.size)),
            "per_fold": b.tolist(), "sigma2": [f["sigma2"] for f in folds],
            "all_interior": all(f["interior"] for f in folds),
            "all_converged": all(f["converged"] for f in folds)}

