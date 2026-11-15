"""Pairwise maximum-entropy models of long words (L >= 6), fitted without enumeration.

The model is the one of Stephens & Bialek (Phys. Rev. E 81, 066119, 2010;
arXiv:0801.0253) and of ``langfit.maxent`` / ``langfit.potts``: a word s of L letters
has probability

    P(s) = exp( sum_{i<j} V_ij(s_i, s_j) ) / Z,

one 26x26 tensor per pair of positions (single-site terms live inside the pair
tensors, exactly as in the frozen L = 4 study). For L >= 6 the 26^L states cannot be
enumerated at every optimisation step, so the three steps of the exact study are
replaced as follows.

Fitting: maximum pseudo-likelihood (Besag 1975), symmetric form of Ekeberg et al.
(PRE 87, 012707, 2013). Each position's conditional distribution given the other
L-1 letters is a 26-way softmax with logits eta_i(a|s) = sum_{j != i} V_ij(a, s_j), so
the pseudo-log-likelihood sum_i ln P(s_i | s_\\i) and its gradient are closed form and
cost O(n_types L^2 26) -- no partition function. With a Gaussian prior V ~ N(0, s2)
on every cell (the ``map`` mode of the exact study) it is minimised by L-BFGS.

Partition function: exact for L <= L_EXACT_MAX (8) by "split enumeration": the last
five positions are summed densely once (26^5 states), and the remaining outer
configurations are contracted against that tensor in batches; this is exact
enumeration reorganised so that each outer configuration costs one small matrix
product. The same pass carries the derivative with respect to inverse temperature,
which gives <ln w>_P and hence the exact model entropy. Each further letter multiplies
the cost by 26 (about two minutes at L = 8, an hour at L = 9 on one core), so beyond
L = 8 log Z comes from annealed importance sampling (Neal 2001) from the
independent-site model, with a delta-method error bar and an effective sample size.

Landscape: a word is a strict local minimum of E = -ln P under single-letter
substitution iff every letter is the unique argmax of its conditional logits. This is
checked word by word, and steepest descent (same rule and tie-break as
``maxent.descend``) runs from any set of starting words. Exhaustive search over 26^L
is impossible for L >= 6, so minima are *found*, never *counted*.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp

from . import maxent as me

A = me.A
LN2 = np.log(2.0)

# Number of trailing positions summed densely in split enumeration. 26^5 float64 is
# 95 MB per tensor (two are held); 26^6 would be 2.5 GB. A memory convention.
INNER = 5
# Exact enumeration is used up to this length by default: L = 8 takes about two
# minutes per partition function, L = 9 about an hour (26x more). Compute convention.
L_EXACT_MAX = 8
# Elements per vectorised block (types x L x 26), ~32 MB of float64. Memory convention.
BLOCK = 1 << 22


def pairs(L: int) -> list[tuple[int, int]]:
    return list(combinations(range(L), 2))


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #

@dataclass
class PairModel:
    """P(s) proportional to exp(sum_{i<j} J[p](s_i, s_j)), pairs p in combinations order.

    ``J`` has shape (L(L-1)/2, 26, 26); -inf marks a forbidden letter combination.
    """

    L: int
    J: np.ndarray
    sigma2: float | None = None

    @property
    def pairs(self) -> list[tuple[int, int]]:
        return pairs(self.L)

    def full(self) -> np.ndarray:
        """Coupling tensor laid out [j, b, i, a] = V_ij(a, b) (both orientations, zero
        diagonal), so that the logits of every position given letter b at j are one
        contiguous row: eta[:, i, a] = sum_j F[j, s_j, i, a]."""
        F = np.zeros((self.L, A, self.L, A))
        for p, (i, j) in enumerate(self.pairs):
            F[j, :, i, :] = self.J[p].T
            F[i, :, j, :] = self.J[p]
        return F

    def log_weight(self, states: np.ndarray) -> np.ndarray:
        """sum_{i<j} V_ij(s_i, s_j) for each row of ``states`` (nats, unnormalised)."""
        pi, pj = np.array(self.pairs).T
        out = np.empty(len(states))
        step = max(1, BLOCK // max(len(pi), 1))
        for k in range(0, len(states), step):
            s = states[k:k + step]
            out[k:k + step] = self.J[np.arange(len(pi)), s[:, pi], s[:, pj]].sum(axis=1)
        return out

    def logits(self, states: np.ndarray, F: np.ndarray | None = None) -> np.ndarray:
        """Conditional logits eta (n, L, 26) of every position given the others."""
        F = self.full() if F is None else F
        eta = F[0][states[:, 0]].copy()
        for j in range(1, self.L):
            eta += F[j][states[:, j]]
        return eta

    def dense_log_weights(self) -> np.ndarray:
        """All 26^L unnormalised log-weights (only for L <= INNER)."""
        if self.L > INNER:
            raise ValueError("dense grid only for L <= %d" % INNER)
        lw = np.zeros((A,) * self.L)
        for p, (i, j) in enumerate(self.pairs):
            shape = [1] * self.L
            shape[i] = shape[j] = A
            lw = lw + self.J[p].reshape(shape)
        return lw

    def to_V(self) -> list[np.ndarray]:
        """Pair tensors in the layout of ``maxent.MaxEnt`` / ``potts.Potts``."""
        return [self.J[p].copy() for p in range(len(self.pairs))]


@dataclass
class Independent:
    """Independent-site model; ``logq`` (L, 26) are normalised log-probabilities."""

    logq: np.ndarray

    @property
    def L(self) -> int:
        return self.logq.shape[0]

    def log_prob(self, states: np.ndarray) -> np.ndarray:
        return self.logq[np.arange(self.L), states].sum(axis=1)

    def entropy_bits(self) -> float:
        q = np.exp(self.logq)
        m = q > 0
        return float(-(q[m] * self.logq[m]).sum() / LN2)


def site_counts(states: np.ndarray, counts: np.ndarray, L: int) -> np.ndarray:
    n = np.zeros((L, A))
    for i in range(L):
        n[i] = np.bincount(states[:, i], weights=counts, minlength=A)
    return n


def pair_counts(states: np.ndarray, counts: np.ndarray, L: int) -> np.ndarray:
    out = np.empty((L * (L - 1) // 2, A, A))
    for p, (i, j) in enumerate(pairs(L)):
        out[p] = np.bincount(states[:, i] * A + states[:, j], weights=counts,
                             minlength=A * A).reshape(A, A)
    return out


def fit_independent(states: np.ndarray, counts: np.ndarray, L: int,
                    sigma2: float | None = None) -> Independent:
    """Order-1 maximum entropy model. ``sigma2=None``: the exact (S&B) closed form,
    log of the empirical site marginals (-inf for unused letters). Otherwise the MAP
    fit with prior v ~ N(0, sigma2) on each of the 26 log-potentials per site, the
    order-1 analogue of ``maxent.fit(..., "l2")``."""
    n = site_counts(states, counts, L)
    N = float(counts.sum())
    if sigma2 is None:
        with np.errstate(divide="ignore"):
            return Independent(np.log(n / N))
    lam = me.prior_lambda(sigma2, N)

    def f(v):
        v = v.reshape(L, A)
        lse = logsumexp(v, axis=1)
        val = -(n * (v - lse[:, None])).sum() / N + lam * (v * v).sum()
        g = -(n - N * np.exp(v - lse[:, None])) / N + 2 * lam * v
        return val, g.ravel()

    res = minimize(f, np.zeros(L * A), jac=True, method="L-BFGS-B",
                   options={"gtol": 0.0, "ftol": 0.0, "maxiter": 10000})
    v = res.x.reshape(L, A)
    return Independent(v - logsumexp(v, axis=1)[:, None])


# --------------------------------------------------------------------------- #
# pseudo-likelihood
# --------------------------------------------------------------------------- #

class _PLData:
    """Training data in vectorised blocks. Each block carries its one-hot design
    X (n, L*26), X[t, j*26 + b] = [s_tj = b], so that the conditional logits of all
    positions are one sparse-dense product, eta = X F, and the gradient scatter of
    the conditional residuals R is another, X^T R."""

    def __init__(self, states: np.ndarray, counts: np.ndarray, L: int):
        keep = counts > 0
        self.states = states[keep]
        self.counts = counts[keep]
        self.L = L
        self.N = float(self.counts.sum())
        step = max(1, BLOCK // (L * A))
        self.blocks = []
        for k in range(0, len(self.states), step):
            s = self.states[k:k + step]
            c = self.counts[k:k + step]
            n = len(s)
            cols = (s + A * np.arange(L)).ravel()
            X = sparse.csr_matrix((np.ones(n * L), cols, np.arange(0, n * L + 1, L)),
                                  shape=(n, L * A))
            self.blocks.append((s, c, X, X.T.tocsr()))


def _lse(eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """log-sum-exp over the last axis and the softmax, sharing one exp pass."""
    mx = eta.max(axis=-1, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    e = np.exp(eta - mx)
    z = e.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore"):
        return (np.log(z) + mx)[..., 0], e / z


def pseudo_loglik(model: PairModel, states: np.ndarray, counts: np.ndarray,
                  per_position: bool = False):
    """Pseudo-log-likelihood in bits per token, sum_i log2 P(s_i | s_\\i), averaged over
    tokens (optionally the per-position terms)."""
    F = model.full()
    tot = np.zeros(model.L)
    keep = counts > 0
    st, ct = states[keep], counts[keep]
    step = max(1, BLOCK // (model.L * A))
    for k in range(0, len(st), step):
        s, c = st[k:k + step], ct[k:k + step]
        eta = model.logits(s, F)
        obs = np.take_along_axis(eta, s[:, :, None], axis=2)[..., 0]
        tot += (c[:, None] * (obs - logsumexp(eta, axis=2))).sum(axis=0)
    tot /= ct.sum() * LN2
    return tot if per_position else float(tot.sum())


def _plm_objective(theta, free, L, data: _PLData, lam):
    """Negative pseudo-log-likelihood per token plus lam * |theta|^2, and gradient."""
    J = np.full((L * (L - 1) // 2, A, A), -np.inf)
    J[free] = theta
    model = PairModel(L, J)
    F = model.full()
    G = np.zeros((L, A, L * A))                       # [j, b, (i, a)]
    pll = 0.0
    Ff = F.reshape(L * A, L * A)
    # Dense BLAS products are an order of magnitude faster than sparse ones here, but
    # 0 * (-inf) is undefined, so models with forbidden cells use the sparse path.
    dense = bool(np.isfinite(Ff).all())
    for s, c, X, XT in data.blocks:
        Xd = X.toarray() if dense else X
        eta = (Xd @ Ff).reshape(len(s), L, A)
        lse, R = _lse(eta)
        obs = np.take_along_axis(eta, s[:, :, None], axis=2)[..., 0]
        pll += float((c[:, None] * (obs - lse)).sum())
        R *= -c[:, None, None]
        np.put_along_axis(R, s[:, :, None],
                          np.take_along_axis(R, s[:, :, None], axis=2) + c[:, None, None], axis=2)
        R2 = R.reshape(len(s), L * A)
        G += ((Xd.T @ R2) if dense else (XT @ R2)).reshape(L, A, L * A)
    G = G.reshape(L, A, L, A)
    grad = np.empty_like(J)
    for p, (i, j) in enumerate(pairs(L)):
        grad[p] = G[j, :, i, :].T + G[i, :, j, :]
    f = -pll / data.N + lam * float(theta @ theta)
    g = -grad[free] / data.N + 2.0 * lam * theta
    return f, g


def fit_plm(states: np.ndarray, counts: np.ndarray, L: int, sigma2: float | None = None,
            mask: np.ndarray | None = None, init: PairModel | None = None,
            resolution: float = 0.01, maxiter: int = 20000) -> tuple[PairModel, dict]:
    """Maximum pseudo-likelihood fit of the pairwise model.

    ``sigma2``: Gaussian prior width (lam = 1/(2 sigma2 N), as in ``maxent.prior_lambda``);
    None for the unregularised estimator. ``mask``: boolean (P, 26, 26) of free cells;
    the rest are forbidden (-inf), the analogue of S&B's ``exact`` mode when ``mask`` is
    the set of pair cells seen in training.

    Stopping rule, relative to sampling noise as in ``maxent.fit``: the gradient of the
    per-token objective in cell (p, a, b) is a difference of two pair frequencies (the
    empirical one, counted twice, and the two conditionals' predictions), so each
    component is compared with 2 se_c, se_c = max(sqrt(mu (1-mu) / N), 1/N), and the fit
    stops when every component is below ``resolution`` (a fraction of one standard
    error) of it.
    """
    data = _PLData(states, counts, L)
    P = L * (L - 1) // 2
    free = np.ones((P, A, A), dtype=bool) if mask is None else mask.astype(bool)
    lam = 0.0 if sigma2 is None else me.prior_lambda(sigma2, data.N)
    mu = pair_counts(data.states, data.counts, L)[free] / data.N
    scale = 2.0 * np.maximum(np.sqrt(mu * (1.0 - mu) / data.N), 1.0 / data.N)
    theta0 = np.zeros(int(free.sum()))
    if init is not None:
        theta0 = np.where(np.isfinite(init.J[free]), init.J[free], 0.0)
    state = {"worst": np.inf}

    def fun(theta):
        f, g = _plm_objective(theta, free, L, data, lam)
        state["worst"] = float(np.abs(g / scale).max())
        return f, g

    def stop(intermediate_result):
        if state["worst"] < resolution:
            raise StopIteration

    res = minimize(fun, theta0, jac=True, method="L-BFGS-B", callback=stop,
                   options={"gtol": 0.0, "ftol": 0.0, "maxiter": maxiter,
                            "maxfun": 2 * maxiter, "maxcor": 30})
    _, g = _plm_objective(res.x, free, L, data, lam)
    worst = float(np.abs(g / scale).max())
    J = np.full((P, A, A), -np.inf)
    J[free] = res.x
    return PairModel(L, J, sigma2), {
        "converged": worst < resolution, "max_gradient_in_se": worst,
        "resolution": resolution, "iterations": int(res.nit), "evaluations": int(res.nfev),
        "pll_bits": float(-_plm_objective(res.x, free, L, data, 0.0)[0] / LN2),
        "message": str(res.message)}


# --------------------------------------------------------------------------- #
# partition function and entropy
# --------------------------------------------------------------------------- #

def _expect(lw: np.ndarray) -> tuple[float, float]:
    lz = float(logsumexp(lw))
    p = np.exp(lw - lz)
    m = p > 0
    return lz, float((p[m] * lw[m]).sum())


def exact_log_partition(model: PairModel, batch: int = A, max_L: int = L_EXACT_MAX) -> dict:
    """Exact ln Z and <ln w>_P (hence the entropy) by split enumeration, L <= L_EXACT_MAX.

    Positions are split into outer O = {0..L-6} and inner I = {L-5..L-1}. With
    B(s_I) the inner pair sum, h_k(b | s_O) = sum_{o in O} V_ok(s_o, b) the field the
    outer letters exert on inner position k, and E_OO(s_O) the outer pair sum,

        Z = sum_{s_O} exp(E_OO) sum_{s_I} exp(B(s_I)) prod_k exp(h_k(s_k | s_O)).

    The inner sum is a multilinear contraction of the fixed 26^5 tensor exp(B) with
    five vectors per outer configuration, done axis by axis for a batch of outer
    configurations at once (the first axis as a matrix product). Every term of the sum
    is included; only the order of summation differs from brute force. The same
    contraction carried with first-order dual numbers gives sum_s w(s) ln w(s), so the
    entropy S = ln Z - <ln w> is exact too. Shifts by the maxima keep all exponentials
    at most 1.
    """
    L = model.L
    if L <= INNER:
        lz, mlw = _expect(model.dense_log_weights().ravel())
        return {"logZ": lz, "mean_lw": mlw, "entropy_bits": (lz - mlw) / LN2, "exact": True}
    if L > max_L:
        raise ValueError(f"exact enumeration capped at L = {max_L}")
    m = L - INNER
    prs = model.pairs
    inner = PairModel(INNER, np.stack([model.J[p] for p, (i, j) in enumerate(prs) if i >= m]))
    B = inner.dense_log_weights()
    bmax = float(B[np.isfinite(B)].max())
    W = np.exp(B - bmax)
    dW = np.where(W > 0, W * np.where(np.isfinite(B), B, 0.0), 0.0)
    del B
    outer_pairs = [(p, i, j) for p, (i, j) in enumerate(prs) if j < m]
    cross = {k: [(p, i) for p, (i, j) in enumerate(prs) if j == k and i < m]
             for k in range(m, L)}
    grid = np.indices((A,) * m).reshape(m, -1).T            # all outer configurations
    logZ_parts, lw_parts = [], []
    for b0 in range(0, len(grid), batch):
        so = grid[b0:b0 + batch]
        C = len(so)
        eoo = np.zeros(C)
        for p, i, j in outer_pairs:
            eoo += model.J[p][so[:, i], so[:, j]]
        h = np.zeros((L - m, C, A))
        for k in range(m, L):
            for p, i in cross[k]:
                h[k - m] += model.J[p][so[:, i]]
        with np.errstate(invalid="ignore"):
            hmax = h.max(axis=2)                              # (5, C)
            u = np.exp(h - hmax[..., None])
        u = np.where(np.isfinite(hmax)[..., None], u, 0.0)
        du = np.where(u > 0, u * np.where(np.isfinite(h), h, 0.0), 0.0)
        # contract the last inner axis first: (26^4, 26) @ (26, C)
        T = W.reshape(-1, A) @ u[-1].T
        dT = dW.reshape(-1, A) @ u[-1].T + W.reshape(-1, A) @ du[-1].T
        for k in range(INNER - 2, -1, -1):
            T3 = T.reshape(-1, A, C)
            dT3 = dT.reshape(-1, A, C)
            T = np.einsum("xbc,cb->xc", T3, u[k])
            dT = np.einsum("xbc,cb->xc", dT3, u[k]) + np.einsum("xbc,cb->xc", T3, du[k])
        T, dT = T.ravel(), dT.ravel()
        shift = eoo + bmax + hmax.sum(axis=0)
        ok = np.isfinite(shift) & (T > 0)
        logZ_parts.append(np.where(ok, shift + np.log(np.where(ok, T, 1.0)), -np.inf))
        lw_parts.append(np.where(ok, np.where(np.isfinite(eoo), eoo, 0.0)
                                 + dT / np.where(ok, T, 1.0), 0.0))
    lzo = np.concatenate(logZ_parts)
    mlo = np.concatenate(lw_parts)
    lz = float(logsumexp(lzo))
    po = np.exp(lzo - lz)
    mlw = float((po * mlo).sum())
    return {"logZ": lz, "mean_lw": mlw, "entropy_bits": (lz - mlw) / LN2, "exact": True}


def ais_log_partition(model: PairModel, base: Independent, n_chains: int, n_temps: int,
                      seed: int = 0) -> dict:
    """Annealed importance sampling (Neal 2001) for ln Z of a finite-coupling model.

    Path: f_beta = q0^(1-beta) w^beta from the normalised independent-site model q0
    (Z_0 = 1) to the pairwise weights w, beta on a uniform grid of ``n_temps`` steps;
    transition at each beta: one systematic-scan Gibbs sweep that leaves f_beta
    invariant. Returns ln Z_hat = ln mean(w_k) over chains, its delta-method standard
    error sd(w)/(mean(w) sqrt(K)), the effective sample size (sum w)^2 / sum w^2, and
    the self-normalised estimate of <ln w>_P (hence the entropy).

    Z_hat is unbiased, so ln Z_hat is biased low (Jensen); the delta-method error is
    itself unreliable when the weight distribution is heavy-tailed (small ESS), which is
    exactly the failure mode on a multimodal landscape. Hence the validation against
    ``exact_log_partition`` for L <= 9 in the study.
    """
    if not np.all(np.isfinite(model.J)):
        raise ValueError("AIS path needs finite couplings (forbidden cells kill chains)")
    rng = np.random.default_rng(seed)
    L, K = model.L, n_chains
    F = model.full()
    q = np.exp(base.logq)
    cdf = np.cumsum(q, axis=1)
    x = np.stack([np.minimum(np.searchsorted(cdf[i], rng.random(K) * cdf[i, -1]), A - 1)
                  for i in range(L)], axis=1)
    lw = model.log_weight(x)
    lq = base.log_prob(x)
    logw = np.zeros(K)
    betas = np.linspace(0.0, 1.0, n_temps + 1)
    rows = np.arange(K)
    for t in range(1, n_temps + 1):
        b = betas[t]
        logw += (b - betas[t - 1]) * (lw - lq)
        for i in range(L):
            eta = np.zeros((K, A))
            for j in range(L):
                if j != i:
                    eta += F[j, :, i, :][x[:, j]]
            g = b * eta + (1.0 - b) * base.logq[i] + rng.gumbel(size=(K, A))
            new = g.argmax(axis=1)
            # incremental update of ln w and ln q0 for the changed letter
            d = eta[rows, new] - eta[rows, x[:, i]]
            lw += d
            lq += base.logq[i, new] - base.logq[i, x[:, i]]
            x[:, i] = new
    lz = float(logsumexp(logw) - np.log(K))
    w = np.exp(logw - logw.max())
    se = float(w.std(ddof=1) / w.mean() / np.sqrt(K))
    ess = float(w.sum() ** 2 / (w * w).sum())
    wn = w / w.sum()
    mlw = float((wn * lw).sum())
    # delta-method sd of the self-normalised mean, and of the entropy estimate
    # (ln Z_hat and <ln w>_hat come from the same weights; their errors are combined
    # conservatively by adding in quadrature as if independent)
    se_mlw = float(np.sqrt((wn * wn * (lw - mlw) ** 2).sum()))
    return {"logZ": lz, "se": se, "ess": ess, "n_chains": K, "n_temps": n_temps,
            "mean_lw": mlw, "se_mean_lw": se_mlw,
            "entropy_bits": (lz - mlw) / LN2,
            "entropy_se_bits": float(np.hypot(se, se_mlw) / LN2), "exact": False,
            "samples": x, "weights": wn}


# --------------------------------------------------------------------------- #
# landscape
# --------------------------------------------------------------------------- #

def is_strict_minimum(model: PairModel, states: np.ndarray) -> np.ndarray:
    """True where every single-letter substitution strictly raises E = -ln w, i.e. each
    letter is the unique maximiser of its conditional logits (S&B's definition, the
    per-word form of ``maxent.local_minima``)."""
    F = model.full()
    out = np.empty(len(states), dtype=bool)
    step = max(1, BLOCK // (model.L * A))
    for k in range(0, len(states), step):
        s = states[k:k + step]
        eta = model.logits(s, F)
        own = np.take_along_axis(eta, s[:, :, None], axis=2)
        np.put_along_axis(eta, s[:, :, None], -np.inf, axis=2)
        out[k:k + step] = np.all(own[..., 0] > eta.max(axis=2), axis=1) & np.isfinite(own[..., 0]).all(axis=1)
    return out


def descend(model: PairModel, states: np.ndarray) -> np.ndarray:
    """Steepest descent under single-letter substitution from each starting word, with
    ``maxent.descend``'s rule: move to the lowest-energy neighbour if strictly lower;
    ties to the earliest position, then the earliest letter. Returns terminal words.

    From a finite-weight word the energy change of a substitution at i is the change of
    the conditional logit eta_i. From a forbidden word (some pair cell -inf) it is not:
    the neighbour may still contain another forbidden pair. There the neighbour's full
    log-weight R_i(s) + eta_i(a | s) is used, R_i being the sum of the pairs that do not
    involve i, and the word moves only to a finite-weight neighbour (dense reference:
    every finite energy is below +inf; a word whose whole neighbourhood is forbidden
    stays put)."""
    F = model.full()
    x = states.copy()
    active = np.ones(len(x), dtype=bool)
    L = model.L
    pi, pj = np.array(model.pairs).T
    touches = np.stack([(pi == i) | (pj == i) for i in range(L)])      # (L, P)
    while active.any():
        idx = np.flatnonzero(active)
        s = x[idx]
        eta = model.logits(s, F)
        own = np.take_along_axis(eta, s[:, :, None], axis=2)
        lw_s = model.log_weight(s)
        bad = ~np.isfinite(lw_s)
        gain = np.empty_like(eta)
        ok = ~bad
        gain[ok] = eta[ok] - own[ok]                       # ln w(neighbour) - ln w(s)
        if bad.any():
            sb = s[bad]
            vals = model.J[np.arange(len(pi)), sb[:, pi], sb[:, pj]]   # (nb, P)
            R = np.stack([np.where(touches[i], 0.0, vals).sum(axis=1) for i in range(L)],
                         axis=1)                                         # (nb, L)
            nb_lw = R[:, :, None] + eta[bad]
            # any finite neighbour is an infinite improvement; rank them by log-weight
            gain[bad] = np.where(np.isfinite(nb_lw), nb_lw, -np.inf)
        np.put_along_axis(gain, s[:, :, None], -np.inf, axis=2)
        flat = gain.reshape(len(s), L * A)
        best = flat.argmax(axis=1)
        top = flat[np.arange(len(s)), best]
        moved = np.where(bad, np.isfinite(top), top > 0)
        pos, let = np.divmod(best[moved], A)
        s_m = s[moved]
        s_m[np.arange(len(s_m)), pos] = let
        x[idx[moved]] = s_m
        active[idx[~moved]] = False
    return x


def words_of(states: np.ndarray) -> list[str]:
    alph = np.array(list(me.ALPHABET))
    return ["".join(r) for r in alph[states]]


# --------------------------------------------------------------------------- #
# model selection
# --------------------------------------------------------------------------- #

def select_sigma2(states: np.ndarray, train: np.ndarray, val: np.ndarray, L: int,
                  score: str = "pll", resolution: float = 0.01) -> dict:
    """Choose the prior width by validation score on a held-out token split, on the
    half-decade grid of ``maxent`` (``SIGMA2_START``), extended outward until the optimum
    is interior. ``score``: "pll" (held-out pseudo-log-likelihood, always computable) or
    "ll" (held-out log-likelihood with exact ln Z, L <= L_EXACT_MAX). Scores are bits
    per token, higher is better; fits are warm-started from the nearest fitted width."""
    fitted: dict[float, PairModel] = {}
    scores: dict[float, float] = {}
    diags: dict[float, dict] = {}

    def run(s2, init):
        m, d = fit_plm(states, train, L, s2, init=init, resolution=resolution)
        fitted[s2], diags[s2] = m, d
        if score == "pll":
            scores[s2] = pseudo_loglik(m, states, val)
        else:
            lz = exact_log_partition(m)["logZ"]
            keep = val > 0
            scores[s2] = float((val[keep] * (m.log_weight(states[keep]) - lz)).sum()
                               / val[keep].sum() / LN2)

    init = None
    for s2 in sorted(me.SIGMA2_START):
        run(s2, init)
        init = fitted[s2]
    for _ in range(me.SIGMA2_MAX_EXTENSIONS):
        grid = sorted(scores)
        best = max(grid, key=scores.get)
        if 0 < grid.index(best) < len(grid) - 1:
            break
        if best == grid[0]:
            new, init = grid[0] / me.SIGMA2_STEP, fitted[grid[0]]
        else:
            new, init = grid[-1] * me.SIGMA2_STEP, fitted[grid[-1]]
        run(new, init)
    grid = sorted(scores)
    best = max(grid, key=scores.get)
    return {"sigma2": grid, "score_bits": [scores[s] for s in grid], "score": score,
            "best_sigma2": best, "interior": 0 < grid.index(best) < len(grid) - 1,
            "converged": [diags[s]["converged"] for s in grid],
            "model": fitted[best]}


def select_sigma2_independent(states, train, val, L) -> dict:
    """Same grid protocol for the order-1 MAP baseline, scored by held-out log-loss."""
    scores = {}

    def run(s2):
        m = fit_independent(states, train, L, s2)
        keep = val > 0
        scores[s2] = float((val[keep] * m.log_prob(states[keep])).sum() / val[keep].sum() / LN2)

    for s2 in me.SIGMA2_START:
        run(s2)
    for _ in range(me.SIGMA2_MAX_EXTENSIONS):
        grid = sorted(scores)
        best = max(grid, key=scores.get)
        if 0 < grid.index(best) < len(grid) - 1:
            break
        run(grid[0] / me.SIGMA2_STEP if best == grid[0] else grid[-1] * me.SIGMA2_STEP)
    grid = sorted(scores)
    best = max(grid, key=scores.get)
    return {"sigma2": grid, "score_bits": [scores[s] for s in grid], "best_sigma2": best,
            "interior": 0 < grid.index(best) < len(grid) - 1}


# --------------------------------------------------------------------------- #
# comparisons
# --------------------------------------------------------------------------- #

def double_centre(V: np.ndarray) -> np.ndarray:
    """Gauge-invariant part of a finite pair tensor: V - row mean - column mean + mean.

    Moving a single-site term between pair tensors that share a position adds a
    function of one letter only, which double-centring removes; what is left is the
    interaction proper (the zero-sum or Ising gauge of the coupling)."""
    return V - V.mean(axis=0, keepdims=True) - V.mean(axis=1, keepdims=True) + V.mean()


def coupling_agreement(Va: Sequence[np.ndarray], Vb: Sequence[np.ndarray]) -> dict:
    a = np.concatenate([double_centre(v).ravel() for v in Va])
    b = np.concatenate([double_centre(v).ravel() for v in Vb])
    return {"pearson": float(np.corrcoef(a, b)[0, 1]),
            "relative_l2": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
            "slope": float((a @ b) / (b @ b)),
            "rms_b": float(np.sqrt((b * b).mean()))}
