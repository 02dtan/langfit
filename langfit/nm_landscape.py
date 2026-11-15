"""Grand-canonical energy landscape of the variable-length pairwise maximum-entropy model.

This module produces the visualisation data for the *new* (augmented) model, distinct
from the frozen four-letter Stephens--Bialek study in ``langfit.maxent``.  It places
words of every modelled length L = 1..8 on one grand-canonical energy scale and views
the result through a length-invariant projection.

Model
-----
For each length L the word distribution is the pairwise maximum-entropy / Potts model

    P_L(s) = exp( sum_{i<j} V^L_ij(s_i, s_j) ) / Z_L,      s in {a..z}^L,

fitted with a Gaussian prior V ~ N(0, sigma2) on every coupling cell:

  * L <= 5 : exact MAP by dense enumeration (``langfit.potts``, mode "map");
  * 6 <= L <= 8 : maximum pseudo-likelihood (``langfit.longwords``) with the exact
    log Z from split enumeration (L <= L_EXACT_MAX = 8);
  * L = 1 : there are no position pairs, so the model is uniform over 26 letters.

The lengths are joined on one grand-canonical scale (as in ``langfit.edits``):

    E(w) = -log2 P(L(w)) - log2 P_L(w)(w) = H_L(w) - mu_L,

    H_L(w) = -sum_{i<j} V^L_ij(w_i, w_j) / ln 2,
    mu_L   = log2 P(L) - log2 Z_L.

P(L) is the posterior mean under a symmetric Dirichlet(1, ..., 1) prior over the eight
modelled lengths, so with no data (the empty frame) P(L) = 1/8 and, with V = 0, every
length block is uniform: the prior model.

Move set (``langfit.edits`` conventions)
----------------------------------------
substitution, adjacent transposition, insertion, deletion.  The modelled space is the
union of the 26^L grids for L = 1..8.  Boundary: a length-1 word has no deletion move
(L = 0 is not modelled) and a length-8 word has no insertion move (L = 9 is not
modelled); all other moves stay inside the space.

Projection
----------
Each word is represented by its *boundary-marked bigram profile*: the count vector of
the 27^2 bigrams over the 27 symbols {a..z, "#"} (where "#" marks the left and right
boundary).  A word w_1..w_L has the L + 1 bigrams "#w_1", "w_1w_2", ..., "w_{L-1}w_L",
"w_L#".  The profile is length-invariant and the coordinate of a word along any linear
axis u is a sum over its L + 1 bigrams,

    x(w) = sum_{k=0}^{L} u_x[ w_k, w_{k+1} ]   (w_0 = w_{L+1} = "#"),

which is what lets us treat every length on the same footing.  The two axes are the
top two principal axes of the bigram-profile covariance under the *final* grand-
canonical model P(w) = P(L) P_L(w).  The covariance is estimated by importance
sampling (unbiased; the same sampler used for the bin probabilities), because a closed
form would need marginals of order four.

Height
------
The height is the bin energy (Zwanzig 1954 exponential average) over *all* strings of
all modelled lengths that project into a cell:

    E_bin = -log2 ( P(bin) / |bin| ),

    P(bin) = sum_L P(L) sum_{s in bin, |s| = L} P_L(s),
    |bin|  = sum_L |{ s in bin : |s| = L }|.

For L <= 5 every string is enumerated, so P(bin | L) and |bin|_L are exact.  For
6 <= L <= 8 (26^8 ~ 2.1e11) the strings cannot be enumerated: P(bin | L) is estimated
by importance sampling from a mixture of the independent-letter model and the
empirical word distribution, using the exact Z_L (unbiased, with per-cell standard
errors), and |bin|_L is estimated by uniform sampling (unbiased, with standard errors).
Per-cell uncertainty is reported and low-confidence cells are masked.

The displayed height is relative to the *prior* (empty-data) model, so that the empty
frame is exactly flat:

    z_bin = E_bin - E_bin^prior = log2 P_prior(bin) - log2 P(bin),

    P_prior(bin) = sum_L pi_L |bin|_L / 26^L,   pi_L = 1/8 (Dirichlet prior mean).

Wells
-----
Strict string-level minima are minima of E on the move graph (``edits``).  Projected
wells are local minima of the z_bin cell surface with cell adjacency given by the move
set, ranked by 0-dimensional sublevel-set persistence (``langfit.projection``).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from . import edits, longwords, potts, projection as pj

ALPHABET = edits.ALPHABET            # "abc..z"
A = len(ALPHABET)                    # 26
BOUNDARY = "#"
SYMBOLS = ALPHABET + BOUNDARY
S = len(SYMBOLS)                     # 27
N_BIGRAMS = S * S                    # 729
BOUNDARY_IDX = S - 1                 # 26

L_MIN = 1
L_MAX = 8
LENGTHS = tuple(range(L_MIN, L_MAX + 1))
MOVES = edits.MOVES                  # ("sub", "trans", "del", "ins")

LN2 = np.log(2.0)

# Display / estimation parameters (they change the picture, not the model):
GRID = 32                            # bins per axis of the projection grid
PERSIST_BITS = 1.0                   # label wells with persistence >= 1 bit
TOP_STRING_MINIMA = 200              # string-level minima emitted (overlay shows a subset)
BASIN_WORDS = 3                      # most probable words listed per labelled well
CENTIBITS = 100                      # quantisation of trajectory surfaces: 0.01 bit
MIXTURE_WEIGHT = 0.5                 # proposal = 0.5 * independent + 0.5 * empirical
N_IS_PER_LENGTH = 2_000_000          # importance samples per length (L = 6..8)
N_UNIFORM_PER_LENGTH = 2_000_000     # uniform samples per length for |bin| (L = 6..8)
N_BASIS_PER_LENGTH = 400_000         # importance samples per length for the basis fit
DIRICHLET_ALPHA = 1.0                # symmetric Dirichlet prior strength for P(L)
ESS_THRESHOLD = 100.0                # ESS below which a length's estimates are unreliable


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def load_monthly(name: str = "twitter_month_L1-8_all.pkl.gz") -> dict:
    from . import social
    return social.load_periods(name)


def monthly_counters(name: str = "twitter_month_L1-8_all.pkl.gz"):
    """(sorted month keys, [Counter per month over all modelled lengths])."""
    P = load_monthly(name)["periods"]
    months = sorted(P)
    out = []
    for k in months:
        c = Counter()
        for L, d in P[k]["counts"].items():
            if L in LENGTHS:
                c.update({w: n for w, n in d.items() if len(w) == L})
        out.append(c)
    return months, out


def cumulative_frames(name: str = "twitter_month_L1-8_all.pkl.gz", period: str = "year"):
    """Cumulative frames: the empty frame plus one frame per cumulative threshold.

    ``period`` groups months ("month" | "quarter" | "year"); a frame is emitted
    whenever the month key's period changes.  Returns (labels, frames, per_length)
    with ``frames[t]`` a per-length Counter of cumulative counts and ``per_length[t]``
    a dict {L: tokens}.  ``labels[t]`` is the month key of the last month in frame t
    ("(none)" for the empty frame).
    """
    months, ctrs = monthly_counters(name)
    tot = Counter()
    labels = ["(none)"]
    frames = [Counter()]
    per_length = [{L: 0 for L in LENGTHS}]
    last = None
    for m, c in zip(months, ctrs):
        tot.update(c)
        key = _period_key(m, period)
        if key == last:
            frames[-1] = tot.copy()
            labels[-1] = m
        else:
            frames.append(tot.copy())
            labels.append(m)
            last = key
        per_length.append({L: sum(n for w, n in tot.items() if len(w) == L)
                           for L in LENGTHS})
    return labels, frames, per_length


def _period_key(m: str, period: str) -> str:
    if period == "month":
        return m
    if period == "quarter":
        return m[:4] + {"01": "Q1", "02": "Q1", "03": "Q1", "04": "Q2", "05": "Q2",
                        "06": "Q2", "07": "Q3", "08": "Q3", "09": "Q3"}.get(m[5:7], "Q4")
    return m[:4]                       # year


def length_prior(counts_by_length: dict[int, float], alpha: float = DIRICHLET_ALPHA
                 ) -> np.ndarray:
    """P(L) = (n_L + alpha) / (N + alpha * |L|), the posterior mean of a symmetric
    Dirichlet(alpha) prior over the modelled lengths.  Index 0 corresponds to L = 1."""
    n = np.array([float(counts_by_length.get(L, 0.0)) for L in LENGTHS])
    return (n + alpha) / (n.sum() + alpha * len(LENGTHS))


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #

@dataclass
class LengthModel:
    """A fitted per-length pairwise model plus its exact partition function."""
    L: int
    backend: str                      # "uniform", "potts" or "plm"
    sigma2: float | None
    V: list | None = None             # potts: list of (A, A) pair tensors
    J: np.ndarray | None = None       # plm: (L(L-1)/2, A, A)
    logZ: float = 0.0                 # nats
    entropy_bits: float = 0.0

    def _potts(self):
        return potts.Potts(self.L, potts.cliques(2, self.L),
                           [v.copy() for v in self.V], self.sigma2)

    def _plm(self):
        return longwords.PairModel(self.L, self.J, self.sigma2)

    def dense_log_weights(self) -> np.ndarray | None:
        """Dense A^L log-weights (nats) for L <= 5, else None."""
        if self.L > 5:
            return None
        if self.backend == "uniform":
            return np.zeros((A,) * self.L)
        return self._potts().log_weights()

    def log_weight(self, states: np.ndarray) -> np.ndarray:
        """Unnormalised log-weight (nats) per row of ``states`` (n, L)."""
        if self.backend == "uniform":
            return np.zeros(len(states))
        if self.backend == "potts":
            lw = self.dense_log_weights().ravel()
            return lw[potts.flat_index(states, self.L)]
        return self._plm().log_weight(states)

    def log2_prob(self, states: np.ndarray) -> np.ndarray:
        """log2 P(s | L) per row of ``states`` (n, L)."""
        return (self.log_weight(states) - self.logZ) / LN2


def _fit_uniform(L: int) -> LengthModel:
    return LengthModel(L, "uniform", None, logZ=float(L * np.log(A)),
                       entropy_bits=float(L * np.log2(A)))


def fit_length(states: np.ndarray, counts: np.ndarray, L: int, sigma2: float | None,
               init: LengthModel | None = None) -> tuple[LengthModel, dict]:
    """Fit the pairwise model of length L (exact potts for L <= 5, PLM for 6..8)."""
    if L == 1:
        return _fit_uniform(1), {"converged": True, "uniform": True}
    if float(np.asarray(counts).sum()) == 0.0:
        # empty frame: the MAP estimate under the centred Gaussian prior is V = 0,
        # i.e. the uniform model within this length.
        if L <= 5:
            V = [np.zeros((A, A)) for _ in potts.cliques(2, L)]
            lm = LengthModel(L, "potts", sigma2, V=V, logZ=float(L * np.log(A)),
                             entropy_bits=float(L * np.log2(A)))
        else:
            J = np.zeros((L * (L - 1) // 2, A, A))
            lm = LengthModel(L, "plm", sigma2, J=J, logZ=float(L * np.log(A)),
                             entropy_bits=float(L * np.log2(A)))
        return lm, {"converged": True, "uniform": True}
    if L <= 5:
        init_m = init._potts() if (init is not None and init.backend == "potts") else None
        m, d = potts.fit(states, counts, L, 2, "map", sigma2=sigma2, init=init_m)
        lw = m.log_weights()
        mx = float(lw.max())
        logZ = mx + float(np.log(np.exp(lw - mx).sum()))
        lm = LengthModel(L, "potts", sigma2, V=m.V, logZ=logZ,
                         entropy_bits=m.entropy_bits())
        return lm, d
    init_m = init._plm() if (init is not None and init.backend == "plm") else None
    m, d = longwords.fit_plm(states, counts, L, sigma2, init=init_m)
    z = longwords.exact_log_partition(m)
    return LengthModel(L, "plm", sigma2, J=m.J, logZ=z["logZ"],
                       entropy_bits=z["entropy_bits"]), d


# --------------------------------------------------------------------------- #
# grand-canonical energy
# --------------------------------------------------------------------------- #

def word_energy_bits(models: dict[int, LengthModel], p_len: np.ndarray, L: int,
                     states: np.ndarray) -> np.ndarray:
    """E(w) = -log2 P(L) - log2 P_L(w) in bits, per row of ``states`` (n, L)."""
    with np.errstate(divide="ignore"):
        return -np.log2(p_len[L - L_MIN]) - models[L].log2_prob(states)


# --------------------------------------------------------------------------- #
# bigram profile, coordinates, basis
# --------------------------------------------------------------------------- #

def bigram_ids(states: np.ndarray) -> np.ndarray:
    """The L+1 boundary-marked bigram ids (0..728) of each row of ``states`` (n, L)."""
    n, L = states.shape
    syms = np.concatenate([np.full((n, 1), BOUNDARY_IDX, dtype=np.int64),
                           states, np.full((n, 1), BOUNDARY_IDX, dtype=np.int64)], axis=1)
    return syms[:, :-1] * S + syms[:, 1:]


def coords_of_states(U: np.ndarray, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) coordinates of each row of ``states``; U is (2, 27, 27) bigram scores.

    x(w) = sum over the word's L+1 boundary-marked bigrams of U[0][from, to].  Computed
    with per-position accumulations so only O(n) scratch is allocated (important for the
    L = 5 enumeration and the insertion adjacency, which touch hundreds of millions of
    length-6 states)."""
    n, L = states.shape
    U0, U1 = U[0], U[1]
    B = BOUNDARY_IDX
    x = U0[B, states[:, 0]] + U0[states[:, L - 1], B]
    y = U1[B, states[:, 0]] + U1[states[:, L - 1], B]
    for i in range(L - 1):
        x = x + U0[states[:, i], states[:, i + 1]]
        y = y + U1[states[:, i], states[:, i + 1]]
    return x, y


def enumerate_states(L: int) -> np.ndarray:
    """All 26^L letter-index states of length L as an (26^L, L) int64 array."""
    return np.stack(np.unravel_index(np.arange(A ** L), (A,) * L), axis=1).astype(np.int64)


def accumulate_bigram_moments(states: np.ndarray, weights: np.ndarray,
                              mean: np.ndarray, outer: np.ndarray) -> float:
    """Accumulate weighted bigram-profile mean and outer second moment in place.

    ``states`` is (n, L) letter indices; ``weights`` the non-negative grand-canonical
    importance weights.  Returns the added total weight.  ``outer`` is (729, 729)."""
    if len(states) == 0:
        return 0.0
    ids = bigram_ids(states)                        # (n, L+1)
    k = ids.shape[1]
    np.add.at(mean, ids.ravel(), np.repeat(weights, k))
    g1 = ids[:, :, None]
    g2 = ids[:, None, :]
    flat = (g1 * N_BIGRAMS + g2).reshape(-1)        # (n * k * k,)
    wt = np.repeat(weights, k * k)
    np.add.at(outer.reshape(-1), flat, wt)
    return float(weights.sum())


def principal_basis(C: np.ndarray) -> np.ndarray:
    """Top-2 eigenvectors of a (729, 729) covariance -> (2, 27, 27) score tensors."""
    lam, V = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, V = np.clip(lam[order], 0.0, None), V[:, order]
    piv = np.abs(V).argmax(axis=0)
    V = V * np.sign(V[piv, np.arange(V.shape[1])])
    return V[:, :2].T.reshape(2, S, S)


def axis_report(U: np.ndarray, lam: np.ndarray, k: int) -> dict:
    """Composition of bigram-profile axis k: variance share and largest loadings."""
    u = U[k].ravel()
    sq = u ** 2
    order = np.argsort(-np.abs(u))[:12]
    return {"variance_share": float(lam[k] / lam.sum()),
            "eigenvalue": float(lam[k]),
            "top_bigrams": [(SYMBOLS[i // S] + SYMBOLS[i % S], float(u[i]), float(sq[i]))
                            for i in order]}


# --------------------------------------------------------------------------- #
# importance sampling
# --------------------------------------------------------------------------- #

@dataclass
class Proposal:
    """Mixture proposal q = lam * independent + (1 - lam) * empirical, per length L."""
    L: int
    lam: float
    logq0: np.ndarray                # (L, 26) independent log-probs (nats)
    emp_states: np.ndarray           # (T, L) observed types
    emp_logp: np.ndarray             # (T,) log token freqs
    _lookup: dict | None = None

    def __post_init__(self):
        self._lookup = {tuple(row): i for i, row in enumerate(self.emp_states)}

    @classmethod
    def from_data(cls, states, counts, L, lam=MIXTURE_WEIGHT) -> "Proposal":
        ind = longwords.fit_independent(states, counts, L)
        return cls(L, lam, ind.logq, states.copy(), np.log(counts / counts.sum()))

    def _sample_q0(self, n: int, rng) -> np.ndarray:
        q = np.exp(self.logq0)
        cdf = np.cumsum(q, axis=1)
        return np.stack([np.minimum(np.searchsorted(cdf[i], rng.random(n)), A - 1)
                         for i in range(self.L)], axis=1).astype(np.int64)

    def _sample_emp(self, n: int, rng) -> np.ndarray:
        p = np.exp(self.emp_logp)
        idx = rng.choice(len(p), n, p=p)
        return self.emp_states[idx]

    def sample(self, n: int, rng) -> np.ndarray:
        n0 = int(rng.binomial(n, self.lam))
        s0 = self._sample_q0(n0, rng)
        s1 = self._sample_emp(n - n0, rng)
        out = np.concatenate([s0, s1], axis=0)
        rng.shuffle(out)
        return out

    def log_q(self, states: np.ndarray) -> np.ndarray:
        lq0 = self.logq0[np.arange(self.L), states].sum(axis=1)
        rows = np.array([self._lookup.get(tuple(s), -1) for s in states])
        lqe = np.where(rows >= 0, self.emp_logp[rows], -np.inf)
        mx = np.maximum(lq0, np.where(np.isinf(lqe), lq0, lqe))
        q0 = self.lam * np.exp(lq0 - mx)
        qe = (1.0 - self.lam) * np.exp(np.where(np.isinf(lqe), -np.inf, lqe - mx))
        return mx + np.log(q0 + qe)


def importance_samples(model: LengthModel, prop: Proposal, n: int, rng,
                       batch: int = 200_000):
    """Draw n proposal samples; return (states, w) with w = P_L(s)/q(s)."""
    parts_s, parts_w = [], []
    for off in range(0, n, batch):
        nb = min(batch, n - off)
        s = prop.sample(nb, rng)
        lq = prop.log_q(s)
        w = np.exp(model.log_weight(s) - model.logZ - lq)
        parts_s.append(s)
        parts_w.append(w)
    return np.concatenate(parts_s), np.concatenate(parts_w)


def uniform_samples(L: int, n: int, rng, batch: int = 200_000) -> np.ndarray:
    """n strings of length L drawn uniformly from 26^L."""
    return np.stack([rng.integers(0, A, n) for _ in range(L)], axis=1).astype(np.int64)


def bin_accumulate(bins: np.ndarray, vals: np.ndarray, n_cells: int):
    """First and second raw moments per bin of ``vals`` over binned samples."""
    m1 = np.bincount(bins, weights=vals, minlength=n_cells)
    m2 = np.bincount(bins, weights=vals * vals, minlength=n_cells)
    return m1, m2


def mean_and_se(m1: np.ndarray, m2: np.ndarray, n: int):
    """Mean and standard error of a Monte-Carlo estimate m1/n from raw moments."""
    mean = m1 / n
    var = np.maximum(m2 - m1 * m1 / n, 0.0) / (n * (n - 1))
    return mean, np.sqrt(var)


# --------------------------------------------------------------------------- #
# bin energies
# --------------------------------------------------------------------------- #

def grid_from_coords(xs, ys, n: int) -> np.ndarray:
    """Edges (2, n+1) spanning the given coordinate collections (lists of arrays)."""
    x = np.concatenate([np.asarray(c).ravel() for c in xs if len(c)])
    y = np.concatenate([np.asarray(c).ravel() for c in ys if len(c)])
    return pj.grid_edges(np.stack([x, y]), n)


def bin_index_xy(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return pj.bin_index(np.stack([x, y]), edges)


def combine_bin_energy(p_len: np.ndarray, p_bin_by_L: dict[int, np.ndarray],
                       p_bin_se_by_L: dict[int, np.ndarray],
                       p_prior_by_L: dict[int, np.ndarray],
                       p_prior_se_by_L: dict[int, np.ndarray],
                       ess_by_L: dict[int, float | None],
                       n_cells: int) -> dict:
    """Combine per-length model and prior bin probabilities into z_bin and its SE.

    z_bin = log2 P_prior(bin) - log2 P_bin, where

        P_bin    = sum_L P(L)  P(bin | L),           P(bin | L) = model mass of the cell,
        P_prior  = sum_L pi_L  P_prior(bin | L),     pi_L = 1/8 (Dirichlet prior mean),

    and the prior is uniform within each length, P_prior(bin | L) = |bin|_L / 26^L.
    Both P(bin | L) and P_prior(bin | L) are estimated from the SAME importance draws
    (exact for L <= 5), so |bin| itself never needs a separate estimator and there is
    no count floor: a cell with model mass necessarily has prior mass, and vice versa.
    Cells with no mass under either estimator are gaps; cells whose mass is dominated by
    a length with ESS below ``ESS_THRESHOLD`` (the L = 8 sampler has ESS ~ 10) are
    flagged low-confidence, because their SE is a lower bound, not a calibrated error.
    """
    P_bin = np.zeros(n_cells)
    P_var = np.zeros(n_cells)
    P_prior = np.zeros(n_cells)
    P_prior_var = np.zeros(n_cells)
    low_ess_mass = np.zeros(n_cells)
    pi = DIRICHLET_ALPHA / (DIRICHLET_ALPHA * len(LENGTHS))
    for L in LENGTHS:
        pL = p_len[L - L_MIN]
        pb = p_bin_by_L.get(L, np.zeros(n_cells))
        pbs = p_bin_se_by_L.get(L, np.zeros(n_cells))
        pp = p_prior_by_L.get(L, np.zeros(n_cells))
        pps = p_prior_se_by_L.get(L, np.zeros(n_cells))
        P_bin += pL * pb
        P_var += (pL * pbs) ** 2
        P_prior += pi * pp
        P_prior_var += (pi * pps) ** 2
        ess = ess_by_L.get(L)
        if ess is not None and ess < ESS_THRESHOLD:
            low_ess_mass += pL * pb

    valid = (P_bin > 0) & (P_prior > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_bin = np.where(valid, np.log2(P_prior) - np.log2(P_bin), np.nan)
        rel_P = np.where(P_bin > 0, P_var / (P_bin * P_bin), np.inf)
        rel_Pr = np.where(P_prior > 0, P_prior_var / (P_prior * P_prior), np.inf)
    z_se = np.sqrt(rel_P + rel_Pr) / LN2
    z_se[~valid] = np.nan
    gap = ~valid                       # no mass under either estimator
    one_sided = (P_bin > 0) != (P_prior > 0)
    low_ess = valid & (low_ess_mass > 0.5 * P_bin)
    return {"z_bin": z_bin, "z_se": z_se, "P_bin": P_bin, "P_prior": P_prior,
            "gap": gap, "one_sided": one_sided, "low_ess": low_ess}


def confidence_mask(z_se: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Cells whose z_bin standard error is above ``threshold`` bits are low-confidence."""
    return z_se <= threshold


def quantize_surface(z: np.ndarray, centibits: float = 100.0) -> np.ndarray:
    """Quantise a z grid (flattened cells) to int16 centibits for the trajectory.

    Non-finite cells (gaps) become -32768, which must survive: only the finite
    (rounded) values are clipped to the int16 range, so the sentinel round-trips to a
    NaN gap on decoding."""
    q = np.where(np.isfinite(z), np.clip(np.round(z * centibits), -32767, 32767),
                 -32768)
    return q.astype(np.int16)


# --------------------------------------------------------------------------- #
# cell adjacency (move set)
# --------------------------------------------------------------------------- #

def _mark_pairs(adjmat: np.ndarray, u: np.ndarray, v: np.ndarray):
    m = u != v
    if not m.any():
        return
    a, b = np.minimum(u[m], v[m]), np.maximum(u[m], v[m])
    adjmat[a, b] = True


def move_adjacency_from_states(states: np.ndarray, bins: np.ndarray, L: int,
                               bin_fn, adjmat: np.ndarray):
    """Mark in ``adjmat`` (n_cells x n_cells) the cell pairs linked by one move of
    {sub, trans, del, ins} starting from the length-L ``states`` (n, L) with flat cell
    indices ``bins``.  ``bin_fn(s)`` returns the flat cell index of the states ``s``
    (length inferred from ``s``).  Boundary: no insertion from L = 8 and no deletion
    from L = 1."""
    n = len(states)
    if n == 0:
        return
    # substitution (L x 25)
    for p in range(L):
        sp = states.copy()
        for a in range(A):
            sp[:, p] = a
            m = states[:, p] != a
            if m.any():
                _mark_pairs(adjmat, bins[m], bin_fn(sp[m]))
    # adjacent transposition (L - 1)
    for p in range(L - 1):
        sp = states.copy()
        sp[:, [p, p + 1]] = sp[:, [p + 1, p]]
        m = np.any(sp != states, axis=1)
        if m.any():
            _mark_pairs(adjmat, bins[m], bin_fn(sp[m]))
    # deletion (L -> L-1) if L - 1 in the modelled range
    if L - 1 >= L_MIN:
        for p in range(L):
            sd = np.delete(states, p, axis=1)
            _mark_pairs(adjmat, bins, bin_fn(sd))
    # insertion (L -> L+1) if L + 1 in the modelled range
    if L + 1 <= L_MAX:
        si = np.empty((n, L + 1), dtype=states.dtype)
        for p in range(L + 1):
            si[:, :p] = states[:, :p]
            si[:, p + 1:] = states[:, p:]
            for a in range(A):
                si[:, p] = a
                _mark_pairs(adjmat, bins, bin_fn(si))


def adjacency_list(adjmat: np.ndarray) -> list[np.ndarray]:
    """Per-cell neighbour lists from a symmetric (n_cells, n_cells) adjacency matrix."""
    n_cells = adjmat.shape[0]
    adjmat = adjmat | adjmat.T                         # symmetrise (upper -> both)
    return [np.flatnonzero(adjmat[k]).astype(np.int64) for k in range(n_cells)]


# --------------------------------------------------------------------------- #
# string-level minima
# --------------------------------------------------------------------------- #

def states_of_words(words: list[str]) -> np.ndarray:
    return (np.frombuffer("".join(words).encode("ascii"), dtype=np.uint8)
            - 97).reshape(-1, len(words[0]) if words else 0).astype(np.int64)


def words_of(states: np.ndarray) -> list[str]:
    alph = np.array(list(ALPHABET))
    return ["".join(r) for r in alph[states]]


def full_move_strict_minima(models: dict[int, LengthModel], p_len: np.ndarray, L: int,
                            states: np.ndarray) -> np.ndarray:
    """Boolean mask: row of ``states`` is a strict minimum of E under {sub, trans, del,
    ins}.  A word is a strict minimum iff its energy is strictly below every modelled
    neighbour's energy (finite energies everywhere under the prior)."""
    n = len(states)
    e0 = word_energy_bits(models, p_len, L, states)
    best = np.full(n, np.inf)
    # substitution
    for p in range(L):
        sp = states.copy()
        for a in range(A):
            np.copyto(sp[:, p], a)
            m = states[:, p] != a
            best[m] = np.minimum(best[m], word_energy_bits(models, p_len, L, sp[m]))
    # transposition
    for p in range(L - 1):
        sp = states.copy()
        sp[:, [p, p + 1]] = sp[:, [p + 1, p]]
        m = np.any(sp != states, axis=1)
        best[m] = np.minimum(best[m], word_energy_bits(models, p_len, L, sp[m]))
    # deletion
    if L - 1 >= L_MIN:
        for p in range(L):
            sd = np.delete(states, p, axis=1)
            best = np.minimum(best, word_energy_bits(models, p_len, L - 1, sd))
    # insertion
    if L + 1 <= L_MAX:
        si = np.empty((n, L + 1), dtype=states.dtype)
        for p in range(L + 1):
            si[:, :p] = states[:, :p]
            si[:, p + 1:] = states[:, p:]
            for a in range(A):
                si[:, p] = a
                best = np.minimum(best, word_energy_bits(models, p_len, L + 1, si))
    return e0 < best


# --------------------------------------------------------------------------- #
# projected wells
# --------------------------------------------------------------------------- #

def projected_wells(E: np.ndarray, adj: list[np.ndarray], threshold: float,
                    cell_top_words: dict[int, list[dict]],
                    centres, P_bin: np.ndarray | None = None
                    ) -> tuple[list[dict], dict | None, np.ndarray]:
    """Persistent wells of the cell surface (reusing ``pj.persistence``) with the most
    probable strings of each basin.

    ``cell_top_words`` maps a flat cell index to its top strings (any length) as a list
    of ``{w, L, p, approx}`` sorted by grand-canonical probability p descending.  A
    well's ``words`` are the top ``BASIN_WORDS`` strings of the whole basin ranked by p
    (the union of the basin cells' top lists, sorted by p — never by cell index or
    corpus order), each with ``share = p / basin_mass``.  ``min_cell_top`` is the
    minimum cell's own top string, stored separately because at fine resolutions the
    minimum cell is often a low-mass single string while the basin's mass is carried by
    other strings (a length-mixing effect)."""
    wells = pj.persistence(E, adj)
    by_cell = {w["cell"]: w for w in wells}
    lab = pj.significant_basins(E, wells, threshold, adj)
    out = []
    for w in (w for w in wells if w["persistence"] >= threshold):
        idx = np.flatnonzero(lab.ravel() == w["cell"])
        best = {}
        for c in idx:
            for e in cell_top_words.get(int(c), []):
                wd = e["w"]
                if wd not in best or e["p"] > best[wd]["p"]:
                    best[wd] = e
        top = sorted(best.values(), key=lambda e: -e["p"])[:BASIN_WORDS]
        basin_mass = float(P_bin[idx].sum()) if P_bin is not None else 0.0
        words = [{"w": e["w"], "L": e["L"], "p": float(e["p"]),
                  "approx": bool(e.get("approx", False)),
                  "share": float(e["p"] / basin_mass) if basin_mass > 0 else None}
                 for e in top]
        mt = cell_top_words.get(int(w["cell"]), [])
        min_cell_top = ({"w": mt[0]["w"], "L": mt[0]["L"], "p": float(mt[0]["p"]),
                         "approx": bool(mt[0].get("approx", False))} if mt else None)
        x, y = _cell_xy(w["cell"], centres)
        out.append({"cell": w["cell"], "x": x, "y": y, "E": w["E_min"],
                    "persistence": w["persistence"], "essential": w["essential"],
                    "words": words, "min_cell_top": min_cell_top,
                    "basin_mass": basin_mass,
                    "cells": int((lab == w["cell"]).sum())})
    barrier = None
    merging = [w for w in wells if w["persistence"] >= threshold and not w["essential"]]
    if merging:
        w = merging[0]
        other = by_cell[w["merges_into"]]
        x, y = _cell_xy(w["saddle"], centres)
        barrier = {"x": x, "y": y, "E": w["E_saddle"], "cell": w["saddle"],
                   "wells": [w["cell"], other["cell"]],
                   "above": [w["E_saddle"] - w["E_min"],
                             w["E_saddle"] - other["E_min"]]}
    return out, barrier, lab


def _cell_xy(cell: int, centres) -> tuple[float, float]:
    n = len(centres[0])
    r, c = divmod(int(cell), n)
    return float(centres[0][c]), float(centres[1][r])


def cell_centres(edges: np.ndarray) -> list[np.ndarray]:
    return [(e[:-1] + e[1:]) / 2 for e in edges]
