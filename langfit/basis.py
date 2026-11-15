"""Projection bases for viewing an energy landscape over A^L strings.

A basis is a pair of real functions y(s) = (y1(s), y2(s)) on the A^L strings. A
*linear* basis is y(s) = V^T (x(s) - mu), with x(s) the one-hot (position, letter)
indicator vector of dimension L*A, so each coordinate is a sum of L per-(position,
letter) loadings; a nonlinear basis is any embedding of all strings.

The kinetics used to judge and to build bases is the random-site heat-bath (Gibbs)
chain of Stephens & Bialek's move set: pick a position i uniformly, redraw its letter
from P(l_i | rest). It is reversible with stationary distribution P (every update is a
P-orthogonal projection, so T is self-adjoint and positive semi-definite on L2(P)).
A deterministic sweep i = 1..L is not reversible, which is why the random-site update
is used. Time is counted in single-site updates; L updates make one sweep.

Everything is computed exactly, by enumerating all A^L strings: T acts on a function f
by fibre averages, (T f)(s) = (1/L) Σ_i Σ_b P(b | s_-i) f(s_-i, b), which costs O(L A^L).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.sparse.linalg import LinearOperator, eigsh

# Relative eigenvalue floor below which a covariance direction is treated as null
# (numerical parameter: the indicator covariance has exact null directions, one per
# position because each position's indicators sum to 1, plus any unused letter).
NULL_TOL = 1e-10


# --------------------------------------------------------------------------- #
# heat-bath kinetics, exact
# --------------------------------------------------------------------------- #

def heat_bath_apply(P: np.ndarray, F: np.ndarray) -> np.ndarray:
    """One random-site heat-bath update applied to functions: (T f)(s) for each f.

    P: (A,)*L probabilities; F: (k,)+(A,)*L function values. Fibres whose strings all
    have P = 0 are never visited by the chain from the support; their value is set to 0.
    """
    L = P.ndim
    PF = P * F
    out = np.zeros_like(F)
    for i in range(L):
        num = PF.sum(axis=i + 1, keepdims=True)
        den = P.sum(axis=i, keepdims=True)
        out += np.divide(num, den, out=np.zeros(num.shape), where=den > 0)
    out /= L
    return out


def indicator_functions(L: int, A: int, positions) -> np.ndarray:
    """Dense one-hot indicators x_{(i,a)}(s) for the given positions, shape (k,)+(A,)*L."""
    out = []
    for i in positions:
        shape = [1] * L
        shape[i] = A
        eye = np.eye(A).reshape((A, *shape))
        out.append(np.broadcast_to(eye, (A,) + (A,) * L))
    return np.concatenate(out).astype(float)


def _pair_with_indicators(P: np.ndarray, G: np.ndarray) -> np.ndarray:
    """E_P[x_(j,a)(s) g(s)] for every (j, a) and every function g in G: (L*A, k)."""
    L = P.ndim
    PG = P * G
    return np.concatenate([PG.sum(axis=tuple(1 + m for m in range(L) if m != j)).T
                           for j in range(L)])


def indicator_lagged_covariances(P: np.ndarray, lags, chunk_positions: int = 1) -> dict:
    """Exact lag-τ covariances C(τ) = Cov_P[x(s_0), x(s_τ)] of the (position, letter)
    indicators under the heat-bath chain, for each τ in ``lags`` (single-site updates).

    C(0) is the static indicator covariance. Uses E_P[x_a T^τ x_b] = Σ over strings
    with s_j = a of P(s) (T^τ x_b)(s), so one forward propagation of the indicators
    serves every lag. Indicators are propagated one position (A functions) at a time to
    bound memory. Returned matrices are symmetrised; the largest asymmetry before
    symmetrisation (zero in exact arithmetic, by reversibility) is reported.
    """
    L, A = P.ndim, P.shape[0]
    lags = sorted(set(int(t) for t in lags) | {0})
    mu = np.concatenate([P.sum(axis=tuple(m for m in range(L) if m != j)) for j in range(L)])
    raw = {t: np.zeros((L * A, L * A)) for t in lags}
    for start in range(0, L, chunk_positions):
        pos = range(start, min(L, start + chunk_positions))
        G = indicator_functions(L, A, pos)
        cols = slice(start * A, (start + len(pos)) * A)
        t = 0
        while True:
            if t in raw:
                raw[t][:, cols] = _pair_with_indicators(P, G)
            if t == lags[-1]:
                break
            G = heat_bath_apply(P, G)
            t += 1
        del G
    out, asym = {}, 0.0
    for t, M in raw.items():
        M = M - np.outer(mu, mu)
        asym = max(asym, float(np.abs(M - M.T).max()))
        out[t] = (M + M.T) / 2
    return {"C": out, "mean": mu, "max_asymmetry": asym}


def function_lagged_covariances(P: np.ndarray, Y: np.ndarray, lags) -> dict:
    """Exact lag-τ covariances of arbitrary functions Y (k, A^L) under the chain."""
    L, A = P.ndim, P.shape[0]
    lags = sorted(set(int(t) for t in lags) | {0})
    w = P.ravel()
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    mu = Y @ w
    Yc = (Y - mu[:, None]).reshape((len(Y),) + P.shape)
    G = Yc.copy()
    out, t = {}, 0
    while True:
        if t in lags:
            M = (Yc.reshape(len(Y), -1) * w) @ G.reshape(len(Y), -1).T
            out[t] = (M + M.T) / 2
        if t == lags[-1]:
            break
        G = heat_bath_apply(P, G)
        t += 1
    return {"C": out, "mean": mu}


# --------------------------------------------------------------------------- #
# linear bases: generalized eigenproblems in the indicator space
# --------------------------------------------------------------------------- #

def whitener(C0: np.ndarray, tol: float = NULL_TOL) -> np.ndarray:
    """W with W^T C0 W = I on the range of C0 (null directions dropped)."""
    lam, U = np.linalg.eigh(C0)
    keep = lam > tol * lam.max()
    return U[:, keep] / np.sqrt(lam[keep])


def _sign_fix(V: np.ndarray) -> np.ndarray:
    """Each column's largest-|loading| entry positive (as projection.principal_axes)."""
    piv = np.abs(V).argmax(axis=0)
    return V * np.sign(V[piv, np.arange(V.shape[1])])


def generalized_axes(M: np.ndarray, C0: np.ndarray, tol: float = NULL_TOL
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Solve M v = λ C0 v on the range of C0 (M symmetric): eigenvalues descending and
    C0-orthonormal eigenvectors (columns, v^T C0 v = 1)."""
    W = whitener(C0, tol)
    lam, R = np.linalg.eigh(W.T @ M @ W)
    order = np.argsort(lam)[::-1]
    return lam[order], _sign_fix(W @ R[:, order])


def pca_axes(C0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit-norm principal axes (the current artifact basis)."""
    lam, U = np.linalg.eigh(C0)
    order = np.argsort(lam)[::-1]
    return np.clip(lam[order], 0, None), _sign_fix(U[:, order])


def tica_axes(C0: np.ndarray, Ct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TICA / linear VAC (Pérez-Hernández et al. 2013; Noé & Nüske 2013): the
    C0-orthonormal solutions of C(τ) v = λ C(0) v, slowest first. λ is the lag-τ
    autocorrelation of the coordinate v^T x."""
    return generalized_axes(Ct, C0)


def class_means(P: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P-mass π_k and indicator means μ_k (k, L*A) of every class of a labelling of the
    A^L strings (labels: int per flat string, -1 = unlabelled)."""
    L, A = P.ndim, P.shape[0]
    w = P.ravel()
    lab = labels.ravel()
    ok = (lab >= 0) & (w > 0)
    uniq, inv = np.unique(lab[ok], return_inverse=True)
    pi = np.bincount(inv, weights=w[ok], minlength=len(uniq))
    digits = np.stack(np.unravel_index(np.flatnonzero(ok), P.shape))       # (L, n)
    S = np.zeros((len(uniq), L * A))
    for j in range(L):
        np.add.at(S, (inv, j * A + digits[j]), w[ok])
    return uniq, pi, S / pi[:, None]


def lda_axes(P: np.ndarray, labels: np.ndarray, C0: np.ndarray, mean: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    """Fisher discriminant axes of a P-weighted labelling: S_B v = λ C0 v with
    S_B = Σ_k π_k (μ_k - μ)(μ_k - μ)^T. λ is the share of a coordinate's variance that
    is between classes (its correlation ratio η^2)."""
    _, pi, mu = class_means(P, labels)
    pi = pi / pi.sum()
    D = mu - mean
    SB = (D * pi[:, None]).T @ D
    return generalized_axes(SB, C0)


def _lda_axes_regularised(P: np.ndarray, labels: np.ndarray, L: int, A: int,
                          var_keep: float = 0.95) -> np.ndarray:
    """Fisher discriminant axes restricted to the leading-PC subspace carrying
    ``var_keep`` of C(0)'s variance (regularisation: avoids the rare-letter whitening
    blow-up of the raw generalised eigenproblem)."""
    mean = np.concatenate([P.sum(axis=tuple(k for k in range(L) if k != j))
                           for j in range(L)])
    C0 = _indicator_covariance(P)
    _, pi, mu = class_means(P, labels)
    pi = pi / pi.sum()
    SB = ((mu - mean) * pi[:, None]).T @ (mu - mean)
    lam, U = np.linalg.eigh(C0)
    order = np.argsort(lam)[::-1]
    lam, U = lam[order], U[:, order]
    keep = np.cumsum(lam) <= var_keep * lam.sum()
    keep[-1] = True
    sub = U[:, keep]
    Ms = sub.T @ SB @ sub
    l2, R = np.linalg.eigh(Ms)
    V = sub @ R[:, np.argsort(l2)[::-1][:2]]
    return _sign_fix(V / np.linalg.norm(V, axis=0))


def coords_linear(V: np.ndarray, mean: np.ndarray, L: int, A: int) -> np.ndarray:
    """(k, A^L) coordinates V^T (x(s) - mean) of every string."""
    k = V.shape[1]
    out = np.zeros((k,) + (A,) * L)
    for i in range(L):
        shape = [1] * L
        shape[i] = A
        out += V[i * A:(i + 1) * A].T.reshape((k, *shape))
    return out.reshape(k, -1) - (V.T @ mean)[:, None]


# --------------------------------------------------------------------------- #
# kinetic scores
# --------------------------------------------------------------------------- #

def _whitened_koopman(C0y: np.ndarray, Cty: np.ndarray) -> np.ndarray:
    lam, U = np.linalg.eigh(C0y)
    keep = lam > NULL_TOL * lam.max()
    W = U[:, keep] / np.sqrt(lam[keep])
    return W.T @ Cty @ W


def vamp2(C0y: np.ndarray, Cty: np.ndarray) -> float:
    """VAMP-2 score (Wu & Noé 2020) of mean-free features with covariances C0y, C(τ)y,
    for a reversible chain (C11 = C00): ||C00^-1/2 C0τ C00^-1/2||_F^2. The constant
    function's contribution (1) is excluded, so a 2D basis scores in [0, 2]."""
    K = _whitened_koopman(C0y, Cty)
    return float((K ** 2).sum())


def autocorrelations(C0y: np.ndarray, Cty: np.ndarray) -> np.ndarray:
    """Eigenvalues (descending) of the whitened lag-τ covariance of the features."""
    return np.sort(np.linalg.eigvalsh(_whitened_koopman(C0y, Cty)))[::-1]


def implied_timescales(lam: np.ndarray, tau: int) -> np.ndarray:
    """t = -τ / ln λ (single-site updates); inf for λ >= 1, nan for λ <= 0."""
    lam = np.asarray(lam, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -tau / np.log(lam)
    t[lam >= 1] = np.inf
    t[lam <= 0] = np.nan
    return t


def mass_weighted_scale(E: np.ndarray, mass: np.ndarray, occ: np.ndarray) -> dict:
    """Outlier-robust scales of a per-cell scalar E (e.g. E_bin) over the occupied cells:
    the P-mass-weighted mean and SD, and the mass-weighted 5-95 inter-percentile range.
    Unlike the raw range, these are insensitive to a single sparse (tiny-mass) outlier
    cell stretching the range, so they are the right denominators for normalising
    surface roughness."""
    m = mass[occ]
    e = E[occ]
    mu = float((m * e).sum() / m.sum())
    sd = float(np.sqrt((m * (e - mu) ** 2).sum() / m.sum()))
    order = np.argsort(e)
    cdf = np.cumsum(m[order]) / m[order].sum()
    lo = float(e[order[int(np.searchsorted(cdf, 0.05))]])
    hi = float(e[order[min(int(np.searchsorted(cdf, 0.95)), len(e) - 1)]])
    return {"mean": mu, "sd": sd, "iqr595": hi - lo}


# --------------------------------------------------------------------------- #
# exact slow modes of the chain (nonlinear optimum of VAMP)
# --------------------------------------------------------------------------- #

def support_components(P: np.ndarray) -> np.ndarray:
    """Connected components of the support {P > 0} under single-letter substitution:
    flat label per string (-1 off the support), by fibre-minimum label propagation."""
    L = P.ndim
    sup = P > 0
    big = np.iinfo(np.int64).max
    lab = np.where(sup, np.arange(P.size).reshape(P.shape), big)
    while True:
        new = lab
        for i in range(L):
            new = np.minimum(new, np.where(sup, new.min(axis=i, keepdims=True), big))
        if np.array_equal(new, lab):
            break
        lab = new
    out = np.where(sup, lab, -1).ravel()
    u, inv = np.unique(out[out >= 0], return_inverse=True)
    res = np.full(P.size, -1)
    res[out >= 0] = inv
    return res


def slow_modes(P: np.ndarray, k: int, tol: float = 1e-10, maxiter: int | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Leading eigenpairs of the heat-bath operator T on L2(P), by Lanczos on the
    symmetrised operator P^1/2 T P^-1/2 restricted to the support.

    Returns eigenvalues (descending, the first is 1 for the constants) and the
    eigenfunctions ψ (k, A^L), P-orthonormal, 0 off the support. These are the
    diffusion-map coordinates (Coifman & Lafon 2006) of the heat-bath kernel and the
    exact maximisers of VAMP over all functions."""
    sup = (P > 0).ravel()
    w = P.ravel()[sup]
    sq = np.sqrt(w)
    shape = P.shape

    def mv(v):
        f = np.zeros(P.size)
        f[sup] = np.asarray(v).ravel() / sq
        g = heat_bath_apply(P, f.reshape((1,) + shape)).ravel()[sup]
        return g * sq

    op = LinearOperator((sup.sum(), sup.sum()), matvec=mv, dtype=float)
    lam, U = eigsh(op, k=k, which="LA", tol=tol, maxiter=maxiter, v0=sq.copy())
    order = np.argsort(lam)[::-1]
    lam, U = lam[order], U[:, order]
    psi = np.zeros((k, P.size))
    psi[:, sup] = (U / sq[:, None]).T
    piv = np.abs(psi * np.sqrt(P.ravel())).argmax(axis=1)
    psi *= np.sign(psi[np.arange(k), piv])[:, None]
    return lam, psi


# --------------------------------------------------------------------------- #
# recommended basis and annotation (the chosen 2-D view)
# --------------------------------------------------------------------------- #

def annotate_axes(V: np.ndarray, C0: np.ndarray, L: int, alphabet: str,
                  top: int = 10) -> list[dict]:
    """Per-axis annotation of a linear basis V (columns = loadings, L*A x k), in the
    same shape the artifact template renders: variance share, squared loadings by
    position (1-based), and the ``top`` largest signed loadings as (pos, letter, u, u^2).
    """
    A = V.shape[0] // L
    lam = np.array([float(v @ C0 @ v) for v in V.T])
    out = []
    for k in range(V.shape[1]):
        u = V[:, k]
        piv = int(np.argmax(np.abs(u)))
        sgn = np.sign(u[piv])
        sq = u ** 2
        order = np.argsort(-np.abs(u))[:top]
        out.append({
            "variance_share": float(lam[k] / np.trace(C0)),
            "eigenvalue": float(lam[k]),
            "position_share": sq.reshape(L, A).sum(axis=1).tolist(),
            "top_loadings": [(int(i // A) + 1, alphabet[i % A], float(u[i] * sgn),
                              float(sq[i])) for i in order],
        })
    return out


def pca_basis(P: np.ndarray, top: int = 2, alphabet: str = "abcdefghijklmnopqrstuvwxyz"
              ) -> dict:
    """The recommended 2-D basis for the S&B landscape: the top-``top`` principal axes
    of the P-weighted indicator covariance (validated in scripts/basis_study.py).

    Returns {"V": loadings (L*A x top), "eigenvalues", "mean": indicator means,
    "coordinates": (top, A^L), "annotation": per-axis reports}.
    """
    L, A = P.ndim, P.shape[0]
    C0 = _indicator_covariance(P)
    lam, V = pca_axes(C0)
    V = V[:, :top]
    mean = np.concatenate([P.sum(axis=tuple(k for k in range(L) if k != j))
                           for j in range(L)])
    return {"V": V, "eigenvalues": lam[:top], "mean": mean,
            "coordinates": coords_linear(V, mean, L, A),
            "annotation": annotate_axes(V, C0, L, alphabet)}


def _indicator_covariance(P: np.ndarray) -> np.ndarray:
    """Exact covariance of the one-hot (position, letter) indicators under P."""
    L, A = P.ndim, P.shape[0]
    p1 = np.stack([P.sum(axis=tuple(k for k in range(L) if k != i)) for i in range(L)])
    C = np.empty((L * A, L * A))
    for i in range(L):
        C[i * A:(i + 1) * A, i * A:(i + 1) * A] = np.diag(p1[i]) - np.outer(p1[i], p1[i])
    for i in range(L):
        for j in range(i + 1, L):
            pij = P.sum(axis=tuple(k for k in range(L) if k not in (i, j)))
            blk = pij - np.outer(p1[i], p1[j])
            C[i * A:(i + 1) * A, j * A:(j + 1) * A] = blk
            C[j * A:(j + 1) * A, i * A:(i + 1) * A] = blk.T
    return C


# --------------------------------------------------------------------------- #
# variable-length (new model): boundary-marked q-gram profiles
# --------------------------------------------------------------------------- #

def qgram_profile(word: str, q: int = 2, alphabet: str = "abcdefghijklmnopqrstuvwxyz",
                  boundary: str = "#") -> np.ndarray:
    """Counts of the q-grams of the boundary-marked string ``boundary + word + boundary``
    over the extended alphabet of ``len(alphabet) + 1`` symbols (the boundary marker is
    the last symbol).

    This is the recommended length-invariant feature for the variable-length landscape:
    the L1 distance between two profiles is <= 2q per single-letter substitution and
    <= 4q for an adjacent transposition (for q = 2: 4 and 6), so the q-gram distance
    lower-bounds the edit distance (Ukkonen 1992) — every edit move is local in profile
    space, while the profile is a linear sum of per-(q-gram) counts (so a grid cell's
    ``|bin|_L`` is computable by a transfer matrix over the L+1 boundary-marked q-grams).

    Returns a float array of length ``(len(alphabet)+1)**q`` (729 for the default).
    """
    sym = {c: i for i, c in enumerate(alphabet)}
    n = len(alphabet) + 1
    b = n - 1                                   # the boundary symbol's index
    codes = [sym[c] for c in word]
    seq = [b] + codes + [b]
    out = np.zeros(n ** q)
    for i in range(len(seq) - q + 1):
        idx = 0
        for k in range(q):
            idx = idx * n + seq[i + k]
        out[idx] += 1.0
    return out


# --------------------------------------------------------------------------- #
# selectable 2-D bases (registry + compute_basis)
# --------------------------------------------------------------------------- #

ALPHABET_DEFAULT = "abcdefghijklmnopqrstuvwxyz"


def _model_p_e(model, L: int, A: int):
    """Resolve (P, E, mean) from a MaxEnt/Potts-like model (``prob``/``energy``)."""
    P = model.prob()
    with np.errstate(divide="ignore"):
        E = -np.log(P)                            # nats; +inf for forbidden strings
    E = np.where(np.isfinite(E), E, 0.0)          # 0 on forbidden (P=0 -> no weight)
    mean = np.concatenate([P.sum(axis=tuple(k for k in range(L) if k != j))
                           for j in range(L)])
    return P, E, mean


def _coupling_spectral_axes(model, L: int, A: int) -> np.ndarray:
    """Top-2 eigenvectors of the gauge-fixed (double-centred) coupling matrix J.

    E(s) = -1/2 x^T J x (pairwise model, zero fields): the directions of largest
    quadratic energy variation are the leading eigenvectors of the centred J, i.e. the
    Hopfield-pattern (memory) directions along which the wells are oriented.
    """
    J = np.zeros((L * A, L * A))
    for (i, j), V in zip(model.cliques, model.V):
        Vf = np.where(np.isfinite(V), V, 0.0)   # forbidden pairs (V=-inf) are hard
        # support constraints, not part of the well geometry -> zero coupling here
        Vc = Vf - Vf.mean(axis=0, keepdims=True) - Vf.mean(axis=1, keepdims=True) + Vf.mean()
        J[i * A:(i + 1) * A, j * A:(j + 1) * A] = Vc
        J[j * A:(j + 1) * A, i * A:(i + 1) * A] = Vc.T
    lam, U = np.linalg.eigh(J)
    order = np.argsort(lam)[::-1]
    return _sign_fix(U[:, order][:, :2])


def _energy_cross(P: np.ndarray, E: np.ndarray) -> np.ndarray:
    """E_P[x_{(i,a)} E] for every (i, a): (L*A,)."""
    L = P.ndim
    PE = P * E
    return np.concatenate([PE.sum(axis=tuple(k for k in range(L) if k != j))
                           for j in range(L)])


def _pls_axes(P: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Two PLS loadings of the P-weighted regression of E(s) on the indicators.

    Axis 1 is the direction of maximal covariance between x and E (the linear
    predictor of energy); axis 2 is the next PLS component (deflated). Unit norm.
    """
    L = P.ndim
    mean_x = np.concatenate([P.sum(axis=tuple(k for k in range(L) if k != j))
                             for j in range(L)])
    C = _indicator_covariance(P)
    mean_E = float((P * E).sum())
    w = _energy_cross(P, E) - mean_x * mean_E
    u1 = w / np.linalg.norm(w)
    s1 = float(u1 @ C @ u1)
    w2 = w - (w @ u1) / s1 * (C @ u1)
    u2 = w2 / np.linalg.norm(w2)
    return _sign_fix(np.column_stack([u1, u2]))


def _tempered_axes(P: np.ndarray, alpha: float, L: int, A: int) -> np.ndarray:
    """PCA of the covariance weighted by P^alpha (alpha=0: uniform over all strings,
    classical MDS/Torgerson; alpha=1: the default P-weighted PCA)."""
    if alpha == 0.0:
        p1 = np.full((L, A), 1.0 / A)
        C = np.zeros((L * A, L * A))
        for i in range(L):
            C[i * A:(i + 1) * A, i * A:(i + 1) * A] = np.diag(p1[i]) - np.outer(p1[i], p1[i])
        # cross-position blocks are 0 (independent under the uniform product measure)
    else:
        Q = P ** alpha
        Q /= Q.sum()
        C = _indicator_covariance(Q)
    lam, U = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    return _sign_fix(U[:, order][:, :2])


BASES = {
    "pca": {
        "label": "PCA (default)",
        "summary": "Top-2 principal axes of the P-weighted indicator covariance.",
        "generalises": True,
    },
    "coupling_spectral": {
        "label": "Coupling spectral",
        "summary": "Top-2 eigenvectors of the gauge-fixed coupling matrix J "
                   "(directions of largest quadratic energy variation).",
        "generalises": False,          # needs the quadratic energy form
    },
    "energy_aligned": {
        "label": "Energy-aligned (PLS)",
        "summary": "P-weighted regression of E(s) on the indicators (PLS loadings).",
        "generalises": True,
    },
    "tempered_pca": {
        "label": "Tempered PCA",
        "summary": "PCA of the covariance weighted by P^alpha (alpha=0 uniform MDS, "
                   "alpha=1 standard PCA).",
        "generalises": True,
    },
    "lda": {
        "label": "Well-discriminant (LDA)",
        "summary": "Fisher discriminant axes of the string-level descent basins "
                   "(best separates the landscape's wells), P-weighted, regularised.",
        "generalises": True,
    },
    "energy_metric": {
        "label": "Energy-metric layout \u2014 DEGENERATE (diagnostic)",
        "summary": "Landmark-MDS (Isomap) of the substitution graph weighted by energy "
                   "slope; found degenerate (y ~ energy ramp, x ~ linear). Diagnostic "
                   "only, not a recommended basis.",
        "generalises": False,
        "kind": "coords",
        "diagnostic": True,
    },
}


def compute_basis(name: str, model=None, P=None, E=None, L: int = 4, A: int = 26,
                  alpha: float = 0.75, mass: float = 0.99, ridge: float = 1e-6,
                  lam: float | None = None, K: int = 200,
                  alphabet: str = ALPHABET_DEFAULT) -> dict:
    """Compute a named 2-D basis.  ``model`` has ``prob()``/``cliques``/``V`` (the
    pairwise MaxEnt/Potts); otherwise pass ``P`` (and ``E`` for the energy-aligned
    basis).  Linear bases return {"U": (L*A, 2) unit loadings, "mean": (L*A,),
    "label": str, "annotation": [per-axis reports], "kind": name}; the non-parametric
    ``energy_metric`` basis returns {"coords": (2, A^L) float32, "mean": (L*A,),
    "label", "kind": "coords", "annotation": None} instead of loadings.

    All linear columns are unit-norm with the largest-|loading| entry positive (the same
    sign convention as ``langfit.projection.principal_axes``), so the coordinates are
    ``basis.coords_linear(U, mean, L, A)``.

    ``alpha`` weights ``tempered_pca`` (P^alpha); ``lam``/``K`` are the energy-metric
    length scale and landmark count; ``mass`` and ``ridge`` are legacy no-ops.
    """
    if name not in BASES:
        raise KeyError(f"unknown basis {name!r}; choose from {sorted(BASES)}")
    if P is None:
        P, E, mean = _model_p_e(model, L, A)
    else:
        mean = np.concatenate([P.sum(axis=tuple(k for k in range(L) if k != j))
                               for j in range(L)])
    if name == "energy_metric":
        with np.errstate(divide="ignore"):
            Ebits = -np.log(P) / np.log(2)
        Ebits = _cap_infinite(Ebits)
        if lam is None:
            lam = _flux_median_slope(P, L, A)
        coords = energy_metric_layout(Ebits.ravel(), P.ravel(), L, A, lam, K=K)
        return {"coords": coords, "mean": mean, "label": BASES[name]["label"],
                "kind": "coords", "diagnostic": True, "annotation": None}
    C0 = _indicator_covariance(P)
    if name == "pca":
        U = pca_axes(C0)[1][:, :2]
    elif name == "coupling_spectral":
        U = _coupling_spectral_axes(model, L, A)
    elif name == "energy_aligned":
        if E is None:
            with np.errstate(divide="ignore"):
                E = -np.log(P)
            E = np.where(np.isfinite(E), E, 0.0)
        U = _pls_axes(P, E)
    elif name == "tempered_pca":
        U = _tempered_axes(P, alpha, L, A)
    elif name == "lda":
        import langfit.maxent as _me
        with np.errstate(divide="ignore"):
            Efull = -np.log(P)
        labels = _me.descend(Efull).astype(np.int64).ravel()
        U = _lda_axes_regularised(P, labels, L, A)
    else:  # pragma: no cover
        raise KeyError(name)
    # every sub-basis returns unit-norm, sign-fixed columns already (and PCA must stay
    # bit-identical to projection.principal_axes), so no further normalisation here
    return {"U": U, "mean": mean, "label": BASES[name]["label"], "kind": name,
            "annotation": annotate_axes(U, C0, L, alphabet)}


def compute_basis_generic(name: str, features: np.ndarray, weights: np.ndarray,
                          energies: np.ndarray, adjacency) -> dict:
    """The bases that generalise to an arbitrary length-invariant feature map phi(s).

    ``features``: (N, d); ``weights``: (N,) >= 0 (the model's P); ``energies``: (N,)
    (-log P, finite); ``adjacency``: an (E, 2) int array of undirected neighbour pairs
    (e.g. the four edit moves).  Supports pca, tempered_pca (``alpha`` via kwarg not
    needed here; use tempered covariance directly), energy_aligned, locality; raises
    for coupling_spectral (which needs the quadratic energy form).
    """
    if name not in ("pca", "tempered_pca", "energy_aligned", "locality"):
        raise KeyError(name)
    F = np.asarray(features, float)
    w = np.asarray(weights, float)
    w = w / w.sum()
    m = F.T @ w
    Fc = F - m
    C = (Fc * w[:, None]).T @ Fc
    if name == "pca":
        lam, U = np.linalg.eigh(C)
        U = U[:, np.argsort(lam)[::-1][:2]]
    elif name == "tempered_pca":
        # alpha fixed to 1/2 (Hellinger) for the generic feature map
        wh = np.sqrt(w)
        wh /= wh.sum()
        mh = F.T @ wh
        Ch = ((F - mh) * wh[:, None]).T @ (F - mh)
        lam, U = np.linalg.eigh(Ch)
        U = U[:, np.argsort(lam)[::-1][:2]]
    elif name == "energy_aligned":
        e = np.asarray(energies, float)
        me = float(w @ e)
        cr = Fc.T @ (w * (e - me))
        u1 = cr / np.linalg.norm(cr)
        s1 = float(u1 @ C @ u1)
        cr2 = cr - (cr @ u1) / s1 * (C @ u1)
        U = np.column_stack([u1, cr2 / np.linalg.norm(cr2)])
    elif name == "locality":
        N = len(F)
        a = np.asarray(adjacency, np.int64)
        deg = np.bincount(np.concatenate([a[:, 0], a[:, 1]]), minlength=N)
        XLX = np.zeros((F.shape[1], F.shape[1]))
        d = F[a[:, 0]] - F[a[:, 1]]
        XLX = d.T @ d
        B = (F * deg[:, None]).T @ F + 1e-4 * np.trace((F * deg[:, None]).T @ F) * np.eye(F.shape[1])
        lam, V = scipy.linalg.eigh(XLX, B)
        null = int(np.sum(lam < 1e-8 * max(1.0, lam[-1])))
        U = V[:, null:null + 2]
    else:
        raise KeyError(f"{name} does not generalise to a generic feature map")
    U = U / np.linalg.norm(U, axis=0)
    U = _sign_fix(U)
    return {"U": U, "mean": m, "label": BASES.get(name, {}).get("label", name),
            "kind": name}


import scipy.linalg  # noqa: E402  (used by the LPP solvers)


# --------------------------------------------------------------------------- #
# sixth basis: energy-metric layout (geodesic MDS of the 3-D landscape surface)
# --------------------------------------------------------------------------- #

def _ham_graph_csr(E: np.ndarray, L: int, A: int, lam: float):
    """Weighted single-substitution (Hamming) graph as a scipy CSR, edge length
    l_ij = sqrt(1 + ((E_i - E_j)/lam)^2) (hop length 1, energy slope 1/lam)."""
    from scipy.sparse import csr_matrix
    N = A ** L
    E = np.asarray(E, np.float32).ravel()
    digits = np.stack(np.unravel_index(np.arange(N), (A,) * L)).T     # (N, L)
    Kcol = L * A
    indptr = np.arange(N + 1, dtype=np.int64) * Kcol
    indices = np.empty(N * Kcol, np.int32)
    data = np.empty(N * Kcol, np.float32)
    base = np.arange(N)
    for p in range(L):
        off = A ** (L - 1 - p)
        d = digits[:, p]
        for a in range(A):
            col = p * A + a
            dst = base + (a - d) * off
            idx = base * Kcol + col
            indices[idx] = dst
            dE = E[dst] - E
            data[idx] = np.sqrt(1.0 + (dE / lam) ** 2)
    return csr_matrix((data, indices, indptr), shape=(N, N))


def _p_farthest_landmarks(csr, P, K, beta=0.5, seed=0):
    """P-aware farthest-point landmark sampling: K Dijkstra runs that also return the
    (K, N) geodesic distance matrix D."""
    from scipy.sparse.csgraph import dijkstra
    N = csr.shape[0]
    w = np.asarray(P, np.float64).ravel()
    w = np.maximum(w, 0.0) ** beta
    lm0 = int(np.argmax(w))
    landmarks = [lm0]
    rows = []
    dmin = np.full(N, np.inf)
    for _ in range(K):
        d = np.asarray(dijkstra(csr, directed=True, indices=[landmarks[-1]],
                                min_only=True)).ravel().astype(np.float32)
        rows.append(d)
        dmin = np.minimum(dmin, d)
        # next landmark: highest (P^beta * min-distance), excluding chosen ones
        score = dmin * w
        score[landmarks] = -1.0
        landmarks.append(int(np.argmax(score)))
    return np.stack(rows), np.array(landmarks[:-1])


def _landmark_mds(D, dim=2):
    """Classical (landmark) MDS of a (K, N) distance matrix -> (N, dim) coords."""
    K, N = D.shape
    D2 = D.astype(np.float64) ** 2
    B = -0.5 * (D2 - D2.mean(axis=0, keepdims=True) - D2.mean(axis=1, keepdims=True)
                + D2.mean())
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    coords = (Vt[:dim] * np.sqrt(np.maximum(s[:dim], 0.0))[:, None]).T   # (N, dim)
    return coords.astype(np.float32)


def _orient_layout(coords, E, P):
    """Deterministic orientation: rotate to the P-weighted principal axes, then sign
    each axis so its P-weighted Spearman correlation with E is non-negative."""
    w = np.asarray(P, np.float64).ravel()
    w = w / w.sum()
    mu = (w[:, None] * coords).sum(0)
    C = (coords - mu).T @ ((coords - mu) * w[:, None])
    lam, V = np.linalg.eigh(C)
    coords = (coords - mu) @ V[:, ::-1]               # principal axes, descending var
    from scipy.stats import spearmanr
    E = np.asarray(E).ravel()
    if np.ptp(E) > 0:
        for k in range(2):
            if spearmanr(coords[:, k], E).correlation < 0:
                coords[:, k] *= -1
    return coords.astype(np.float32)


def _cap_infinite(E: np.ndarray) -> np.ndarray:
    """Cap +inf (forbidden-string energy) at the largest finite value."""
    E = np.asarray(E, np.float32)
    finite = np.isfinite(E)
    cap = float(E[finite].max()) if finite.any() else 0.0
    return np.where(finite, E, cap).astype(np.float32)


def _flux_median_slope(P: np.ndarray, L: int, A: int, bins: int = 2048) -> float:
    """Default length scale lam = flux-weighted median of |dE| over substitution edges,
    with flux sqrt(P_i P_j): a typical edge then has equal hop and energetic length
    (l_ij = sqrt(1 + (dE/lam)^2) ~ sqrt(2) at the median).  Computed via a flux-weighted
    histogram over ``bins`` bins (memory-light; exact to the bin width)."""
    with np.errstate(divide="ignore"):
        E = -np.log(P) / np.log(2)
    E = _cap_infinite(E)
    P = np.asarray(P, np.float32)
    hi = float(E.max() - E.min())
    hist = np.zeros(bins, np.float64)
    for p in range(L):
        Ep = np.moveaxis(E, p, 0).reshape(A, -1)
        Pp = np.moveaxis(P, p, 0).reshape(A, -1)
        for a in range(A):
            for b in range(a + 1, A):
                de = np.abs(Ep[a] - Ep[b])
                w = np.sqrt(Pp[a] * Pp[b])
                idx = np.clip((de / hi * bins).astype(np.int64), 0, bins - 1)
                np.add.at(hist, idx, w)
    cdf = np.cumsum(hist)
    med = int(np.searchsorted(cdf, 0.5 * cdf[-1]))
    return float((med + 0.5) / bins * hi) if hi > 0 else 1e-3


def energy_metric_layout(E: np.ndarray, P: np.ndarray, L: int, A: int, lam: float,
                         K: int = 200, seed: int = 0, beta: float = 0.5) -> np.ndarray:
    """The energy-metric layout: landmark-MDS (Isomap) of the geodesic distances of the
    substitution graph weighted by edge length sqrt(1 + (dE/lam)^2).  lam is the energy
    per layout unit: lam->inf gives the pure graph (Hamming) geometry, lam->0 gives pure
    energy order.  Returns (N, 2) float32 coordinates, deterministically oriented."""
    lam = max(float(lam), 1e-9)          # guard against lam = 0 (constant energy)
    csr = _ham_graph_csr(E, L, A, lam)
    D, _ = _p_farthest_landmarks(csr, P, K, beta, seed)
    del csr
    coords = _landmark_mds(D)
    return _orient_layout(coords, E, P).T.astype(np.float32)


def _energy_locality_axes(P: np.ndarray, E: np.ndarray, L: int, A: int,
                           ell: float) -> np.ndarray:
    """Linear energy-weighted locality: the two slowest linear coordinates of the
    substitution graph weighted by w_st = exp(-(dE/ell)^2) (a length scale ell on the
    energy difference), with the P-weighted second moment as the denominator.  This is
    the linear restriction of the energy-weighted Laplacian eigenmap; the objective is
    literally 'put strings one substitution apart AND of similar energy close together'."""
    d = L * A
    XLX = np.zeros((d, d))
    Ep_all = E.reshape((A,) * L)
    for p in range(L):
        Ep = np.moveaxis(Ep_all, p, 0).reshape(A, -1)
        for a in range(A):
            for b in range(a + 1, A):
                c = float(np.exp(-((Ep[a] - Ep[b]) ** 2) / ell ** 2).sum())
                ia, ib = p * A + a, p * A + b
                XLX[ia, ia] += c
                XLX[ib, ib] += c
                XLX[ia, ib] -= c
                XLX[ib, ia] -= c
    C0 = _indicator_covariance(P)
    mu = np.concatenate([P.sum(axis=tuple(k for k in range(L) if k != j))
                         for j in range(L)])
    B = C0 + np.outer(mu, mu)
    B = B + 1e-8 * np.trace(B) * np.eye(d)
    lam, V = scipy.linalg.eigh(XLX, B)
    null = int(np.sum(lam < 1e-8 * max(1.0, float(lam[-1]))))
    U = V[:, null:null + 2]
    return _sign_fix(U / np.linalg.norm(U, axis=0))


def energy_rank_layout(E: np.ndarray, L: int, A: int) -> np.ndarray:
    """Degenerate control (lam->0 limit): strings laid out in a boustrophedon raster
    sorted by energy, so the surface is the quantile function of E and x,y encode only
    the energy rank.  Returns (N, 2) float32."""
    N = A ** L
    E = np.asarray(E).ravel()
    rank = np.argsort(np.argsort(E, kind="stable"))           # rank per node (0..N-1)
    width = int(np.ceil(np.sqrt(N)))
    row = rank // width
    col = rank % width
    col = np.where((row % 2) == 1, width - 1 - col, col)      # boustrophedon: odd rows
    coords = np.stack([col, row]).astype(np.float32)
    coords = coords - coords.mean(axis=1, keepdims=True)
    return coords


def hamming_layout(P: np.ndarray, L: int, A: int, K: int = 200, seed: int = 0,
                   beta: float = 0.5) -> np.ndarray:
    """lambda->inf control: landmark MDS of the pure Hamming geodesic distance
    (the substitution-graph geometry with no energy; the geodesic is the Hamming
    distance, computed in closed form).  Returns (2, N) float32."""
    N = A ** L
    digits = np.stack(np.unravel_index(np.arange(N), (A,) * L)).T        # (N, L)
    w = np.asarray(P, np.float64).ravel() ** beta
    landmarks = [int(np.argmax(w))]
    dmin = np.full(N, np.inf)
    D = []
    for _ in range(K):
        d = L - (digits[landmarks[-1]] == digits).sum(axis=1).astype(np.float64)
        D.append(d)
        dmin = np.minimum(dmin, d)
        score = dmin * w
        score[landmarks] = -1.0
        landmarks.append(int(np.argmax(score)))
    coords = _landmark_mds(np.stack(D))
    E = np.zeros(N, np.float32)
    return _orient_layout(coords, E, P).T.astype(np.float32)


def occupied_components(occ: np.ndarray, G: int, k: int = 4) -> tuple[int, int]:
    """Number of k-connected components (k in {4, 8}) of the occupied cells of a G x G
    grid, and the size of the largest.  ``occ`` is a length-G*G boolean mask (row-major,
    cell = row*G + col); edges join occupied cells only, so the count reflects the actual
    connected structure of the occupied cells."""
    from scipy.sparse import coo_matrix as _coo
    from scipy.sparse.csgraph import connected_components as _cc
    offs = ([(1, 0), (-1, 0), (0, 1), (0, -1)] if k == 4 else
            [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)])
    rows, cols = [], []
    for c in np.flatnonzero(occ):
        r, cc = divmod(int(c), G)
        for dr, dc in offs:
            rr, rcc = r + dr, cc + dc
            if 0 <= rr < G and 0 <= rcc < G:
                j = rr * G + rcc
                if occ[j] and j > c:
                    rows.append(c)
                    cols.append(j)
    g = _coo((np.ones(len(rows)), (rows, cols)), shape=(G * G, G * G)).tocsr()
    _, lab = _cc(g, directed=False)
    ol = lab[occ]
    return int(len(np.unique(ol))), int(np.bincount(ol).max())
