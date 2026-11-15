"""Two-dimensional views of a maximum-entropy energy landscape over A^L strings.

A string s is represented by its one-hot indicator vector x(s) over (position, letter)
pairs, of dimension L*A. Three facts make the principal axes of x under the model the
natural plane for viewing the landscape:

* For the exponential family P(s) ∝ exp(Σ_i h_i(s_i) + Σ_{i<j} V_ij(s_i, s_j)), the
  covariance C = Cov_P[x] is the Fisher information with respect to the single-site
  fields h (Amari & Nagaoka, *Methods of Information Geometry*, 2000). It is computed
  exactly from the model's single-site and pairwise marginals:
  C[(i,a),(j,b)] = P_ij(a,b) - P_i(a) P_j(b), with diagonal blocks diag(P_i) - P_i P_i^T.
* ||x(s) - x(s')||^2 = 2 Hamming(s, s'), so PCA of x weighted by P is classical
  multidimensional scaling (Torgerson 1952) of the Hamming metric; one substitution
  (Stephens & Bialek's move) changes x by a vector of norm sqrt(2), so it moves a
  string by at most sqrt(2) in any orthonormal projection.
* For a fit that matches the empirical pairwise marginals exactly, C equals the
  empirical indicator covariance.

Heights are bin energies E_bin = -log2 <2^{-E(s)}>_{s in bin} = -log2(P(bin)/|bin|), the
Zwanzig (1954) exponential average of the string energies in a grid cell. Wells are
local minima of E_bin on a graph of cells (the 8-neighbour grid, or cells linked by a
single-letter substitution) ranked by 0-dimensional sublevel-set persistence
(Edelsbrunner, Letscher & Zomorodian 2002).

For display only, a grid of cell values can be replaced by a smooth surface via a
Gaussian (Boltzmann-domain) filter implemented in the browser (landscape_engine.js
``smoothGrid``), not in this module.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

# Neighbourhood on the projection grid: cells sharing an edge or a corner.
NEIGHBOURS_8 = tuple((di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0))


# --------------------------------------------------------------------------- #
# covariance and principal axes
# --------------------------------------------------------------------------- #

def site_marginals(P: np.ndarray) -> np.ndarray:
    """(L, A) single-site marginals of a dense A^L probability tensor."""
    L = P.ndim
    return np.stack([P.sum(axis=tuple(k for k in range(L) if k != i)) for i in range(L)])


def pair_marginals(P: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    L = P.ndim
    return {(i, j): P.sum(axis=tuple(k for k in range(L) if k not in (i, j)))
            for i, j in combinations(range(L), 2)}


def indicator_covariance(P: np.ndarray) -> np.ndarray:
    """Exact covariance of the one-hot (position, letter) indicators under P.

    Index (i, a) -> i*A + a. Uses only the single-site and pairwise marginals, the
    sufficient statistics of a pairwise maximum-entropy model.
    """
    L, A = P.ndim, P.shape[0]
    p1 = site_marginals(P)
    C = np.empty((L * A, L * A))
    for i in range(L):
        C[i * A:(i + 1) * A, i * A:(i + 1) * A] = np.diag(p1[i]) - np.outer(p1[i], p1[i])
    for (i, j), pij in pair_marginals(P).items():
        blk = pij - np.outer(p1[i], p1[j])
        C[i * A:(i + 1) * A, j * A:(j + 1) * A] = blk
        C[j * A:(j + 1) * A, i * A:(i + 1) * A] = blk.T
    return C


def principal_axes(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and unit eigenvectors (columns) of C.

    Sign convention: each eigenvector's largest-|loading| entry is positive.
    """
    lam, U = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, U = np.clip(lam[order], 0.0, None), U[:, order]
    piv = np.abs(U).argmax(axis=0)
    U = U * np.sign(U[piv, np.arange(U.shape[1])])
    return lam, U


def axis_report(u: np.ndarray, lam: np.ndarray, k: int, L: int, alphabet: str,
                top: int) -> dict:
    """Composition of principal axis k: variance share, squared loadings by position
    (they sum to 1 because u is a unit vector) and the largest signed loadings."""
    A = len(alphabet)
    sq = u ** 2
    order = np.argsort(-np.abs(u))[:top]
    return {"variance_share": float(lam[k] / lam.sum()),
            "eigenvalue": float(lam[k]),
            "position_share": sq.reshape(L, A).sum(axis=1).tolist(),
            "top_loadings": [(int(i // A) + 1, alphabet[i % A], float(u[i]), float(sq[i]))
                             for i in order]}


def project(U: np.ndarray, mean: np.ndarray, L: int, A: int) -> np.ndarray:
    """Coordinates (k, A^L) of every string: U^T (x(s) - mean), by broadcasting.

    x(s) has one 1 per position, so U^T x(s) = Σ_i U[(i, s_i)], a sum of L terms.
    """
    k = U.shape[1]
    out = np.zeros((k,) + (A,) * L)
    for i in range(L):
        shape = [1] * L
        shape[i] = A
        out += U[i * A:(i + 1) * A].T.reshape((k, *shape))
    return out.reshape(k, -1) - (U.T @ mean)[:, None]


# --------------------------------------------------------------------------- #
# grid and bin energies
# --------------------------------------------------------------------------- #

def grid_edges(coords: np.ndarray, n: int) -> np.ndarray:
    """(2, n+1) equally spaced edges spanning every string's coordinates.

    The grid depends only on the basis (all A^L strings are binned, whatever their
    probability), so it is shared by every model viewed in that basis.
    """
    return np.stack([np.linspace(c.min(), c.max(), n + 1) for c in coords[:2]])


def bin_index(coords: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Flat cell index (row = y bin, column = x bin) of every string."""
    n = edges.shape[1] - 1
    ij = []
    for c, e in zip(coords[:2], edges):
        k = np.floor((c - e[0]) / (e[-1] - e[0]) * n).astype(np.int64)
        ij.append(np.clip(k, 0, n - 1))
    return ij[1] * n + ij[0]


def bin_energy(log2p: np.ndarray, bins: np.ndarray, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """E_bin = -log2( P(bin) / |bin| ) in bits, and |bin|.

    log2p: log2 P(s) per string (-inf for forbidden strings). Cells with no strings
    are NaN; cells whose strings all have P = 0 are +inf. Computed as a per-cell
    log-sum-exp so tiny probabilities do not underflow.
    """
    count = np.bincount(bins, minlength=n_cells).astype(float)
    top = np.full(n_cells, -np.inf)
    np.maximum.at(top, bins, log2p)
    shift = np.where(np.isfinite(top), top, 0.0)
    with np.errstate(invalid="ignore"):
        s = np.bincount(bins, weights=np.exp2(log2p - shift[bins]), minlength=n_cells)
    with np.errstate(divide="ignore", invalid="ignore"):
        E = -(np.log2(s) + shift - np.log2(count))
    E[~np.isfinite(top)] = np.inf
    E[count == 0] = np.nan
    return E, count


# --------------------------------------------------------------------------- #
# adjacency of grid cells
# --------------------------------------------------------------------------- #

def grid_adjacency(shape: tuple[int, int]) -> list[np.ndarray]:
    """8-neighbour adjacency of the cells of a 2-D grid (flat indices)."""
    out = []
    for k in range(shape[0] * shape[1]):
        r, c = divmod(k, shape[1])
        out.append(np.array([(r + dr) * shape[1] + c + dc for dr, dc in NEIGHBOURS_8
                             if 0 <= r + dr < shape[0] and 0 <= c + dc < shape[1]], dtype=np.int64))
    return out


def substitution_adjacency(bins: np.ndarray, n_cells: int) -> list[np.ndarray]:
    """Cells u != v are adjacent iff some string in u and some string in v differ by
    one letter substitution (Stephens & Bialek's move). ``bins`` is the cell of every
    string as an A^L tensor. Depends only on the binning, not on any model."""
    L, A = bins.ndim, bins.shape[0]
    codes = []
    for i in range(L):
        for a in range(A):
            u = np.take(bins, a, axis=i).ravel()
            for b in range(a + 1, A):
                v = np.take(bins, b, axis=i).ravel()
                m = u != v
                lo, hi = np.minimum(u[m], v[m]), np.maximum(u[m], v[m])
                codes.append(np.unique(lo * n_cells + hi))
    pairs = np.unique(np.concatenate(codes))
    lo, hi = np.divmod(pairs, n_cells)
    src, dst = np.r_[lo, hi], np.r_[hi, lo]
    order = np.argsort(src, kind="stable")
    split = np.searchsorted(src[order], np.arange(n_cells + 1))
    return [dst[order][split[k]:split[k + 1]] for k in range(n_cells)]


# --------------------------------------------------------------------------- #
# wells: sublevel-set persistence on a cell graph
# --------------------------------------------------------------------------- #

def persistence(E: np.ndarray, adj: list[np.ndarray]) -> list[dict]:
    """0-dimensional persistence of the sublevel sets of a function on a graph.

    E: value per cell (any shape; flattened). adj[k]: neighbours of cell k. Only
    finite cells take part: empty (NaN) and all-forbidden (+inf) cells are holes.
    Cells are added in increasing (value, index) order; a cell with no earlier
    neighbour births a component at a local minimum; a cell joining several
    components kills all but the one with the lowest birth (the elder rule). Each
    minimum is returned with its birth (E_min), death (the saddle height E_saddle),
    persistence = death - birth, the saddle cell, and the minimum it merged into.

    The minimum of a connected component never dies ("essential"). By convention it
    is paired with the component's highest value, so its persistence is the full
    height range of the component: a flat landscape has no well at all, and the
    essential minimum always outranks every other well in its component.
    """
    flat = np.asarray(E, dtype=float).ravel()
    order = sorted(np.flatnonzero(np.isfinite(flat)), key=lambda k: (flat[k], k))
    parent = {}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    root_min = {}                    # union-find root -> its component's minimum cell
    out = {}
    for k in order:
        roots = {find(int(n)) for n in adj[k] if int(n) in parent}
        parent[k] = k
        if not roots:
            root_min[k] = k
            out[k] = {"cell": int(k), "E_min": float(flat[k]), "E_saddle": np.inf,
                      "persistence": np.inf, "saddle": None, "merges_into": None,
                      "essential": True}
            continue
        mins = sorted((root_min[r] for r in roots), key=lambda m: (flat[m], m))
        elder = mins[0]
        for r in roots:
            parent[r] = k
        root_min[k] = elder
        for m in mins[1:]:
            out[m].update(E_saddle=float(flat[k]), persistence=float(flat[k] - flat[m]),
                          saddle=int(k), merges_into=int(elder), essential=False)
    top = {}                         # essential minimum -> its component's highest value
    for k in order:                  # ascending, so the last write is the maximum
        top[root_min[find(k)]] = float(flat[k])
    for m, hi in top.items():
        out[m]["persistence"] = hi - out[m]["E_min"]
    # Plateaus give zero-persistence pairs (a tie broken by index makes a cell a
    # minimum of the total order); they are kept so every descent basin has a node
    # in the merge tree, and are never wells at any positive threshold.
    return sorted(out.values(), key=lambda w: (-w["persistence"], -w["essential"], w["E_min"]))


def descent_basins(E: np.ndarray, adj: list[np.ndarray]) -> np.ndarray:
    """Steepest-descent basin of every finite cell: the minimum reached by repeatedly
    moving to the lowest neighbour while it is lower in (value, index) order.
    Non-finite cells get -1. Same shape as E."""
    flat = np.asarray(E, dtype=float).ravel()
    nxt = np.full(flat.size, -1)
    for k in np.flatnonzero(np.isfinite(flat)):
        best = k
        for n in adj[k]:
            if np.isfinite(flat[n]) and (flat[n], n) < (flat[best], best):
                best = int(n)
        nxt[k] = best
    lab = nxt.copy()
    while True:
        new = np.where(lab >= 0, nxt[np.maximum(lab, 0)], -1)
        if np.array_equal(new, lab):
            return lab.reshape(np.shape(E))
        lab = new


def significant_basins(E: np.ndarray, wells: list[dict], threshold: float,
                       adj: list[np.ndarray]) -> np.ndarray:
    """Basins after cancelling every well with persistence < threshold: each
    steepest-descent basin is relabelled with the first minimum, following the merge
    tree upward, whose persistence reaches the threshold (or with its component's
    essential minimum, which is never cancelled into anything)."""
    by_cell = {w["cell"]: w for w in wells}

    def rep(m):
        while by_cell[m]["persistence"] < threshold and not by_cell[m]["essential"]:
            m = by_cell[m]["merges_into"]
        return m

    lab = descent_basins(E, adj)
    rmap = {m: rep(m) for m in by_cell}
    return np.vectorize(lambda m: rmap.get(int(m), -1), otypes=[np.int64])(lab)


# --------------------------------------------------------------------------- #
def contour_levels(z: np.ndarray, n: int = 9, digits: int = 1) -> list[float]:
    """Evenly spaced contour levels (in bits) spanning the finite range of a grid,
    for the heatmap's contour lines. ``n`` interior levels (``n + 1`` intervals)."""
    fin = z[np.isfinite(z)]
    if fin.size < 2 or fin.min() == fin.max():
        return []
    return [round(float(v), digits) for v in np.linspace(fin.min(), fin.max(), n + 2)[1:-1]]
