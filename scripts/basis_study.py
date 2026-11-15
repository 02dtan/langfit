"""Compare candidate 2-D projection bases for the S&B four-letter landscape.

Loads the cached pairwise couplings from ``data/derived/landscape_artifact.npz``
(the same tensors the artifact renders), rebuilds each Boltzmann distribution
exactly, and scores several candidate bases on the metrics of the brief:

  * kinetic relevance      — VAMP-2 (Wu & Noe 2020) and implied timescales of the
                             random-site heat-bath (Gibbs) substitution chain;
  * landscape faithfulness — P-weighted basin separation (1-nearest-neighbour in the
                             plane keeps string-level descent basins distinct);
  * neighbourhood quality  — trustworthiness / continuity (Venna & Kaski 2001)
                             between Hamming and plane neighbourhoods, P-weighted;
  * coverage continuity    — isolated grid components, empty cells inside the hull,
                             dynamic range of occupied-cell mass;
  * interpretability       — per-(position, letter) loadings for linear bases;
  * time stability         — how the fixed basis changes when fit on intermediate
                             trajectory frames (trajectory model only).

Output: ``data/derived/basis_study.json`` (an evidence table plus the recommended
basis's per-model annotation data, in the same shape the artifact template expects).

Everything is exact: the chain transition is applied by fibre averages over all 26^4
strings (see ``langfit/basis.py``); the only sampled quantity is the trustworthiness
estimate (P-weighted by sampling N_SAMPLE strings). Runtime is a few minutes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix

import langfit.maxent as me
import langfit.projection as pj
from langfit import basis as bs

DATA = Path("data/derived")
CACHE_NPZ = DATA / "landscape_artifact.npz"
CACHE_JSON = DATA / "landscape_artifact.json"
OUT = DATA / "basis_study.json"

L4, A = 4, me.A
ALPHABET = me.ALPHABET

# ---- numerical / display parameters (all stated, none magic) ----
GRID = 32                       # bins per axis, matching the artifact's projection grid
LAGS = [1, 4, 8]                # single-site updates; tau = L4 = 4 is one sweep
TAU_EVAL = 4                    # the lag at which VAMP-2 is scored (one sweep)
K_NN = 10                       # neighbourhood size for trustworthiness/continuity
N_SAMPLE = 2000                 # P-weighted sample size for trustworthiness
VAR_KEEP = 0.95                 # covariance subspace kept for regularised TICA/LDA
SEED = 0

MODELS = (
    ("austen", "(a) Austen, S&B protocol"),
    ("twitter2023", "(b) Twitter 2023, N-matched"),
    ("bluesky2024", "(c) Bluesky 2024, N-matched"),
    ("pooled", "(d) Twitter pooled 2006-03..2023-04, MAP"),
)


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #

def load_models() -> dict[str, np.ndarray]:
    npz = np.load(CACHE_NPZ)
    js = json.loads(CACHE_JSON.read_text())
    P = {}
    for key, _ in MODELS:
        if key == "pooled":
            V = npz["V_traj"][-1]
        else:
            V = npz[f"V_{key}"]
        m = me.MaxEnt(L=L4, cliques=me.cliques(2), V=[v.copy() for v in V])
        P[key] = m.prob()
    npz.close()
    return P


def load_trajectory() -> tuple[np.ndarray, np.ndarray]:
    """(frames, tokens): the pooled trajectory couplings and per-frame token counts."""
    npz = np.load(CACHE_NPZ)
    V, tok = npz["V_traj"], npz["traj_tokens"]
    npz.close()
    return V, tok


def prob_from_V(V: np.ndarray) -> np.ndarray:
    m = me.MaxEnt(L=L4, cliques=me.cliques(2), V=[v.copy() for v in V])
    return m.prob()


# --------------------------------------------------------------------------- #
# string-level descent basins (for LDA and the landscape-faithfulness metric)
# --------------------------------------------------------------------------- #

def descent_basin_labels(P: np.ndarray) -> np.ndarray:
    """Flat int label per string: the strict local minimum its steepest descent reaches
    (a support string's terminal state is always a strict minimum, verified)."""
    with np.errstate(divide="ignore"):
        E = -np.log(P)
    return me.descend(E).astype(np.int64).ravel()


def string_statistics(P: np.ndarray) -> dict:
    with np.errstate(divide="ignore"):
        E = -np.log(P)
    mins = me.local_minima(E)
    sup = P.ravel() > 0
    labels = descent_basin_labels(P)[sup]
    n_basins = len(np.unique(labels))
    w = P.ravel()[sup]
    return {
        "support": int(sup.sum()),
        "n_strict_minima": int(mins.sum()),
        "n_basins": n_basins,
        "mass_strict_minima": float(P[mins].sum()),
        "entropy_bits": float(-(w * np.log2(w)).sum()),
    }


# --------------------------------------------------------------------------- #
# candidate bases  (each returns (name, V) with V of shape (104, 2))
# --------------------------------------------------------------------------- #

def truncated_subspace(C0: np.ndarray, frac: float) -> np.ndarray:
    """Leading eigenvectors of C0 whose cumulative variance reaches ``frac`` (104, d)."""
    lam, U = np.linalg.eigh(C0)
    order = np.argsort(lam)[::-1]
    lam, U = lam[order], U[:, order]
    keep = np.cumsum(lam) <= frac * lam.sum()
    keep[-1] = True
    return U[:, keep]


def axes_in_subspace(M: np.ndarray, C0: np.ndarray, subspace: np.ndarray) -> np.ndarray:
    """Generalised axes of M vs C0 restricted to ``subspace`` (avoids the rare-letter
    blow-up of whitening tiny covariance directions)."""
    Ms = subspace.T @ M @ subspace
    lam, R = np.linalg.eigh(Ms)
    order = np.argsort(lam)[::-1]
    V = subspace @ R[:, order]
    return bs._sign_fix(V)


def candidate_bases(C0: np.ndarray, r: dict) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    # (a) current basis: PCA
    lam, U = bs.pca_axes(C0)
    out["pca"] = U[:, :2]
    # (b) TICA at each lag (raw, and regularised in the leading-PC subspace)
    sub = truncated_subspace(C0, VAR_KEEP)
    for t in LAGS:
        Ct = r["C"][t]
        _, Vt = bs.tica_axes(C0, Ct)
        out[f"tica{t}"] = Vt[:, :2]
        out[f"tica{t}-reg"] = axes_in_subspace(Ct, C0, sub)[:, :2]
    return out


def _lda_scatter(P: np.ndarray, labels: np.ndarray, mean: np.ndarray) -> np.ndarray:
    _, pi, mu = bs.class_means(P, labels)
    pi = pi / pi.sum()
    D = mu - mean
    return (D * pi[:, None]).T @ D


def _candidate_lda(P: np.ndarray, labels: np.ndarray, C0: np.ndarray,
                   mean: np.ndarray) -> np.ndarray:
    SB = _lda_scatter(P, labels, mean)
    return axes_in_subspace(SB, C0, truncated_subspace(C0, VAR_KEEP))[:, :2]


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def coords_of(P: np.ndarray, V: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """(2, A^L) plane coordinates of every string (V: 104 x 2)."""
    return bs.coords_linear(V, mean, L4, A)


def vamp2_score(V: np.ndarray, C0: np.ndarray, r: dict, tau: int) -> float:
    C0y = V.T @ C0 @ V
    Cty = V.T @ r["C"][tau] @ V
    return bs.vamp2(C0y, Cty)


def autocorr_and_timescales(V: np.ndarray, C0: np.ndarray, r: dict) -> dict:
    C0y = V.T @ C0 @ V
    Cty = V.T @ r["C"][1] @ V
    ac = bs.autocorrelations(C0y, Cty)
    return {"autocorrelations": ac.tolist(),
            "implied_timescales": bs.implied_timescales(ac, 1).tolist()}


def variance_share(V: np.ndarray, C0: np.ndarray) -> float:
    """Share of total indicator variance captured by projecting onto span(V)
    (Euclidean-orthonormalised, so it obeys Eckart-Young: <= (lam1+lam2)/trace)."""
    Q, _ = np.linalg.qr(V)
    return float(np.trace(Q.T @ C0 @ Q) / np.trace(C0))


def basin_separation(P: np.ndarray, coords: np.ndarray, labels: np.ndarray,
                     mean: np.ndarray) -> dict:
    """P-weighted fraction of support strings whose plane-nearest support string is in
    the same descent basin (a string-level basin is 'resolved' iff its nearest
    projected neighbour shares its basin).  Uses a KD-tree over support coordinates."""
    from scipy.spatial import cKDTree
    sup = P.ravel() > 0
    w = P.ravel()[sup]
    lab = labels[sup]
    xy = coords[:, sup].T
    tree = cKDTree(xy)
    # k=3 to skip the point itself (and any exact-coordinate duplicates)
    _, idx = tree.query(xy, k=3)
    nbr = idx[:, 1]
    same = lab[nbr] == lab
    return {
        "mass_same_basin_nn": float((w * same).sum()),
        "n_strings": int(sup.sum()),
        "n_basins": len(np.unique(lab)),
    }


def trustworthiness_continuity(P: np.ndarray, coords: np.ndarray, k: int = K_NN,
                               n: int = N_SAMPLE, seed: int = SEED) -> dict:
    """Venna & Kaski (2001) trustworthiness and continuity, P-weighted by sampling.

    High-dimensional neighbourhoods are Hamming (single-letter substitution) on the
    sampled strings; low-dimensional are Euclidean in the plane."""
    rng = np.random.default_rng(seed)
    sup = P.ravel() > 0
    w = P.ravel()[sup]
    digits = np.stack(np.unravel_index(np.flatnonzero(sup), (A,) * L4)).T  # (n, L)
    idx = rng.choice(len(w), size=n, replace=True, p=w / w.sum())
    D = digits[idx]                              # (n, L)
    X_hi = (D[:, None, :] != D[None, :, :]).sum(axis=2).astype(float)   # Hamming (n, n)
    X_lo = cdist(coords[:, sup].T[idx], coords[:, sup].T[idx])
    return {
        "trustworthiness": _trustworthiness(X_hi, X_lo, k),
        "continuity": _trustworthiness(X_lo, X_hi, k),
        "k": k, "n_sample": n,
    }


def _trustworthiness(X_hi: np.ndarray, X_lo: np.ndarray, k: int) -> float:
    """T = 1 - 2/(Nk(2N-3k-1)) * sum_i sum_{j in U_k(i)} (r(i,j) - k),
    U_k(i) = k-NN of i in the *low*-dim space, r = rank in the *high*-dim order."""
    N = len(X_hi)
    rank_hi = np.argsort(np.argsort(X_hi, axis=1), axis=1)      # rank of j for i
    rank_lo = np.argsort(X_lo, axis=1)                          # sorted neighbour order
    pen = 0.0
    for i in range(N):
        U = rank_lo[i, 1:k + 1]                                  # exclude self
        pen += (rank_hi[i, U] - k).sum()
    return float(1.0 - 2.0 * pen / (N * k * (2 * N - 3 * k - 1)))


def coverage(P: np.ndarray, coords: np.ndarray, grid: int = GRID) -> dict:
    """Continuity of coverage on a grid×grid lattice spanning all strings' coordinates."""
    sup = P.ravel() > 0
    w = P.ravel()
    edges = pj.grid_edges(coords, grid)
    bins = pj.bin_index(coords, edges)           # flat cell index per string
    mass = np.bincount(bins[sup], weights=w[sup], minlength=grid * grid)
    count = np.bincount(bins[sup], minlength=grid * grid)
    occ = mass > 0                                # occupied cells
    # isolated mass: P-mass in occupied cells not in the largest 8-connected component
    adj = pj.grid_adjacency((grid, grid))
    rows = np.repeat(np.arange(grid * grid), [len(a) for a in adj])
    cols = np.concatenate(adj)
    n_comp, lab = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)),
                                                  shape=(grid * grid, grid * grid)),
                                       directed=False)
    big = int(np.argmax(np.bincount(lab[occ])))
    iso = occ & (lab != big)
    # hull emptiness: empty cells inside the convex hull of occupied-cell centres
    cy, cx = np.unravel_index(np.flatnonzero(occ), (grid, grid))
    hull_empty = _hull_emptiness(cy, cx, occ, grid)
    pos = mass[mass > 0]
    return {
        "grid": grid,
        "n_occupied": int(occ.sum()),
        "isolated_mass": float(mass[iso].sum()),
        "isolated_n_cells": int(iso.sum()),
        "hull_empty_fraction": float(hull_empty),
        "occupancy_log10_range": float(np.log10(pos.max() / pos.min())) if len(pos) else None,
        "min_cell_mass": float(pos.min()) if len(pos) else None,
    }


def _hull_emptiness(cy: np.ndarray, cx: np.ndarray, occ: np.ndarray,
                    grid: int) -> float:
    from scipy.spatial import ConvexHull
    if len(cy) < 3:
        return 0.0
    pts = np.column_stack([cx, cy])
    hull = ConvexHull(pts)
    # grid cell centres inside the hull
    ys, xs = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    q = np.column_stack([xs.ravel(), ys.ravel()])
    inside = _points_in_hull(q, pts[hull.vertices])
    empty_inside = inside & ~occ.ravel()
    return float(empty_inside.sum() / max(1, inside.sum()))


def _points_in_hull(q: np.ndarray, hull_pts: np.ndarray) -> np.ndarray:
    """Point-in-convex-hull test (all hull edge normals point outward)."""
    from scipy.spatial import ConvexHull
    h = ConvexHull(hull_pts)
    eq = h.equations                       # [A, b] with A x + b <= 0 inside
    return np.all(q @ eq[:, :2].T + eq[:, 2] <= 1e-9, axis=1)


def outliers(P: np.ndarray, coords: np.ndarray) -> dict:
    """How far the tail of the P-mass sits from the P-weighted centroid (the brief's
    'isolated points far from the data mass').  Distances are in units of the
    P-weighted RMS radius of the cloud."""
    sup = P.ravel() > 0
    w = P.ravel()[sup]
    xy = coords[:, sup].T                     # (n, 2)
    mu = (w[:, None] * xy).sum(0) / w.sum()
    d = np.linalg.norm(xy - mu, axis=1)
    rms = float(np.sqrt((w * d ** 2).sum() / w.sum()))
    q = d / rms                               # in RMS-radius units
    order = np.argsort(q)
    w_sorted = w[order]
    q_sorted = q[order]
    cdf = np.cumsum(w_sorted) / w_sorted.sum()
    def quantile(f):
        i = np.searchsorted(cdf, f, side="right")
        return float(q_sorted[min(i, len(q_sorted) - 1)])
    return {
        "max_rms_units": float(q.max()),
        "p999_rms_units": quantile(0.999),
        "p99_rms_units": quantile(0.99),
        "mass_beyond_5rms": float(w[q > 5].sum()),
    }


# --------------------------------------------------------------------------- #
# annotation (linear bases): reuse the artifact's axis-report shape
# --------------------------------------------------------------------------- #

def annotate(V: np.ndarray, C0: np.ndarray, mean: np.ndarray, top: int = 10) -> list[dict]:
    """Per-axis annotation: variance share, squared loadings by position, top loadings."""
    lam = np.array([float(v @ C0 @ v) for v in V.T])
    reports = []
    for k in range(V.shape[1]):
        v = V[:, k]
        share = float(lam[k] / np.trace(C0))
        pos_share = [float((v[i * A:(i + 1) * A] ** 2).sum()) for i in range(L4)]
        piv = int(np.argmax(np.abs(v)))
        sgn = np.sign(v[piv])
        order = np.argsort(-np.abs(v))
        top_loadings = []
        for i in order[:top]:
            pos, let = divmod(int(i), A)
            top_loadings.append([pos, ALPHABET[let], float(v[i] * sgn),
                                 float(v[i] ** 2)])
        reports.append({"variance_share": share, "eigenvalue": lam[k],
                        "position_share": pos_share, "top_loadings": top_loadings})
    return reports


# --------------------------------------------------------------------------- #
# time stability (trajectory model only)
# --------------------------------------------------------------------------- #

def _subspace_angle(Va: np.ndarray, Vb: np.ndarray) -> float:
    """Principal angle (degrees) between two 2-D subspaces (columns = basis)."""
    Qa, _ = np.linalg.qr(Va)
    Qb, _ = np.linalg.qr(Vb)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.rad2deg(np.arccos(s.min())))


def time_stability(kinds: list[str], n_frames: int = 12) -> dict:
    """Change of each basis when fit on intermediate trajectory frames vs the final
    frame, measured by principal angles of the 2-D subspaces.  Per-frame covariances
    are computed once and shared across kinds."""
    V, _ = load_trajectory()
    final = prob_from_V(V[-1])
    Cf = pj.indicator_covariance(final)
    rf = bs.indicator_lagged_covariances(final, lags=LAGS, chunk_positions=1)
    labels_f = descent_basin_labels(final)
    basis_final = {k: _basis_of_kind(k, final, Cf, rf, labels_f, rf["mean"]) for k in kinds}

    # the last ``n_frames`` frames (near-converged) plus a coarse sample of earlier ones
    last = np.arange(len(V) - n_frames, len(V))
    early = np.linspace(0, len(V) - n_frames - 1, 6).astype(int)
    frames = np.unique(np.concatenate([early, last]))
    angles = {k: [] for k in kinds}
    for f in frames:
        P = prob_from_V(V[f])
        C0 = pj.indicator_covariance(P)
        r = bs.indicator_lagged_covariances(P, lags=LAGS, chunk_positions=1)
        labels = descent_basin_labels(P)
        for k in kinds:
            Vb = _basis_of_kind(k, P, C0, r, labels, r["mean"])
            angles[k].append(_subspace_angle(Vb, basis_final[k]))
    out = {}
    for k in kinds:
        a = np.array(angles[k])
        out[k] = {"frames": frames.tolist(), "principal_angles_deg": a.tolist(),
                  "max_deg": float(a.max()), "median_deg": float(np.median(a)),
                  "max_last12_deg": float(a[-n_frames:].max()),
                  "median_last12_deg": float(np.median(a[-n_frames:]))}
    return out


def _basis_of_kind(kind: str, P, C0, r, labels, mean) -> np.ndarray:
    if kind == "pca":
        return bs.pca_axes(C0)[1][:, :2]
    if kind == "tica4":
        return bs.tica_axes(C0, r["C"][4])[1][:, :2]
    if kind == "tica4-reg":
        return axes_in_subspace(r["C"][4], C0, truncated_subspace(C0, VAR_KEEP))[:, :2]
    if kind == "lda":
        return _candidate_lda(P, labels, C0, mean)
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def run() -> None:
    t0 = time.time()
    P_all = load_models()
    out = {
        "method": {
            "chain": "random-site heat-bath (Gibbs) update, stationary distribution P; "
                     "time counted in single-site updates (L=4 updates = one sweep)",
            "candidates": {
                "pca": "top-2 eigenvectors of the P-weighted indicator covariance C(0)",
                "tica1/4/8": "TICA / linear VAC: C(tau) v = lam C(0) v, lag tau",
                "tica*-reg": "TICA in the leading-PC subspace keeping VAR_KEEP of variance",
                "lda": "Fisher LDA on string-level descent basins (regularised)",
            },
            "metrics": {
                "vamp2_tau": "VAMP-2 (Wu & Noe 2020) of the 2-D basis at lag TAU_EVAL",
                "implied_timescales": "t = -1/ln(autocorrelation) in single-site updates",
                "variance_share": "share of total indicator variance in the plane",
                "basin_separation": "P-weighted fraction of strings whose plane-nearest "
                                    "support string is in the same descent basin",
                "trustworthiness/continuity": "Venna & Kaski (2001), P-weighted by "
                                              "sampling N_SAMPLE strings, k=K_NN",
                "coverage": "grid=GRID; isolated_mass = P-mass in cells outside the "
                            "largest 8-connected occupied component",
            },
            "parameters": {"grid": GRID, "lags": LAGS, "tau_eval": TAU_EVAL,
                           "k_nn": K_NN, "n_sample": N_SAMPLE,
                           "var_keep": VAR_KEEP, "seed": SEED},
            "note_eckart_young": "For any linear basis V (a fixed 2-D subspace), "
                                 "trace(V'C0 V) <= lambda1+lambda2 of C0: no linear "
                                 "combination of the indicators exceeds PC1+PC2 in "
                                 "explained variance.",
        },
        "models": {},
        "time_stability": {},
    }

    for key, label in MODELS:
        log(f"== {label} ==")
        P = P_all[key]
        stats = string_statistics(P)
        log(f"   support={stats['support']} strict_minima={stats['n_strict_minima']} "
            f"basins={stats['n_basins']} entropy={stats['entropy_bits']:.2f} bits")
        C0 = pj.indicator_covariance(P)
        r = bs.indicator_lagged_covariances(P, lags=LAGS, chunk_positions=1)
        mean = r["mean"]
        labels = descent_basin_labels(P)
        bases = candidate_bases(C0, r)
        bases["lda"] = _candidate_lda(P, labels, C0, mean)

        # spectrum: slow eigenvalues (TICA at lag 1) and implied timescales
        lam_slow, _ = bs.tica_axes(C0, r["C"][1])
        record = {
            "label": label,
            "statistics": stats,
            "slow_timescales": bs.implied_timescales(lam_slow[:8], 1).tolist(),
            "candidates": {},
        }
        for name, V in bases.items():
            coord = coords_of(P, V, mean)
            metrics = {
                "vamp2_tau4": vamp2_score(V, C0, r, TAU_EVAL),
                "vamp2_tau1": vamp2_score(V, C0, r, 1),
                "variance_share": variance_share(V, C0),
                **autocorr_and_timescales(V, C0, r),
                "basin_separation": basin_separation(P, coord, labels, mean),
                "trustworthiness": trustworthiness_continuity(P, coord),
                "coverage": coverage(P, coord),
                "outliers": outliers(P, coord),
                "annotation": annotate(V, C0, mean),
            }
            record["candidates"][name] = metrics
            log(f"   {name:12s} vamp2={metrics['vamp2_tau4']:.4f} "
                f"var={metrics['variance_share']:.3f} "
                f"basin_nn={metrics['basin_separation']['mass_same_basin_nn']:.3f} "
                f"iso_mass={metrics['coverage']['isolated_mass']:.3f} "
                f"tw={metrics['trustworthiness']['trustworthiness']:.3f}")
        out["models"][key] = record

    log("== time stability (trajectory, pooled) ==")
    out["time_stability"] = time_stability(["pca", "tica4", "tica4-reg", "lda"])
    for k, v in out["time_stability"].items():
        log(f"   {k}: max={v['max_deg']:.1f} deg, last-12 max={v['max_last12_deg']:.1f} deg")

    OUT.write_text(json.dumps(out, indent=1))
    log(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run()
