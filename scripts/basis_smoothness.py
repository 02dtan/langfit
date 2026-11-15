"""Smoothness / continuity / uniformity metrics for the selectable 2-D bases.

For each of the four S&B models and each basis in ``basis.BASES``, at grids 16/32/48,
bins every string and scores the RAW (uncapped) E_bin = -log2(P(bin)/|bin|) surface:

  * roughness      — mean |dE| over occupied 4-neighbour pairs (unweighted and
                     mass-weighted), discrete-Laplacian RMS, total variation,
                     all normalised by the E_bin range;
  * autocorrelation— Moran's I of E_bin over occupied cells (4-neighbour weights);
  * homogeneity    — P-weighted ANOVA R^2 of E(s) given cell, mean within-cell std;
  * wells          — number of grid-local minima and of persistent (>= 1 bit) wells,
                     and the spurious-well fraction (wells whose basin holds <= k
                     strings);
  * continuity     — mass-weighted fraction of single-substitution neighbour pairs
                     landing in the same or an 8-adjacent cell;
  * coverage       — occupied-cell fraction, P-mass in the top 99% of cells, outlier
                     radius, coefficient of variation of |bin|;
  * time stability — principal angle of the basis fit one year before the end vs the
                     final model (pooled trajectory).

Output: extends ``data/derived/basis_study.json`` with a top-level ``smoothness`` key.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

import langfit.maxent as me
import langfit.projection as pj
from langfit import basis as bs

DATA = Path("data/derived")
CACHE_NPZ = DATA / "landscape_artifact.npz"
OUT = DATA / "basis_study.json"

L4, A = 4, me.A
GRIDS = [16, 32, 48]
BASIS_NAMES = ["pca", "coupling_spectral", "energy_aligned", "tempered_pca", "lda"]
MODELS = [("austen", "(a) Austen"), ("twitter2023", "(b) Twitter 2023"),
          ("bluesky2024", "(c) Bluesky 2024"), ("pooled", "(d) Twitter pooled")]
K_SPURIOUS = 8                  # a persistent well backed by <= K strings is "spurious"
N_SAMPLE = 8000                 # P-weighted sample for the continuity fraction
MASS_FRAC = 0.99                # "top 99% of cells" coverage mass threshold
TEMPER_ALPHA = 0.75             # default tempered-PCA alpha (chosen by connected-then-smoother)


def log(msg: str) -> None:
    print(msg, flush=True)


def _load_model(key: str) -> me.MaxEnt:
    npz = np.load(CACHE_NPZ)
    V = npz["V_traj"][-1] if key == "pooled" else npz[f"V_{key}"]
    m = me.MaxEnt(L=L4, cliques=me.cliques(2), V=[v.copy() for v in V])
    npz.close()
    return m


def _grid_adjacency4(shape: tuple[int, int]) -> list[np.ndarray]:
    out = []
    for k in range(shape[0] * shape[1]):
        r, c = divmod(k, shape[1])
        out.append(np.array([(r + dr) * shape[1] + c + dc
                             for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                             if 0 <= r + dr < shape[0] and 0 <= c + dc < shape[1]],
                            dtype=np.int64))
    return out


def _moran_i(E: np.ndarray, occ: np.ndarray, adj4: list[np.ndarray]) -> float:
    """Moran's I of E over occupied cells with 4-neighbour weights."""
    idx = np.flatnonzero(occ)
    n = len(idx)
    if n < 2:
        return float("nan")
    pos = {int(k): i for i, k in enumerate(idx)}
    z = E[idx] - E[idx].mean()
    W = np.zeros((n, n))
    for k in idx:
        i = pos[int(k)]
        for j in adj4[k]:
            if occ[j] and j > k:
                W[i, pos[int(j)]] = 1.0
                W[pos[int(j)], i] = 1.0
    s0 = W.sum()
    if s0 == 0:
        return float("nan")
    den = (z ** 2).sum()
    if den == 0:
        return float("nan")
    return float(n / s0 * (W * np.outer(z, z)).sum() / den)


def _paired(E: np.ndarray, Pcell: np.ndarray, occ: np.ndarray,
            adj4: list[np.ndarray], cells: set | None) -> dict:
    """mean |dE| and Laplacian RMS over occupied 4-neighbour pairs, plus the
    P-mass-weighted SD of E_bin and its 5-95 inter-percentile range.  If ``cells`` is
    given, only pairs with both ends in ``cells`` are used (common-support core)."""
    dE, w, lap = [], [], []
    for k in np.flatnonzero(occ):
        nb = [j for j in adj4[k] if occ[j] and (cells is None or j in cells)]
        if cells is not None and k not in cells:
            nb = []
        if nb:
            lap.append(E[k] - np.mean(E[nb]))
            for j in nb:
                if j > k:
                    dE.append(abs(E[k] - E[j]))
                    w.append(Pcell[k] + Pcell[j])
    dE = np.array(dE)
    w = np.array(w)
    sc = bs.mass_weighted_scale(E, Pcell, occ)
    sd = sc["sd"]
    iqr = sc["iqr595"]
    return {
        "mean_abs_dE": float(dE.mean()) if len(dE) else float("nan"),
        "laplacian_rms": float(np.sqrt(np.mean(np.array(lap) ** 2))) if lap else float("nan"),
        "sd": sd,
        "iqr595": iqr,
        "mean_abs_dE_sd_norm": float(dE.mean() / sd) if sd > 0 and len(dE) else float("nan"),
        "laplacian_rms_sd_norm": float(np.sqrt(np.mean(np.array(lap) ** 2)) / sd)
        if sd > 0 and lap else float("nan"),
        "mean_abs_dE_iqr_norm": float(dE.mean() / iqr) if iqr > 0 and len(dE) else float("nan"),
    }


def _roughness(E: np.ndarray, Pcell: np.ndarray, occ: np.ndarray,
               adj4: list[np.ndarray]) -> dict:
    full = _paired(E, Pcell, occ, adj4, None)
    # common-support core: cells holding the top 90% of P-mass (so a basis cannot win by
    # occupying fewer cells / stretching the range with sparse outlier cells)
    order = np.argsort(-Pcell)
    cdf = np.cumsum(Pcell[order])
    top90 = int(np.searchsorted(cdf, 0.9 * Pcell.sum()) + 1)
    core = set(order[:top90].tolist())
    core = _paired(E, Pcell, occ, adj4, core)
    rng = float(np.nanmax(E[occ]) - np.nanmin(E[occ]))
    dE = np.array([abs(E[k] - E[j]) for k in np.flatnonzero(occ)
                   for j in adj4[k] if occ[j] and j > k])
    return {
        "mean_abs_dE": full["mean_abs_dE"],
        "mean_abs_dE_mass": None,          # kept for backward-compat; see _paired
        "laplacian_rms": full["laplacian_rms"],
        "sd": full["sd"],
        "iqr595": full["iqr595"],
        "mean_abs_dE_sd_norm": full["mean_abs_dE_sd_norm"],
        "laplacian_rms_sd_norm": full["laplacian_rms_sd_norm"],
        "mean_abs_dE_iqr_norm": full["mean_abs_dE_iqr_norm"],
        "range": rng,
        "mean_abs_dE_range_norm": float(full["mean_abs_dE"] / rng) if rng > 0 else float("nan"),
        "laplacian_rms_range_norm": float(full["laplacian_rms"] / rng) if rng > 0 else float("nan"),
        "core_mean_abs_dE_sd_norm": core["mean_abs_dE_sd_norm"],
        "core_laplacian_rms_sd_norm": core["laplacian_rms_sd_norm"],
        "core_n_cells": top90,
    }


def _homogeneity(P: np.ndarray, lp: np.ndarray, bins: np.ndarray,
                 G: int) -> dict:
    Es = -lp                                        # bits; -inf -> +inf on forbidden
    w = P.ravel()
    sup = np.isfinite(lp)
    Efin = np.where(sup, Es, 0.0)
    mean_E = float((w * Efin).sum())
    ss_total = float((w * (Efin - mean_E) ** 2).sum())
    cell_mass = np.bincount(bins[sup], weights=w[sup], minlength=G * G)
    cell_sumE = np.bincount(bins[sup], weights=(w * Efin)[sup], minlength=G * G)
    occ = cell_mass > 0
    mean_cell = np.zeros(G * G)
    mean_cell[occ] = cell_sumE[occ] / cell_mass[occ]
    ss_within = float((w * (Efin - mean_cell[bins]) ** 2).sum())
    r2 = 1.0 - ss_within / ss_total if ss_total > 0 else float("nan")
    # mean within-cell std, mass-weighted over occupied cells
    ws = []
    mass = []
    for k in np.flatnonzero(occ):
        m = sup & (bins == k)
        if m.sum() >= 2:
            ws.append(np.sqrt((w[m] * (Efin[m] - mean_cell[k]) ** 2).sum() / w[m].sum()))
            mass.append(cell_mass[k])
    ws, mass = np.array(ws), np.array(mass)
    return {
        "anova_r2": r2,
        "mean_within_cell_std": float((ws * mass).sum() / mass.sum()) if mass.sum() > 0
        else float("nan"),
    }


def _wells(E: np.ndarray, count: np.ndarray, occ: np.ndarray,
           adj8: list[np.ndarray]) -> dict:
    n_local = 0
    for k in np.flatnonzero(occ):
        nb = [j for j in adj8[k] if occ[j]]
        if nb and all(E[k] <= E[j] for j in nb) and any(E[k] < E[j] for j in nb):
            n_local += 1
    allw = pj.persistence(E, adj8)
    wells = {t: [w for w in allw if w["persistence"] >= t]
             for t in (1.0, 2.0, 3.0)}
    lab = pj.significant_basins(E, allw, 1.0, adj8)
    spurious = 0
    for w in wells[1.0]:
        n_str = int(count[lab == w["cell"]].sum())
        if n_str <= K_SPURIOUS:
            spurious += 1
    return {
        "n_local_minima": n_local,
        "n_persistent_wells_1bit": len(wells[1.0]),
        "n_persistent_wells_2bit": len(wells[2.0]),
        "n_persistent_wells_3bit": len(wells[3.0]),
        "n_wells_all": int(sum(w["persistence"] > 0 for w in allw)),
        "spurious_well_fraction": float(spurious / len(wells[1.0])) if wells[1.0]
        else float("nan"),
    }


def _continuity(P: np.ndarray, bins: np.ndarray, n: int = N_SAMPLE, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    w = P.ravel()
    sup = np.flatnonzero(w > 0)
    idx = rng.choice(sup, size=min(n, len(sup)), replace=False, p=w[sup] / w[sup].sum())
    digits = np.stack(np.unravel_index(idx, (A,) * L4)).T          # (m, L)
    strides = A ** np.arange(L4 - 1, -1, -1)
    base = (digits * strides).sum(axis=1).astype(np.int64)
    G = int(np.sqrt(len(bins)))
    cell = bins[idx].reshape(-1)                                    # (m,)
    frac = 0.0
    m = len(idx)
    for p in range(L4):
        off = A ** (L4 - 1 - p)
        for a in range(A):
            tgt = base + (a - digits[:, p]) * off
            ok = (w[tgt] > 0) & (tgt != idx)                         # in support, not self
            ct = bins[tgt[ok]]                                       # neighbour cell
            cs = cell[ok]                                            # source cell
            dr = np.abs((ct // G) - (cs // G))
            dc = np.abs((ct % G) - (cs % G))
            frac += float((np.maximum(dr, dc) <= 1).sum())           # same or 8-adjacent
    denom = m * L4 * (A - 1)
    return float(frac / denom)


def _coverage(P: np.ndarray, coords: np.ndarray, bins: np.ndarray, G: int) -> dict:
    sup = P.ravel() > 0
    w = P.ravel()
    occ = np.bincount(bins[sup], minlength=G * G) > 0
    mass = np.bincount(bins[sup], weights=w[sup], minlength=G * G)
    order = np.argsort(-mass)
    cdf = np.cumsum(mass[order])
    top99 = int(np.searchsorted(cdf, MASS_FRAC) + 1)
    count = np.bincount(bins[sup], minlength=G * G)
    c_occ = count[occ]
    # outlier radius (max distance from P-weighted centroid, in RMS units)
    xy = coords[:, sup].T
    mu = (w[sup][:, None] * xy).sum(0) / w[sup].sum()
    d = np.linalg.norm(xy - mu, axis=1)
    rms = float(np.sqrt((w[sup] * d ** 2).sum() / w[sup].sum()))
    # collapse guards
    pcell = mass[mass > 0]
    pm = pcell / pcell.sum()
    ent = -float((pm * np.log(pm)).sum())
    # P-weighted variance of the two coordinate axes -> ratio (1 = isotropic)
    vx = float((w[sup] * (xy[:, 0] - mu[0]) ** 2).sum() / w[sup].sum())
    vy = float((w[sup] * (xy[:, 1] - mu[1]) ** 2).sum() / w[sup].sum())
    var_ratio = float(max(vx, vy) / min(vx, vy)) if min(vx, vy) > 0 else float("inf")
    # number of 4- and 8-connected components of the OCCUPIED cells (edges only between
    # occupied cells, so the count reflects the actual connected structure)
    n4, big4 = bs.occupied_components(occ, G, 4)
    n8, big8 = bs.occupied_components(occ, G, 8)
    return {
        "occupied_fraction": float(occ.sum() / (G * G)),
        "mass_in_top99_cells": float(mass[order[:top99]].sum()),
        "n_cells_for_99pct_mass": int(top99),
        "occupancy_cv": float(c_occ.std() / c_occ.mean()) if len(c_occ) else float("nan"),
        "outlier_max_rms": float((d / rms).max()) if rms > 0 else float("nan"),
        "effective_occupied_cells": float(np.exp(ent)),
        "axis_variance_ratio": var_ratio,
        "n_occupied_components": n4,
        "n_components_4": n4,
        "n_components_8": n8,
        "largest_component_4": big4,
        "largest_component_8": big8,
    }


def evaluate(model: me.MaxEnt, name: str, G: int, alpha: float = None) -> dict:
    P = model.prob()
    lp = model.log_prob().ravel() / np.log(2)
    b = bs.compute_basis(name, model=model, alpha=alpha) if alpha is not None \
        else bs.compute_basis(name, model=model)
    U, mean = b["U"], b["mean"]
    coords = bs.coords_linear(U, mean, L4, A)
    edges = pj.grid_edges(coords, G)
    bins = pj.bin_index(coords, edges)
    E, count = pj.bin_energy(lp, bins, G * G)
    occ = np.isfinite(E)
    Pcell = np.bincount(bins, weights=np.exp2(lp), minlength=G * G)
    adj4 = _grid_adjacency4((G, G))
    adj8 = pj.grid_adjacency((G, G))
    return {
        "roughness": _roughness(E, Pcell, occ, adj4),
        "moran_i": _moran_i(E, occ, adj4),
        "homogeneity": _homogeneity(P, lp, bins, G),
        "wells": _wells(E, count, occ, adj8),
        "continuity_fraction": _continuity(P, bins),
        "coverage": _coverage(P, coords, bins, G),
    }


def _subspace_angle(Va: np.ndarray, Vb: np.ndarray) -> float:
    Qa, _ = np.linalg.qr(Va)
    Qb, _ = np.linalg.qr(Vb)
    s = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0.0, 1.0)
    return float(np.rad2deg(np.arccos(s.min())))


def time_stability() -> dict:
    npz = np.load(CACHE_NPZ)
    V = npz["V_traj"]
    npz.close()
    early = me.MaxEnt(L=L4, cliques=me.cliques(2), V=[v.copy() for v in V[191]])
    final = me.MaxEnt(L=L4, cliques=me.cliques(2), V=[v.copy() for v in V[-1]])
    out = {}
    for name in BASIS_NAMES:
        b1 = bs.compute_basis(name, model=early)
        b2 = bs.compute_basis(name, model=final)
        out[name] = _subspace_angle(b1["U"], b2["U"])
    return out


def _tempered_alpha_scan(model: me.MaxEnt, G: int = 32) -> dict:
    """Pick the tempered-PCA alpha in {0.25, 0.5, 0.75} by: connected occupied cells
    (n_components == 1 preferred) then smoothness (lowest normalised |dE|)."""
    scan = {}
    for a in (0.25, 0.5, 0.75):
        m = evaluate(model, "tempered_pca", G, alpha=a)
        scan[str(a)] = {
            "roughness_sd_norm": m["roughness"]["mean_abs_dE_sd_norm"],
            "n_components": m["coverage"]["n_occupied_components"],
            "occupied_fraction": m["coverage"]["occupied_fraction"],
            "continuity": m["continuity_fraction"],
        }
    # criterion: prefer connected, then smoother (robust SD-normalised |dE|)
    def key(a):
        s = scan[str(a)]
        return (s["n_components"], s["roughness_sd_norm"])
    best = min((0.25, 0.5, 0.75), key=key)
    return {"scan": scan, "chosen_alpha": best}


def run() -> None:
    t0 = time.time()
    result = {}
    for key, label in MODELS:
        log(f"== {label} ==")
        model = _load_model(key)
        result[key] = {"label": label, "bases": {}}
        for name in BASIS_NAMES:
            row = {}
            for G in GRIDS:
                row[str(G)] = evaluate(model, name, G)
            result[key]["bases"][name] = row
            m32 = row["32"]
            r = m32["roughness"]
            log(f"  {name:18s} |dE|/sd={r['mean_abs_dE_sd_norm']:.3f} "
                f"lap/sd={r['laplacian_rms_sd_norm']:.3f} "
                f"(range-norm {r['mean_abs_dE_range_norm']:.3f}) "
                f"core|dE|/sd={r['core_mean_abs_dE_sd_norm']:.3f} "
                f"occ={m32['coverage']['occupied_fraction']:.2f} "
                f"eff={m32['coverage']['effective_occupied_cells']:.0f} "
                f"moran={m32['moran_i']:.2f} R2={m32['homogeneity']['anova_r2']:.3f} "
                f"cont={m32['continuity_fraction']:.3f} "
                f"varRatio={m32['coverage']['axis_variance_ratio']:.1f}")
        if key == "pooled":
            result[key]["tempered_alpha"] = _tempered_alpha_scan(model)
            log(f"  tempered alpha scan: {result[key]['tempered_alpha']}")
    log("== time stability (pooled, one year earlier) ==")
    stab = time_stability()
    for k, v in stab.items():
        log(f"  {k:18s} {v:.1f} deg")

    # extend the existing study json
    js = json.loads(OUT.read_text()) if OUT.exists() else {}
    js["smoothness"] = {
        "parameters": {"grids": GRIDS, "bases": BASIS_NAMES, "tempered_alpha": TEMPER_ALPHA,
                       "k_spurious": K_SPURIOUS, "n_sample": N_SAMPLE,
                       "mass_frac": MASS_FRAC},
        "models": result,
        "time_stability_deg": stab,
    }
    OUT.write_text(json.dumps(js, indent=1))
    log(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run()
