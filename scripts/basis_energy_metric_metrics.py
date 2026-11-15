"""Metrics + 'explained organising principle' diagnostics for the energy-metric basis,
the two controls (rank layout, Hamming layout) and the pooled lambda sweep.

Reads the cached layouts from data/derived/energy_metric_layout_<model>.npz and
data/derived/energy_metric_sweep_pooled.npz (written by scripts/basis_energy_metric.py),
and extends data/derived/basis_study.json with a top-level ``energy_metric`` key.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("scripts")))
import basis_smoothness as bsm                 # noqa: E402  (metric helpers)
import langfit.maxent as me                     # noqa: E402
from langfit import basis as bs                  # noqa: E402

DATA = Path("data/derived")
CACHE_NPZ = DATA / "landscape_artifact.npz"
OUT = DATA / "basis_study.json"
L4, A = 4, me.A
G = 32


def load_model(key: str):
    npz = np.load(CACHE_NPZ)
    V = npz["V_traj"][-1] if key == "pooled" else npz[f"V_{key}"]
    m = me.MaxEnt(L=L4, cliques=me.cliques(2), V=[v.copy() for v in V])
    npz.close()
    P = m.prob()
    lp = m.log_prob().ravel() / np.log(2)
    return P, lp


def evaluate_coords(P: np.ndarray, lp: np.ndarray, coords: np.ndarray, G: int = G) -> dict:
    """The same robust metric set as basis_smoothness.evaluate, for arbitrary coords."""
    edges = bsm.pj.grid_edges(coords, G)
    bins = bsm.pj.bin_index(coords, edges)
    E, count = bsm.pj.bin_energy(lp, bins, G * G)
    occ = np.isfinite(E)
    Pcell = np.bincount(bins, weights=np.exp2(lp), minlength=G * G)
    adj4 = bsm._grid_adjacency4((G, G))
    adj8 = bsm.pj.grid_adjacency((G, G))
    return {
        "roughness": bsm._roughness(E, Pcell, occ, adj4),
        "moran_i": bsm._moran_i(E, occ, adj4),
        "homogeneity": bsm._homogeneity(P, lp, bins, G),
        "wells": bsm._wells(E, count, occ, adj8),
        "continuity_fraction": bsm._continuity(P, bins),
        "coverage": bsm._coverage(P, coords, bins, G),
    }


def _linear_r2(y: np.ndarray, P: np.ndarray, L: int, A: int) -> float:
    """P-weighted R^2 of regressing the coordinate y on the 104 one-hot indicators:
    how linear the layout axis is in word features."""
    y = np.asarray(y, np.float64).ravel()
    w = P.ravel()
    C0 = bs._indicator_covariance(P)
    mean = np.concatenate([P.sum(axis=tuple(k for k in range(L) if k != j))
                           for j in range(L)])
    my = float((w * y).sum())
    var = float((w * (y - my) ** 2).sum())
    Y = y.reshape(P.shape)
    cross = np.zeros(L * A)
    for j in range(L):
        Py = P * Y
        cross[j * A:(j + 1) * A] = Py.sum(axis=tuple(k for k in range(L) if k != j)) \
            - mean[j * A:(j + 1) * A] * my
    lam, U = np.linalg.eigh(C0)
    keep = lam > 1e-10 * max(lam[-1], 1e-30)
    z = U[:, keep].T @ cross
    r2 = float(cross @ (U[:, keep] @ (z / lam[keep])) / var) if var > 0 else float("nan")
    return r2


def diagnostics(coords: np.ndarray, P: np.ndarray, E: np.ndarray) -> dict:
    from scipy.stats import spearmanr
    x, y = coords[0].astype(np.float64), coords[1].astype(np.float64)
    E = np.asarray(E, np.float64).ravel()
    sup = P.ravel() > 0                              # the landscape's support only
    def sp(a):
        return float(spearmanr(a[sup], E[sup]).correlation)
    mean = np.concatenate([P.sum(axis=tuple(k for k in range(L4) if k != j))
                           for j in range(L4)])
    pc1 = bs.coords_linear(bs.compute_basis("pca", P=P)["U"], mean, L4, A)[0]
    return {
        "linear_r2_x": _linear_r2(x, P, L4, A),
        "linear_r2_y": _linear_r2(y, P, L4, A),
        "spearman_x_E": sp(x),
        "spearman_y_E": sp(y),
        "pearson_r2_x_E": float(np.corrcoef(x[sup], E[sup])[0, 1] ** 2),
        "pearson_r2_y_E": float(np.corrcoef(y[sup], E[sup])[0, 1] ** 2),
        "spearman_x_pc1": float(spearmanr(x[sup], pc1[sup]).correlation),
        "spearman_y_pc1": float(spearmanr(y[sup], pc1[sup]).correlation),
    }


def run() -> None:
    result = {}
    for key in ("austen", "twitter2023", "bluesky2024", "pooled"):
        f = DATA / f"energy_metric_layout_{key}.npz"
        if not f.exists():
            print(f"[{key}] cache missing, skip")
            continue
        z = np.load(f)
        P, lp = load_model(key)
        coords = z["coords"]
        row = {
            "lam": float(z["lam"]),
            "K": int(z["K"]),
            "stress": float(z["stress"]),
            "metrics": evaluate_coords(P, lp, coords),
            "controls": {
                "rank": evaluate_coords(P, lp, z["rank"]),
                "hamming": evaluate_coords(P, lp, z["hamming"]),
            },
            "diagnostics": diagnostics(coords, P, -lp),
        }
        result[key] = row
        r = row["metrics"]["roughness"]
        print(f"[{key}] lam={z['lam']:.2f} |dE|/sd={r['mean_abs_dE_sd_norm']:.3f} "
              f"core={r['core_mean_abs_dE_sd_norm']:.3f} R2={row['metrics']['homogeneity']['anova_r2']:.3f} "
              f"occ={row['metrics']['coverage']['occupied_fraction']:.2f} "
              f"linR2x={row['diagnostics']['linear_r2_x']:.2f} "
              f"spear_xE={row['diagnostics']['spearman_x_E']:.2f}")
        z.close()

    sweep = DATA / "energy_metric_sweep_pooled.npz"
    if sweep.exists():
        z = np.load(sweep, allow_pickle=True)
        P, lp = load_model("pooled")
        result["sweep"] = {
            "lam": float(z["lam"]),
            "stress_vs_K": {k: float(v) for k, v in z["stress_vs_K"].item().items()},
        }
        for name in ("lam025", "lam100", "lam400"):
            result["sweep"][name] = evaluate_coords(P, lp, z[name])
            result["sweep"][name + "_diagnostics"] = diagnostics(z[name], P, -lp)
        z.close()

    js = json.loads(OUT.read_text()) if OUT.exists() else {}
    js["energy_metric"] = result
    OUT.write_text(json.dumps(js, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    run()
