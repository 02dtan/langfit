"""Compute the energy-metric (geodesic-MDS) layouts for the four S&B models, the two
controls (rank layout, Hamming layout) and the pooled lambda sweep, caching to
data/derived/energy_metric_layout_<model>.npz (float32).  Resumable: skips models whose
cache already exists.  Runs sequentially; the finite-lambda layouts dominate the time
(~K Dijkstra runs each, ~3 s/Dijkstra on the 47.5M-edge Hamming graph).

Usage:  python scripts/basis_energy_metric.py [K] [--pooled-sweep]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import langfit.maxent as me
from langfit import basis as bs

DATA = Path("data/derived")
CACHE_NPZ = DATA / "landscape_artifact.npz"
L4, A = 4, me.A

MODELS = [("austen", "V_austen"), ("twitter2023", "V_twitter2023"),
          ("bluesky2024", "V_bluesky2024"), ("pooled", "V_traj")]


def log(msg: str) -> None:
    print(msg, flush=True)


def load_model(key: str) -> tuple[me.MaxEnt, np.ndarray, np.ndarray]:
    npz = np.load(CACHE_NPZ)
    V = npz["V_traj"][-1] if key == "pooled" else npz[f"V_{key}"]
    m = me.MaxEnt(L=L4, cliques=me.cliques(2), V=[v.copy() for v in V])
    npz.close()
    P = m.prob()
    with np.errstate(divide="ignore"):
        E = -np.log(P) / np.log(2)
    E = bs._cap_infinite(E)
    return m, P.astype(np.float32), E.astype(np.float32)


def layout(P: np.ndarray, E: np.ndarray, lam: float, K: int) -> tuple[np.ndarray, np.ndarray]:
    """energy-metric layout, plus the (K, N) geodesic matrix D and the landmarks."""
    csr = bs._ham_graph_csr(E, L4, A, lam)
    D, landmarks = bs._p_farthest_landmarks(csr, P.ravel(), K)
    coords = bs._landmark_mds(D)
    coords = bs._orient_layout(coords, E.ravel(), P.ravel())
    return coords.T.astype(np.float32), D, landmarks


def kruskal_stress(D: np.ndarray, coords: np.ndarray, landmarks: np.ndarray,
                   Kp: int) -> float:
    d = D[:Kp, landmarks[:Kp]]
    c = coords[:, landmarks[:Kp]].T
    dij = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    return float(np.sqrt(((d - dij) ** 2).sum() / (d ** 2).sum()))


def one_model(key: str, K: int) -> None:
    out = DATA / f"energy_metric_layout_{key}.npz"
    if out.exists():
        log(f"[{key}] cache exists, skip")
        return
    t0 = time.time()
    m, P, E = load_model(key)
    lam = bs._flux_median_slope(P, L4, A)
    log(f"[{key}] lam={lam:.3f} bits/unit; computing layout K={K} ...")
    coords, D, landmarks = layout(P, E, lam, K)
    rank = bs.energy_rank_layout(E, L4, A)
    hamm = bs.hamming_layout(P.ravel(), L4, A, K=K)
    np.savez_compressed(out, coords=coords, rank=rank, hamming=hamm, lam=lam, K=K,
                        E_cap=float(E.max()), stress=float(kruskal_stress(D, coords, landmarks, K)))
    log(f"[{key}] done in {time.time()-t0:.0f}s; stress={kruskal_stress(D, coords, landmarks, K):.4f} "
        f"-> {out} ({out.stat().st_size/1e6:.1f} MB)")


def pooled_sweep(K: int) -> None:
    out = DATA / "energy_metric_sweep_pooled.npz"
    if out.exists():
        log("[pooled-sweep] cache exists, skip")
        return
    m, P, E = load_model("pooled")
    lam = bs._flux_median_slope(P, L4, A)
    result = {"lam": lam}
    # stress vs K
    _, D, landmarks = layout(P, E, lam, K)
    result["stress_vs_K"] = {}
    for Kp in (50, 100, 200):
        c = bs._landmark_mds(D[:Kp])
        c = bs._orient_layout(c, E.ravel(), P.ravel()).T.astype(np.float32)
        result["stress_vs_K"][str(Kp)] = float(kruskal_stress(D, c, landmarks, Kp))
    # lambda sweep (0.25x, 1x, 4x)
    for name, mult in (("lam025", 0.25), ("lam100", 1.0), ("lam400", 4.0)):
        t0 = time.time()
        c, _, _ = layout(P, E, lam * mult, K)
        result[name] = c
        log(f"  pooled lam*{mult} done in {time.time()-t0:.0f}s")
    np.savez_compressed(out, **result)
    log(f"[pooled-sweep] wrote {out} ({out.stat().st_size/1e6:.1f} MB); stress_vs_K={result['stress_vs_K']}")


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    do_sweep = "--pooled-sweep" in sys.argv
    for key, _ in MODELS:
        one_model(key, K)
    if do_sweep:
        pooled_sweep(K)
    log("ALL DONE")
