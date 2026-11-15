"""Build ``index.html`` (repo root, the GitHub Pages entry point): the energy landscape of the pairwise
maximum-entropy model of four-letter words (Stephens & Bialek, arXiv:0801.0253),
projected onto the top two principal axes of the model's indicator covariance.

Stages (each cached; rerunning a stage is needed only if its inputs change):

    prior       sigma^2 for the pooled-Twitter MAP model, by maxent.select_prior
    fits        tab-1 models (a) Austen S&B replication, (b) Twitter 2023 and
                (c) Bluesky 2024 exact fits on the rep-0 matched-N subsample
    trajectory  MAP fits to the cumulative monthly Twitter counts (tab 2); the final
                frame is model (d)
    build       projections, bin energies, wells, HTML (cheap; uses only the caches)

    .venv/bin/python scripts/build_landscape_artifact.py [stage ...]   (default: all)

Caches: ``data/derived/landscape_artifact.npz`` (coupling tensors) and
``data/derived/landscape_artifact.json`` (prior selection, fit diagnostics).
"""

from __future__ import annotations

import base64
import os
import json
import re
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# Seed the global RNG for reproducibility of any randomised step inside scipy's
# trust-region Krylov solver. NOTE: this does NOT pin the Austen strict-minima count —
# that count flips 148/149 on a sub-fit-tolerance tie ("lamb" vs "limb" differ by a few
# 1e-4 bits, below the fit tolerance); see test_austen_minima_tolerance_aware.
np.random.seed(0)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from langfit import basis as bas, maxent as me, projection as pj, social  # noqa: E402
import maxent_study as ms  # noqa: E402

CACHE_NPZ = ms.corpus.DATA / "derived" / "landscape_artifact.npz"
CACHE_JSON = ms.corpus.DATA / "derived" / "landscape_artifact.json"
OUT_HTML = ROOT / "index.html"
MONTHS = "twitter_month_L4_existing16.pkl.gz"
# Compute budget: at most 3 worker processes (cores are shared with other jobs).
WORKERS = min(3, int(os.environ.get("LANDSCAPE_WORKERS", 3)))
PRIOR_FOLDS = ms.INNER           # same K as the study's inner cross-validation
S2_TOL = 1e-4                    # bits; reproduction check against maxent_study.json
PLOTLY = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"

# Display parameters (they change the picture, not the model):
GRID = 32                        # bins per axis of the projection grid
PERSIST_BITS = 1.0               # label wells with persistence >= 1 bit (factor-2 in mean P)
TOP_STRING_MINIMA = 15           # string-level minima overlaid in tab 1
BASIN_WORDS = 3                  # most probable words listed per labelled well
CENTIBITS = 100                  # quantisation of tab-2 surfaces: 0.01 bit
SUB = 3                          # finer display grid: sub x sub subcells per bin (tab 1)
SUB_TRAJ = 2                     # finer display grid for the 203-frame trajectory
CONTOURS = 9                     # contour lines on the 2D heatmap (n+1 intervals)
BASIS_SMOOTH_STRIDE = 8          # non-default trajectory bases: smooth every k-th frame


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #

def load_json() -> dict:
    return json.loads(CACHE_JSON.read_text()) if CACHE_JSON.exists() else {}


def save_json(d: dict) -> None:
    CACHE_JSON.write_text(json.dumps(d, indent=1))


def load_npz() -> dict:
    if not CACHE_NPZ.exists():
        return {}
    with np.load(CACHE_NPZ) as z:
        return {k: z[k] for k in z.files}


def save_npz(d: dict) -> None:
    tmp = CACHE_NPZ.with_suffix(".tmp.npz")
    np.savez(tmp, **d)          # uncompressed: float64/int16 data compresses little,
    tmp.replace(CACHE_NPZ)      # but gzip on the 1.6 MB smooth grid is slow and stalls


def pack_V(m: me.MaxEnt) -> np.ndarray:
    return np.stack(m.V)                                    # (6, 26, 26)


def pack_V_b64(V: np.ndarray) -> str:
    """Couplings as float32 base64 for the client-side LandscapeEngine."""
    return base64.b64encode(np.ascontiguousarray(V, "<f4").tobytes()).decode()


def engine_data(basis: Basis, name: str) -> dict:
    """Embed a basis's axes (U1, U2, mean) and fixed extent for the engine. The extent
    is G-independent, so the resolution slider never re-derives it."""
    U1 = basis.U[:, 0].astype("<f4")
    U2 = basis.U[:, 1].astype("<f4")
    mean = (basis.U.T @ basis.mean).astype("<f4")
    return {"name": name, "U1": base64.b64encode(U1.tobytes()).decode(),
            "U2": base64.b64encode(U2.tobytes()).decode(),
            "mean": [float(mean[0]), float(mean[1])],
            "extent": {"xmin": float(basis.coords[0].min()), "xmax": float(basis.coords[0].max()),
                       "ymin": float(basis.coords[1].min()), "ymax": float(basis.coords[1].max())}}


def energy_metric_coords(key: str) -> dict | None:
    """Cached energy_metric layout (coords basis) as deflated int16 base64 + extent."""
    p = ms.corpus.DATA / "derived" / f"energy_metric_layout_{key}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        coords = z["coords"].astype("<f4")              # (2, 456976)
    xmin, xmax = float(coords[0].min()), float(coords[0].max())
    ymin, ymax = float(coords[1].min()), float(coords[1].max())
    q = np.clip(np.round((coords - np.array([[xmin], [ymin]], "<f4"))
                         / np.array([[xmax - xmin], [ymax - ymin]], "<f4") * 65535.0)
                - 32768.0, -32768, 32767).astype("<i2")
    return {"coords_b64": base64.b64encode(zlib.compress(q.tobytes(), 9)).decode(),
            "deflated": True,
            "extent": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}}


def unpack_V(V: np.ndarray, lam=None) -> me.MaxEnt:
    return me.MaxEnt(L=4, cliques=me.cliques(2), V=[v.copy() for v in V], lam=lam)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def monthly_counters() -> tuple[list[str], list[Counter]]:
    P = social.load_periods(MONTHS)["periods"]
    months = sorted(P)
    return months, [Counter(P[k]["counts"].get(4, {})) for k in months]


def pooled_twitter() -> me.WordCounts:
    tot = Counter()
    for c in monthly_counters()[1]:
        tot.update(c)
    return me.WordCounts.from_counter(tot)


# --------------------------------------------------------------------------- #
# stage: prior
# --------------------------------------------------------------------------- #

def _job_cv_fold(args):
    states, counts, grid, k = args
    return me.cv_fold(me.WordCounts(states, counts, 4), 2, grid, k, PRIOR_FOLDS, ms.SEED)


def select_prior_parallel(wc: me.WordCounts) -> dict:
    """``maxent.select_prior(wc, 2, PRIOR_FOLDS, SEED)`` with its independent folds run
    in parallel: the same ``cv_fold`` calls with the same arguments, the same grid
    extension rule, hence the same result."""
    def run(ex, grid):
        return list(ex.map(_job_cv_fold, [(wc.states, wc.counts, grid, k)
                                          for k in range(PRIOR_FOLDS)]))

    with ProcessPoolExecutor(WORKERS) as ex:
        per_fold = run(ex, list(me.SIGMA2_START))
        agg = me.aggregate_cv(per_fold)
        for _ in range(me.SIGMA2_MAX_EXTENSIONS):
            if agg["interior"]:
                break
            lo = agg["best_sigma2"] == agg["sigma2"][0]
            new = agg["sigma2"][0] / me.SIGMA2_STEP if lo else agg["sigma2"][-1] * me.SIGMA2_STEP
            log(f"  extending the sigma^2 grid to {new:.4g}")
            per_fold = [a + b for a, b in zip(per_fold, run(ex, [new]))]
            agg = me.aggregate_cv(per_fold)
    return agg


def stage_prior() -> None:
    wc = pooled_twitter()
    log(f"select_prior on pooled Twitter: {wc.n_tokens:,.0f} tokens, {wc.counts.size} types, "
        f"{PRIOR_FOLDS} folds, seed {ms.SEED}")
    t = time.time()
    sel = select_prior_parallel(wc)
    sel["seconds"] = time.time() - t
    sel["n_folds"], sel["seed"], sel["tokens"] = PRIOR_FOLDS, ms.SEED, wc.n_tokens
    log(f"  sigma^2 = {sel['best_sigma2']:.4g}  interior {sel['interior']}  grid {sel['sigma2']}")
    d = load_json()
    d["prior"] = sel
    save_json(d)


# --------------------------------------------------------------------------- #
# stage: fits (tab-1 exact models)
# --------------------------------------------------------------------------- #

def _job_exact(args):
    key, states, counts = args
    wc = me.WordCounts(states, counts, 4)
    m, d = me.fit(wc, 2, "exact")
    # Claim (iii): an exact fit reproduces the pairwise marginals, so its indicator
    # covariance equals the empirical one.
    c_diff = float(np.abs(pj.indicator_covariance(m.prob())
                          - pj.indicator_covariance(wc.empirical())).max())
    return key, pack_V(m), {**d, "tokens": wc.n_tokens, "types": int(wc.counts.size),
                            "S2": m.entropy_bits(), "max_abs_C_model_minus_C_empirical": c_diff}


def stage_fits() -> None:
    dmask = ms.dictionary_mask()
    cs = ms.load_corpora()
    N = int(sum(cs["Austen"]["counter"].values()))
    names = ["Austen", "Gutenberg", "Bluesky"] + sorted(
        k for k in cs if k.startswith("Twitter ") and sum(cs[k]["counter"].values()) >= N)
    seed_of = {k: ms.SEED + 1000 * i for i, k in enumerate(names)}
    dic = {me.unravel(i) for i in np.flatnonzero(dmask)}
    jobs = [("austen", *_wc_arrays(me.WordCounts.from_counter(
        Counter({w: n for w, n in cs["Austen"]["counter"].items() if w in dic}))))]
    for key, name in (("twitter2023", "Twitter 2023"), ("bluesky2024", "Bluesky")):
        full = me.WordCounts.from_counter(cs[name]["counter"])
        jobs.append((key, *_wc_arrays(ms.subsample(full.states, full.counts, N, seed_of[name]))))
    study = json.loads(ms.OUT.read_text())
    ref = {"austen": study["replication"]["S_2"],
           "twitter2023": next(r["S2"] for r in study["corpora"]
                               if r["name"] == "Twitter 2023" and r["rep"] == 0),
           "bluesky2024": next(r["S2"] for r in study["corpora"]
                               if r["name"] == "Bluesky" and r["rep"] == 0)}
    seeds = {"twitter2023": seed_of["Twitter 2023"], "bluesky2024": seed_of["Bluesky"]}
    fits, arrays = {}, {}
    with ProcessPoolExecutor(WORKERS) as ex:
        for key, V, diag in ex.map(_job_exact, jobs):
            diag["S2_study"] = ref[key]
            diag["S2_abs_diff"] = abs(diag["S2"] - ref[key])
            diag["S2_reproduced"] = diag["S2_abs_diff"] <= S2_TOL
            if key in seeds:
                diag["seed"] = seeds[key]
                diag["N"] = N
            log(f"  {key}: S2 {diag['S2']:.6f} (study {ref[key]:.6f}, |diff| "
                f"{diag['S2_abs_diff']:.2e})  converged {diag['converged']}")
            arrays[f"V_{key}"] = V
            arrays[f"states_{key}"], arrays[f"counts_{key}"] = next(
                (j[1], j[2]) for j in jobs if j[0] == key)
            fits[key] = diag
    save_npz({**load_npz(), **arrays})
    js = load_json()
    js["fits"] = fits
    save_json(js)


def _wc_arrays(wc: me.WordCounts):
    return wc.states, wc.counts


# --------------------------------------------------------------------------- #
# stage: trajectory (tab 2; final frame = model (d))
# --------------------------------------------------------------------------- #

def _checkpoint(first_frame: int, sigma2: float) -> Path:
    return CACHE_NPZ.parent / f"landscape_artifact_chain{first_frame}_s{sigma2:.6g}.pkl"


def _job_chain(args):
    """Warm-started MAP fits over a contiguous run of cumulative-count frames.

    Each finished frame is checkpointed, so an interrupted run resumes where it
    stopped (warm-starting from the last saved frame)."""
    import pickle
    states, rows, frames, sigma2 = args              # rows[k]: counts of frames[k]
    ckpt = _checkpoint(frames[0], sigma2)
    out = pickle.loads(ckpt.read_bytes()) if ckpt.exists() else []
    init = unpack_V(out[-1][1]) if out and out[-1][2]["tokens"] > 0 else None
    for f in frames[len(out):]:
        c = rows[f - frames[0]]
        n = float(c.sum())
        warm = init is not None
        if n == 0:                                   # no tokens: V = 0, the uniform model
            m = me.MaxEnt(L=4, cliques=me.cliques(2), V=[np.zeros((me.A, me.A))] * 6)
            d = {"converged": True, "iterations": 0, "newton_polish": 0,
                 "max_gradient_in_se": 0.0, "uniform": True}
        else:
            t = time.time()
            m, d = me.fit(me.WordCounts(states, c, 4), 2, "l2",
                          lam=me.prior_lambda(sigma2, n), counts=c, init=init)
            d["seconds"] = time.time() - t
            init = m
        d.update(frame=int(f), tokens=n, S2=m.entropy_bits(), warm=warm)
        out.append((int(f), pack_V(m), d))
        ckpt.write_bytes(pickle.dumps(out))
        print(f"  frame {f}: N {n:,.0f} S2 {d['S2']:.4f} conv {d['converged']} "
              f"it {d.get('iterations')} {d.get('seconds', 0):.0f}s", flush=True)
    return out


def stage_trajectory() -> None:
    js = load_json()
    sigma2 = float(js["prior"]["best_sigma2"])
    months, ctrs = monthly_counters()
    tot = Counter()
    for c in ctrs:
        tot.update(c)
    wc = me.WordCounts.from_counter(tot)
    index = {w: k for k, w in enumerate(wc.words)}
    cum = np.zeros((len(months) + 1, wc.counts.size))   # row 0: no tweets
    for t, c in enumerate(ctrs):
        cum[t + 1] = cum[t]
        for w, n in c.items():
            k = index.get(w)
            if k is not None:
                cum[t + 1, k] += n
    assert np.array_equal(cum[-1], wc.counts)
    nf = cum.shape[0]
    # Contiguous segments, one chain each: cold start at the segment's first frame,
    # warm starts after. Scheduling only (the optimum is unique): segments hold equal
    # shares of log(1 + N_t), a rough cost proxy, since high-N fits are the slow ones.
    w = np.cumsum(np.log1p(cum.sum(axis=1)))
    cuts = [int(np.searchsorted(w, w[-1] * k / WORKERS)) for k in range(1, WORKERS)]
    bounds = np.array([0, *cuts, nf])
    jobs = [(wc.states, cum[a:b], list(range(a, b)), sigma2) for a, b in zip(bounds[:-1], bounds[1:])]
    log(f"trajectory: {nf} frames, sigma^2 {sigma2:.4g}, chains {bounds.tolist()}")
    res = {}
    with ProcessPoolExecutor(WORKERS) as ex:
        for chunk in ex.map(_job_chain, jobs):
            for f, V, d in chunk:
                res[f] = (V, d)
    npz = load_npz()
    npz["V_traj"] = np.stack([res[f][0] for f in range(nf)])
    npz["traj_tokens"] = cum.sum(axis=1)
    save_npz(npz)
    js = load_json()
    js["trajectory"] = {"sigma2": sigma2, "months": ["(none)"] + months,
                        "chain_bounds": bounds.tolist(),
                        "diagnostics": [res[f][1] for f in range(nf)]}
    save_json(js)
    for a in bounds[:-1]:
        _checkpoint(int(a), sigma2).unlink(missing_ok=True)
    log(f"  all converged: {all(res[f][1]['converged'] for f in range(nf))}")


STAGES = {"prior": stage_prior, "fits": stage_fits, "trajectory": stage_trajectory}


def main(argv) -> None:
    for s in argv or ["prior", "fits", "trajectory", "bases", "build"]:
        log(f"stage {s}")
        if s == "build":
            build()
        elif s == "bases":
            stage_bases()
        else:
            STAGES[s]()


# --------------------------------------------------------------------------- #
# stage: build (projection, bin energies, wells, HTML)
# --------------------------------------------------------------------------- #

L4 = 4
LN2 = np.log(2.0)
UNIFORM_BITS = L4 * np.log2(me.A)          # log2 26^4: the flat plane of the uniform model
TOP_LOADINGS = 10
MODELS = (                                  # tab-1 selector: (key, label)
    ("austen", "(a) Austen, S&B protocol"),
    ("twitter2023", "(b) Twitter 2023, N-matched"),
    ("bluesky2024", "(c) Bluesky 2024, N-matched"),
    ("pooled", "(d) Twitter pooled 2006-03..2023-04, MAP"),
)


class Basis:
    """A 2-D projection (unit loadings U, indicator mean) and the fixed grid built from
    it. ``lam`` is the full PCA spectrum when known (None for the non-PCA bases)."""

    def __init__(self, U: np.ndarray, mean: np.ndarray, reports: list, lam=None):
        self.U = np.asarray(U, dtype=float)
        self.mean = np.asarray(mean, dtype=float)
        self.reports = reports
        self.lam = lam
        self.coords = pj.project(self.U, self.mean, L4, me.A)
        self.edges = pj.grid_edges(self.coords, GRID)
        self.bins = pj.bin_index(self.coords, self.edges)
        # Wells and barriers live on the graph of cells linked by one substitution.
        self.adj = pj.substitution_adjacency(self.bins.reshape((me.A,) * L4), GRID * GRID)
        self.centres = [(e[:-1] + e[1:]) / 2 for e in self.edges]

    @classmethod
    def pca(cls, P: np.ndarray) -> "Basis":
        C = pj.indicator_covariance(P)
        lam, U = pj.principal_axes(C)
        mean = pj.site_marginals(P).ravel()
        reports = [pj.axis_report(U[:, k], lam, k, L4, me.ALPHABET, TOP_LOADINGS)
                   for k in range(2)]
        return cls(U[:, :2], mean, reports, lam=lam)

    @classmethod
    def named(cls, name: str, model: me.MaxEnt) -> "Basis":
        r = bas.compute_basis(name, model=model, L=L4, A=me.A)
        return cls(r["U"], r["mean"], r["annotation"])

    def plane_share(self, P: np.ndarray) -> float:
        """Share of a model's total indicator variance lying in this (fixed) plane."""
        C = pj.indicator_covariance(P)
        return float(np.trace(self.U.T @ C @ self.U) / np.trace(C))

    def cumulative_share(self) -> float:
        return float(self.reports[0]["variance_share"] + self.reports[1]["variance_share"])

    def summary(self) -> dict:
        return {"axes": self.reports, "cumulative_share": self.cumulative_share(),
                "x": self.centres[0].round(5).tolist(), "y": self.centres[1].round(5).tolist(),
                "cell": [float(np.diff(e)[0]) for e in self.edges]}


def cell_xy(basis: Basis, cell: int) -> tuple[float, float]:
    r, c = divmod(int(cell), GRID)
    return float(basis.centres[0][c]), float(basis.centres[1][r])


def landscape(m: me.MaxEnt, basis: Basis, n_string_minima: int) -> dict:
    """Bin energies, persistent wells with their basins' most probable strings, the
    main barrier and the string-level minima of one model in one basis."""
    lp = m.log_prob().ravel() / LN2                       # log2 P(s); -inf if forbidden
    E, count = pj.bin_energy(lp, basis.bins, GRID * GRID)
    allw = pj.persistence(E, basis.adj)
    by_cell = {w["cell"]: w for w in allw}
    lab = pj.significant_basins(E, allw, PERSIST_BITS, basis.adj)
    sbasin = lab.ravel()[basis.bins]
    wells = []
    for w in (w for w in allw if w["persistence"] >= PERSIST_BITS):
        idx = np.flatnonzero(sbasin == w["cell"])
        top = idx[np.argsort(-lp[idx], kind="stable")[:BASIN_WORDS]]
        cell_idx = np.flatnonzero(basis.bins == w["cell"])
        cell_top = cell_idx[np.argmax(lp[cell_idx])]      # most probable string in the minimum cell itself
        x, y = cell_xy(basis, w["cell"])
        wells.append({"cell": w["cell"], "x": x, "y": y, "E": w["E_min"],
                      "persistence": w["persistence"], "essential": w["essential"],
                      "words": [me.unravel(i) for i in top],
                      "min_cell_top": {"w": me.unravel(int(cell_top)), "L": L4,
                                       "p": float(np.exp2(lp[cell_top])), "approx": False},
                      "mass": float(np.exp2(lp[idx]).sum()), "cells": int((lab == w["cell"]).sum())})
    barrier = None
    merging = [w for w in allw if w["persistence"] >= PERSIST_BITS and not w["essential"]]
    if merging:
        w = merging[0]                                    # most persistent well that merges
        other = by_cell[w["merges_into"]]
        x, y = cell_xy(basis, w["saddle"])
        barrier = {"x": x, "y": y, "E": w["E_saddle"], "cell": w["saddle"],
                   "wells": [w["cell"], other["cell"]],
                   "above": [w["E_saddle"] - w["E_min"], w["E_saddle"] - other["E_min"]]}
    mins = np.flatnonzero(me.local_minima(m.energy()).ravel())
    top = mins[np.argsort(-lp[mins], kind="stable")[:n_string_minima]]
    string_minima = [{"word": me.unravel(i), "x": float(basis.coords[0, i]),
                      "y": float(basis.coords[1, i]), "E_bin": float(E[basis.bins[i]]),
                      "E": float(-lp[i])} for i in top]
    return {"E": E, "count": count, "lp": lp, "wells": wells, "barrier": barrier,
            "n_wells_all": int(sum(w["persistence"] > 0 for w in allw)),
            "n_string_minima": int(mins.size), "string_minima": string_minima,
            "labels": lab, "S2": m.entropy_bits()}


def cell_top_words(lp: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Most probable string of every grid cell ('' if empty, '-' if all forbidden)."""
    out = np.full(GRID * GRID, "", dtype=object)
    order = np.lexsort((-lp, bins))
    b = bins[order]
    first = np.r_[0, np.flatnonzero(np.diff(b)) + 1]
    for cell, i in zip(b[first], order[first]):
        out[cell] = me.unravel(i) if np.isfinite(lp[i]) else "-"
    return out


def surface_rows(E: np.ndarray, digits: int = 3) -> list:
    """Grid as nested lists, NaN/inf -> None (gaps in the surface)."""
    Eg = E.reshape(GRID, GRID)
    return [[round(float(v), digits) if np.isfinite(v) else None for v in row] for row in Eg]


def _axis_names(name: str) -> list:
    return ["PC1", "PC2"] if name == "pca" else ["axis 1", "axis 2"]


def landscape_dict(basis: Basis, ls: dict, name: str = "pca") -> dict:
    """Per-basis landscape payload for one model (projection-dependent quantities)."""
    words = cell_top_words(ls["lp"], basis.bins).reshape(GRID, GRID)
    E, count = ls["E"], ls["count"]
    return {"basis": name, "axis_names": _axis_names(name),
            "z": surface_rows(E), "words": words.tolist(),
            "count": count.astype(int).reshape(GRID, GRID).tolist(),
            "wells": ls["wells"], "barrier": ls["barrier"], "n_wells_all": ls["n_wells_all"],
            "n_string_minima": ls["n_string_minima"], "string_minima": ls["string_minima"],
            "n_empty": int((count == 0).sum()), "n_forbidden": int(np.isposinf(E).sum()),
            "n_cells": GRID * GRID, "forbidden_strings": int(np.isneginf(ls["lp"]).sum()),
            "axes": basis.reports, "cumulative_share": basis.cumulative_share(),
            "x": basis.centres[0].round(5).tolist(), "y": basis.centres[1].round(5).tolist(),
            "cell": [float(np.diff(e)[0]) for e in basis.edges]}


def tab1_model(key: str, label: str, m: me.MaxEnt, meta: dict) -> dict:
    basis = Basis.pca(m.prob())
    ls = landscape(m, basis, TOP_STRING_MINIMA)
    out = {"key": key, "label": label, **meta, "S2": ls["S2"],
           "total_variance": float(basis.lam.sum()),
           "rank": int((basis.lam > 1e-12 * basis.lam[0]).sum()),
           **landscape_dict(basis, ls, "pca"), "bases": {}}
    out["V_b64"] = pack_V_b64(pack_V(m))
    engines = {"pca": engine_data(basis, "pca")}
    em = energy_metric_coords(key)
    if em is not None:
        engines["energy_metric"] = em
    for name in bas.BASES:
        if name == "pca":
            continue
        if bas.BASES[name].get("kind") == "coords":
            continue                       # coords (diagnostic) bases are engine-only
        b = Basis.named(name, m)
        out["bases"][name] = landscape_dict(b, landscape(m, b, TOP_STRING_MINIMA), name)
        engines[name] = engine_data(b, name)
    out["engines"] = engines
    return out, basis, ls


# Tab-2 per-frame analysis runs in worker processes sharing the fixed basis.
_BASIS: Basis | None = None


def _init_worker(basis: Basis) -> None:
    global _BASIS
    _BASIS = basis


def _job_frame(V: np.ndarray) -> dict:
    m = unpack_V(V)
    ls = landscape(m, _BASIS, 0)
    q = np.where(np.isfinite(ls["E"]), np.round(ls["E"] * CENTIBITS), -32768)
    assert np.all(np.abs(q[np.isfinite(ls["E"])]) < 32767)
    return {"q": q.astype("<i2"), "wells": ls["wells"], "barrier": ls["barrier"],
            "S2": ls["S2"], "n_string_minima": ls["n_string_minima"],
            "plane_share": _BASIS.plane_share(m.prob()),
            "well_cells": [w["cell"] for w in ls["wells"]]}


def stage_bases() -> None:
    """Precompute the trajectory (tab 2) in every non-PCA basis: per-frame bin energies
    and wells in that basis. Smoothing is now live (Boltzmann-domain kernel regression
    in the browser), so no smoothed surface is precomputed here. The basis is fixed
    across frames, computed from the final model (d), as for PCA."""
    js, npz = load_json(), load_npz()
    V = npz.get("V_traj")
    if V is None:
        raise RuntimeError("run 'trajectory' before 'bases'")
    nf = V.shape[0]
    sigma2 = js["trajectory"]["sigma2"]
    tokens = npz["traj_tokens"]
    m_final = unpack_V(V[-1], me.prior_lambda(sigma2, tokens[-1]))
    done = js.get("bases_traj", {})
    for name in bas.BASES:
        if name == "pca" or name in done:
            continue
        if bas.BASES[name].get("kind") == "coords":
            continue                       # coords (diagnostic) bases are engine-only
        basis = Basis.named(name, m_final)
        log(f"bases: {name} ({nf} frames in the fixed final basis)")
        with ProcessPoolExecutor(WORKERS, initializer=_init_worker, initargs=(basis,)) as ex:
            frames = list(ex.map(_job_frame, list(V), chunksize=4))
        q = np.stack([f["q"] for f in frames])
        fin = q[q > -32768]
        done[name] = {
            "zb64": base64.b64encode(q.tobytes()).decode(),
            "x": basis.centres[0].round(5).tolist(), "y": basis.centres[1].round(5).tolist(),
            "zmin": float(fin.min()) / CENTIBITS, "zmax": float(fin.max()) / CENTIBITS,
            "wells": [f["wells"] for f in frames], "barrier": [f["barrier"] for f in frames],
            "plane_share": [f["plane_share"] for f in frames],
            "axes": basis.reports, "cumulative_share": basis.cumulative_share(),
            "axis_names": _axis_names(name), "basis": name}
        js["bases_traj"] = done
        save_json(js)
        log(f"  {name}: {nf} frames")


BLURBS_FILE = ROOT / "scripts" / "artifact_blurbs.json"
INTRO_FILE = ROOT / "scripts" / "artifact_intro.html"
UI_COPY_FILE = ROOT / "scripts" / "ui_copy.json"


def _load_blurbs() -> dict:
    """Short HTML explanations (keys: clip, above_max, smoothing, heatmap, sb_vs_new),
    written by TheoryWriter; injected verbatim into the artifact's in-context notes."""
    return json.loads(BLURBS_FILE.read_text()) if BLURBS_FILE.exists() else {}


def _load_ui_copy() -> dict:
    """Single source of truth for the sidebar/controls copy (written by UiCopyWriter).
    The build fails if the file is missing, per the simplification contract."""
    if not UI_COPY_FILE.exists():
        raise RuntimeError(f"{UI_COPY_FILE} is missing: required for the UI copy binding")
    return json.loads(UI_COPY_FILE.read_text())


def _load_intro() -> str:
    """Short orientation tab from ui_copy.json, with the old long theory moved into a
    closed 'Technical details' <details> at the bottom (KaTeX preamble preserved)."""
    ui = _load_ui_copy()
    tab0 = ui.get("tab0", "")
    details = ui.get("details_heading", "Technical details")
    txt = INTRO_FILE.read_text() if INTRO_FILE.exists() else ""
    idx = txt.find('<section id="tab0"')
    preamble = txt[:idx] if idx >= 0 else txt
    old = txt[idx:] if idx >= 0 else ""
    m = re.search(r"<style>.*?</style>", old, re.S)
    style = m.group(0) if m else ""
    theory = old[m.end():] if m else old
    theory = theory.rsplit("</section>", 1)[0].strip()
    # The theory fragment carries its own scroll wrapper; inside the short tab it must
    # flow with the page instead (one scroller, details below the summary).
    wrap_open = '<div class="intro-body"><div class="intro-width">'
    wrap_close = "</div></div>"
    if not (theory.startswith(wrap_open) and theory.endswith(wrap_close)):
        raise RuntimeError("artifact_intro.html theory wrapper changed; update _load_intro")
    theory = theory[len(wrap_open):-len(wrap_close)]
    inject = style + f'<details class="tech"><summary>{details}</summary>{theory}</details>'
    tail = wrap_close + "</section>"
    tab0 = tab0.rstrip()
    if not tab0.endswith(tail):
        raise RuntimeError("ui_copy.json tab0 must end with the intro-body wrapper and </section>")
    return preamble + tab0[:-len(tail)] + inject + tail


def _bases_section() -> str:
    meta = json.loads(BASES_META_FILE.read_text()).get("bases", {}) if BASES_META_FILE.exists() else {}
    items = [f"<li><b>{info['label']}</b> — {info.get('summary', '')}</li>"
             for name, info in bas.BASES.items()]
    return ('<h2>Projection bases (tabs 1 and 2)</h2>'
            '<p>The S&amp;B landscape can be viewed in any of these linear 2-D projections (selector on tabs 1 and 2); '
            '<b>PCA is the default</b>. The five linear bases are PCA, coupling-spectral, energy-aligned (PLS), '
            'tempered PCA (α = 0.75) and well-discriminant (LDA), plus one clearly-labelled degenerate diagnostic '
            'layout (energy-metric). They differ in coverage, well geometry, within-cell energy homogeneity, '
            'substitution continuity and stability, and none removes the cell-to-cell scatter of E<sub>bin</sub> '
            '(energy spread inside a cell ≈ 6.8 bits) — see the comparison table. Each is rigorously justified — '
            'assumptions, limitations and metric numbers — in the "Why this basis" panel of tab 1.</p>'
            '<ul>' + "".join(items) + "</ul>")


BASES_META_FILE = ROOT / "scripts" / "artifact_bases.json"
BASIS_STUDY_FILE = ms.corpus.DATA / "derived" / "basis_study.json"


def _load_bases_registry() -> list:
    """The selectable-basis registry (id/label/summary) merged with the per-basis
    KaTeX justification HTML and metric numbers from artifact_bases.json."""
    meta = json.loads(BASES_META_FILE.read_text())["bases"] if BASES_META_FILE.exists() else {}
    out = []
    for name, info in bas.BASES.items():
        bm = meta.get(name, {})
        out.append({"id": name, "label": info["label"], "summary": info["summary"],
                    "kind": info.get("kind", "linear"),
                    "diagnostic": bool(info.get("diagnostic", False)),
                    "html": bm.get("html", ""), "metrics": bm.get("metrics", {})})
    return out


def _load_bases_compare() -> dict:
    """Per-model basis comparison table (smoothness metrics, basis_study.json)."""
    if not BASIS_STUDY_FILE.exists():
        return {}
    return json.loads(BASIS_STUDY_FILE.read_text()).get("smoothness", {}).get("models", {})


NM_JSON = ms.corpus.DATA / "derived" / "nm_landscape.json"
NM_NPZ = ms.corpus.DATA / "derived" / "nm_landscape.npz"


def _assemble_newmodel() -> dict | None:
    """Load NewModelLandscape's ``nm_landscape.{json,npz}`` (smoothing is now a live
    Boltzmann-domain kernel regression in the browser, so no smoothing precompute
    here). Returns None while the data is still being written (the tabs then show a
    pending note)."""
    if not NM_JSON.exists():
        return None
    nm = json.loads(NM_JSON.read_text()).get("summary")
    if nm is None or "models" not in nm:
        return None                       # still being written; tabs show a pending note
    nm.pop("minimax_paths", None)             # new-model paths are not shown: no endpoint selector
    for lv in (nm.get("levels") or {}).values():
        lv.pop("barrier_path", None)
    grid = nm["grid"]
    for key, m in nm["models"].items():
        m["grid"] = grid
    if "traj" in nm and nm["traj"].get("zb64"):
        nf = np.frombuffer(base64.b64decode(nm["traj"]["zb64"]), dtype="<i2").size // (grid * grid)
        if NM_NPZ.exists():
            with np.load(NM_NPZ) as z:
                if "mask_traj" in z:
                    mt = z["mask_traj"].reshape(-1, grid, grid)
                    nm["traj"]["mask_b64"] = base64.b64encode(mt.astype(np.int8).tobytes()).decode()
                # Per-level trajectory surfaces (tab 4 resolution slider): z_traj_L{G}
                # and mask_traj_L{G} for every level in levels_order, plus the per-level
                # zmin/zmax (the colour scale changes with the coarser/finer binning).
                traj_levels = {}
                for G in nm.get("levels_order", []):
                    key = f"z_traj_L{G}"
                    if key not in z:
                        continue
                    zt = z[key]  # (nframes, G*G) int16, -32768 = gap, /100 = bits
                    zt = np.ascontiguousarray(zt)
                    fin = zt.astype(np.float32)
                    fin[zt <= -32768] = np.nan
                    entry = {"zb64": base64.b64encode(zlib.compress(zt.tobytes(), 9)).decode(),
                             "scale": 100, "deflated": True,
                             "zmin": float(np.nanmin(fin)) / 100.0, "zmax": float(np.nanmax(fin)) / 100.0}
                    mk = z.get(f"mask_traj_L{G}")
                    if mk is not None:
                        entry["mask_b64"] = base64.b64encode(zlib.compress(np.ascontiguousarray(mk).astype(np.int8).tobytes(), 9)).decode()
                    traj_levels[str(G)] = entry
                if traj_levels:
                    nm["traj"]["levels"] = traj_levels
        log(f"new model: {nf} frames; {len(nm['models'])} static model(s)")
    return _sanitize_json(nm)


def _sanitize_json(o):
    """Replace non-finite floats (inf/nan, e.g. the importance-sampled cells' z_se
    where zero model mass gives an undefined standard error) with None so the data
    blob stays valid JSON (``allow_nan=False``)."""
    if isinstance(o, dict):
        return {k: _sanitize_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize_json(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def build() -> None:
    js, npz = load_json(), load_npz()
    traj = js["trajectory"]
    sigma2 = traj["sigma2"]
    tokens = npz["traj_tokens"]
    fit_meta = {
        "austen": {"desc": "8 Austen novels (S&B ref. 16), four-letter tokens found in the "
                           "dictionary, full sample; exact pairwise fit (unseen letter pairs "
                           "forbidden). Strict-minima count 149 (148 at fit tolerance): 'lamb' vs "
                           "'limb' differ by <4e-4 bits, below the fit tolerance.", "mode": "exact"},
        "twitter2023": {"desc": "Twitter 2023 (enryu43/twitter100m, shards 00-15), rep-0 "
                                "subsample of N = Austen token count drawn without replacement "
                                f"(seed {js['fits']['twitter2023'].get('seed')}); exact fit.",
                        "mode": "exact"},
        "bluesky2024": {"desc": "Bluesky 2024, rep-0 subsample of N = Austen token count "
                                f"(seed {js['fits']['bluesky2024'].get('seed')}); exact fit.",
                        "mode": "exact"},
    }
    models, basis_final = [], None
    for key, label in MODELS:
        if key == "pooled":
            m = unpack_V(npz["V_traj"][-1], me.prior_lambda(sigma2, tokens[-1]))
            d = traj["diagnostics"][-1]
            meta = {"desc": "All Twitter months 2006-03..2023-04 (shards 00-15) pooled; MAP "
                            f"pairwise fit with Gaussian prior σ² = {sigma2:.4g} chosen by "
                            f"{js['prior']['n_folds']}-fold CV (the maxent.select_prior procedure). Same model "
                            "as the final frame of the trajectory tab.",
                    "mode": "l2", "tokens": float(tokens[-1]), "sigma2": sigma2,
                    "converged": d["converged"], "max_gradient_in_se": d["max_gradient_in_se"]}
        else:
            m = unpack_V(npz[f"V_{key}"])
            f = js["fits"][key]
            meta = {**fit_meta[key], "tokens": f["tokens"], "types": f["types"],
                    "S2_study": f["S2_study"], "S2_abs_diff": f["S2_abs_diff"],
                    "converged": f["converged"], "max_gradient_in_se": f["max_gradient_in_se"],
                    "C_check": f["max_abs_C_model_minus_C_empirical"]}
        rec, basis, ls = tab1_model(key, label, m, meta)
        models.append(rec)
        log(f"  {label}: PC1 {rec['axes'][0]['variance_share']:.4f} PC2 "
            f"{rec['axes'][1]['variance_share']:.4f} plane {rec['cumulative_share']:.4f}; "
            f"{len(rec['wells'])} wells >= {PERSIST_BITS} bit; {rec['n_string_minima']} string minima")
        if key == "pooled":
            basis_final, final_ls = basis, ls

    log(f"trajectory: {npz['V_traj'].shape[0]} frames in the fixed final basis")
    with ProcessPoolExecutor(WORKERS, initializer=_init_worker, initargs=(basis_final,)) as ex:
        frames = list(ex.map(_job_frame, list(npz["V_traj"]), chunksize=4))
    tracking = track_wells(final_ls, frames, traj["months"])
    for t in tracking:
        log(f"  final well {t['words'][0]}: persistence >= {PERSIST_BITS} bit first at "
            f"{t['first_month']} (frame {t['first_frame']}), continuously from "
            f"{t['stable_month']}")
    q = np.stack([f["q"] for f in frames])
    fin = q[q > -32768]
    diag = traj["diagnostics"]
    tab2 = {"months": traj["months"], "tokens": tokens.tolist(), "sigma2": sigma2,
            "S2": [f["S2"] for f in frames],
            "n_string_minima": [f["n_string_minima"] for f in frames],
            "plane_share": [f["plane_share"] for f in frames],
            "wells": [f["wells"] for f in frames], "barrier": [f["barrier"] for f in frames],
            "converged": [d["converged"] for d in diag],
            "max_gradient_in_se": [d["max_gradient_in_se"] for d in diag],
            "iterations": [d["iterations"] for d in diag],
            "chain_bounds": traj["chain_bounds"],
            "zb64": base64.b64encode(q.tobytes()).decode(), "scale": CENTIBITS,
            "zmin": float(fin.min()) / CENTIBITS, "zmax": float(fin.max()) / CENTIBITS,
            "x": basis_final.centres[0].round(5).tolist(),
            "y": basis_final.centres[1].round(5).tolist(),
            "tracking": tracking}
    prior = js["prior"]
    tab2["bases"] = js.get("bases_traj", {})
    data = {"grid": GRID, "sub": SUB, "sub_traj": SUB_TRAJ, "contours": CONTOURS,
            "persist_bits": PERSIST_BITS, "uniform_bits": UNIFORM_BITS,
            "basin_words": BASIN_WORDS, "top_minima": TOP_STRING_MINIMA,
            "models": models, "traj": tab2,
            "V_traj_b64": pack_V_b64(npz["V_traj"]),
            "bases": _load_bases_registry(), "bases_compare": _load_bases_compare(),
            "bases_smooth_stride": BASIS_SMOOTH_STRIDE,
            "blurbs": _load_blurbs(), "newmodel": _assemble_newmodel(),
            "ui_copy": _load_ui_copy(),
            "prior": {k: prior[k] for k in ("sigma2", "bits", "se", "best_sigma2", "interior",
                                            "n_folds", "seed", "tokens")}}
    js["summary"] = {"models": {r["key"]: {"axes": r["axes"], "cumulative_share": r["cumulative_share"],
                                           "wells": r["wells"], "barrier": r["barrier"],
                                           "n_string_minima": r["n_string_minima"]}
                                for r in models},
                     "sigma2": sigma2, "tracking": tracking}
    save_json(js)
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = (ROOT / "scripts" / "landscape_template.html").read_text()
    OUT_HTML.write_text(html.replace("__PLOTLY__", PLOTLY)
                            .replace("__INTRO__", _load_intro())
                            .replace("__ENGINE__", _load_engine())
                            .replace("__DATA__", blob))
    log(f"wrote {OUT_HTML} ({OUT_HTML.stat().st_size / 1e6:.2f} MB)")


ENGINE_FILE = ROOT / "scripts" / "landscape_engine.js"


def _load_engine() -> str:
    return ENGINE_FILE.read_text() if ENGINE_FILE.exists() else ""


def track_wells(final_ls: dict, frames: list[dict], months: list[str]) -> list[dict]:
    """For each labelled well of the final frame: the first frame with a well of
    persistence >= PERSIST_BITS whose minimum lies in the final well's basin, and
    the frame from which that holds in every later frame."""
    lab = final_ls["labels"].ravel()
    out = []
    for w in final_ls["wells"]:
        hit = [any(lab[c] == w["cell"] for c in f["well_cells"]) for f in frames]
        first = hit.index(True) if any(hit) else None
        stable = None
        for t in range(len(hit) - 1, -1, -1):
            if not hit[t]:
                break
            stable = t
        out.append({"cell": w["cell"], "words": w["words"], "persistence": w["persistence"],
                    "essential": w["essential"],
                    "first_frame": first, "first_month": months[first] if first is not None else None,
                    "stable_frame": stable,
                    "stable_month": months[stable] if stable is not None else None})
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
