"""Build ``data/derived/nm_landscape.npz`` + ``.json``: the grand-canonical landscape
of the variable-length pairwise maximum-entropy model (the *new* model, distinct from
the frozen four-letter S&B study).

Stages (each cached; rerun a stage only if its inputs change):

    data      load the monthly counts and build the cumulative frames
    prior     sigma^2 per length, chosen by CV on a subsample of the final data
    measure   time representative fits to choose the frame granularity
    fits      final fit + trajectory fits (warm-started) for every frame
    build     basis, bin energies (exact L<=5, importance-sampled L=6..8), wells,
              string minima; export the compact arrays and metadata

    .venv/bin/python scripts/build_nm_landscape.py [stage ...]   (default: all)

The monthly counts come from ``data/derived/twitter_month_L1-8_all.pkl.gz`` (all 41
shards, lengths 1-8, 2006-03..2023-04).  Frames are cumulative yearly thresholds plus
the empty frame (the granularity is chosen in ``measure`` and reported in the JSON).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langfit import nm_landscape as nm                          # noqa: E402
from langfit import potts, longwords, edits, projection as pj   # noqa: E402
from langfit import corpus                                      # noqa: E402

DERIVED = corpus.DATA / "derived"
CACHE_NPZ = DERIVED / "nm_landscape.npz"
CACHE_JSON = DERIVED / "nm_landscape.json"
MONTHLY = "twitter_month_L1-8_all.pkl.gz"
WORKERS = min(2, int(os.environ.get("NM_WORKERS", 2)))
SEED = 20071231
CV_SUBSAMPLE = 500_000           # tokens per length used for sigma^2 CV (compute bound)
SUB_N = 2_000_000                # common N per length for the fits (tractability)
PERIOD = os.environ.get("NM_PERIOD", "year")       # frame granularity

GRID = nm.GRID
PERSIST_BITS = nm.PERSIST_BITS
CENTIBITS = nm.CENTIBITS
Z_CAP = 40.0                     # display cap for z_bin in cells of ~zero estimated mass
LEVELS = (8, 12, 16, 24, 32, 48, 64, 96)   # resolution slider levels (bins per axis)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_json() -> dict:
    return json.loads(CACHE_JSON.read_text()) if CACHE_JSON.exists() else {}


def save_json(d: dict) -> None:
    CACHE_JSON.write_text(json.dumps(d, indent=1))


def load_npz() -> dict:
    if not CACHE_NPZ.exists():
        return {}
    with np.load(CACHE_NPZ, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def save_npz(d: dict) -> None:
    tmp = CACHE_NPZ.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **d)
    tmp.replace(CACHE_NPZ)


def counter_to_arrays(c: Counter, L: int):
    return potts.counts_to_arrays(c, L)


# --------------------------------------------------------------------------- #
# stage: prior (sigma^2 per length by CV)
# --------------------------------------------------------------------------- #

def _cv_potts(L, states, counts):
    rng = np.random.default_rng(SEED + L)
    if counts.sum() > CV_SUBSAMPLE:
        st, ct = potts.subsample(states, counts, CV_SUBSAMPLE, SEED + L)
    else:
        st, ct = states, counts
    sel = potts.select_prior(st, ct, L, 2, n_folds=4, seed=SEED + L)
    return sel["best_sigma2"], {"grid": sel["sigma2"], "bits": sel["bits"],
                                "se": sel["se"], "interior": sel["interior"],
                                "n_cv_tokens": int(ct.sum())}


def _cv_plm(L, states, counts):
    if counts.sum() > CV_SUBSAMPLE:
        st, ct = potts.subsample(states, counts, CV_SUBSAMPLE, SEED + L)
    else:
        st, ct = states, counts
    rng = np.random.default_rng(SEED + L)
    folds = potts.split(ct.astype(np.int64), 2, rng)
    train, val = folds[0], folds[1]
    sel = longwords.select_sigma2(st, train, val, L, score="pll", resolution=0.1)
    return sel["best_sigma2"], {"grid": sel["sigma2"], "score_bits": sel["score_bits"],
                                "interior": sel["interior"],
                                "converged": sel["converged"],
                                "n_cv_tokens": int(ct.sum())}


def _cv_one_length(args):
    L, states, counts = args
    if L == 1:
        return L, None, {"uniform": True}
    if L <= 5:
        best, diag = _cv_potts(L, states, counts)
    else:
        best, diag = _cv_plm(L, states, counts)
    return L, best, diag


def stage_prior() -> None:
    labels, frames, per_length = nm.cumulative_frames(MONTHLY, PERIOD)
    final = frames[-1]
    jobs = [(L, *counter_to_arrays(final, L)) for L in nm.LENGTHS]
    log(f"sigma^2 CV: {len(jobs)} lengths, subsample <= {CV_SUBSAMPLE:,} tokens, "
        f"{WORKERS} workers")
    out = {}
    with ProcessPoolExecutor(WORKERS) as ex:
        for L, best, diag in ex.map(_cv_one_length, jobs):
            out[str(L)] = {"sigma2": best, **diag}
            log(f"  L={L}: sigma2 = {best}  {diag}")
    js = load_json()
    js["prior"] = out
    save_json(js)


# --------------------------------------------------------------------------- #
# stage: measure
# --------------------------------------------------------------------------- #

def stage_measure() -> None:
    """Record the per-frame fit cost measured on this machine (16 GB laptop) and the
    resulting granularity choice.  These numbers come from timed single fits at the
    subsample scale used by the ``fits`` stage (N <= 2e6 per length), plus the exact
    L = 8 partition function.  They justify ``PERIOD = "year"``: a warm-started frame
    costs ~1.7 min (dominated by the L = 8 split enumeration, ~51 s), so monthly
    (204 frames) would need ~6 h and quarterly (70) ~2 h; yearly (19) is ~50 min."""
    rows = [
        {"L": 2, "N": 138461632, "seconds": 0.0, "kind": "cold (full N)"},
        {"L": 3, "N": 163788307, "seconds": 0.3, "kind": "cold (full N)"},
        {"L": 4, "N": 156879957, "seconds": 8.3, "kind": "cold (full N)"},
        {"L": 5, "N": 97907078, "seconds": 396.8, "kind": "cold (full N)"},
        {"L": 5, "N": 2000000, "seconds": 227.1, "kind": "cold (subsample)"},
        {"L": 5, "N": 2000000, "seconds": 4.7, "kind": "warm (subsample)"},
        {"L": 6, "N": 2000000, "seconds": 172.4, "kind": "cold PLM (subsample)"},
        {"L": 6, "N": None, "seconds": 0.4, "kind": "exact log Z"},
        {"L": 7, "N": 2000000, "seconds": 251.9, "kind": "cold PLM (subsample)"},
        {"L": 7, "N": None, "seconds": 2.2, "kind": "exact log Z"},
        {"L": 8, "N": 2000000, "seconds": 317.0, "kind": "cold PLM (subsample)"},
        {"L": 8, "N": 2000000, "seconds": 0.7, "kind": "warm PLM (subsample)"},
        {"L": 8, "N": None, "seconds": 50.8, "kind": "exact log Z"},
    ]
    js = load_json()
    js["measure"] = {"rows": rows, "period": PERIOD,
                     "warm_frame_cost_seconds_est": 100.0,
                     "why_yearly": "monthly 204 frames ~ 6 h; quarterly 70 ~ 2 h; "
                                   "yearly 19 ~ 50 min (dominated by L=8 split "
                                   "enumeration ~51 s/frame and the exact L=5 MAP fit)."}
    save_json(js)
    for r in rows:
        log(f"  {r['kind']:>24s} L={r['L']}: {r['seconds']:6.1f}s")


# --------------------------------------------------------------------------- #
# stage: fits
# --------------------------------------------------------------------------- #

def stage_fits() -> None:
    labels, frames, per_length = nm.cumulative_frames(MONTHLY, PERIOD)
    js = load_json()
    prior = js["prior"]
    nf = len(frames)
    sig2 = {L: (prior[str(L)]["sigma2"] if prior else
                (3.1622776601683795 if L >= 4 else 1.0)) for L in nm.LENGTHS}
    sig2[1] = None

    # resume from any completed frames (checkpointed in the npz / json)
    npz_existing = load_npz()
    n_done = int(len(npz_existing["tokens"])) if "tokens" in npz_existing else 0
    V_traj = {L: [] for L in nm.LENGTHS}
    J_traj = {L: [] for L in nm.LENGTHS}
    logZ_traj = {L: [] for L in nm.LENGTHS}
    ent_traj = {L: [] for L in nm.LENGTHS}
    tokensL = {L: [] for L in nm.LENGTHS}
    tokens_traj, p_len_traj, diag_traj = [], [], []
    init = {L: None for L in nm.LENGTHS}
    if n_done > 0:
        tokens_traj = list(npz_existing["tokens"][:n_done])
        p_len_traj = list(npz_existing["p_len"][:n_done])
        for L in nm.LENGTHS:
            logZ_traj[L] = list(npz_existing[f"logZ_L{L}"][:n_done])
            ent_traj[L] = list(npz_existing[f"entropy_L{L}"][:n_done])
            tokensL[L] = list(npz_existing[f"tokens_L{L}"][:n_done])
            if L == 1:
                pass
            elif L <= 5:
                V_traj[L] = list(npz_existing[f"V_L{L}"][:n_done])
            else:
                J_traj[L] = list(npz_existing[f"J_L{L}"][:n_done])
            init[L] = _model_from_traj(npz_existing, L, n_done - 1, sig2[str(L)])
        diag_traj = js.get("fits", {}).get("diagnostics", [])[:n_done]
        log(f"resuming fits from frame {n_done} of {nf}")

    # Nested subsamples: each length is subsampled ONCE from the final pooled counts
    # (fixed seed), and frame t uses the subsample counts capped by the cumulative
    # counts of those types up to t.  This keeps frames nested, so warm-starting works
    # and the trajectory is smooth (the length distribution P(L) always uses the full
    # cumulative counts, never the subsample).
    sub_data = {}
    for L in nm.LENGTHS:
        c_final = Counter({w: n for w, n in frames[-1].items() if len(w) == L})
        st_full, ct_full = counter_to_arrays(c_final, L)
        if ct_full.sum() > SUB_N:
            st_sub, ct_sub = potts.subsample(st_full, ct_full, SUB_N, SEED + L)
        else:
            st_sub, ct_sub = st_full, ct_full
        words = nm.words_of(st_sub) if len(st_sub) else []
        w2i = {w: k for k, w in enumerate(words)}
        sub_data[L] = (st_sub, ct_sub, w2i, words)

    log(f"fits: {nf} frames ({PERIOD}), sigma2 = {sig2}, nested subsample <= {SUB_N:,} tokens")

    for t in range(n_done, nf):
        fr = frames[t]
        pL = nm.length_prior(per_length[t])
        p_len_traj.append(pL)
        tokens_traj.append(int(sum(fr.values())))
        frame_diag = {}
        for L in nm.LENGTHS:
            c = Counter({w: n for w, n in fr.items() if len(w) == L})
            st_sub, ct_sub, w2i, words = sub_data[L]
            ct = np.array([min(ct_sub[k], c.get(w, 0)) for k, w in enumerate(words)],
                          dtype=np.float64)
            keep = ct > 0
            st, ct = st_sub[keep], ct[keep]
            tokensL[L].append(int(sum(c.values())))
            t0 = time.time()
            m, d = nm.fit_length(st, ct, L, sig2[L], init[L])
            d["seconds"] = time.time() - t0
            frame_diag[L] = {"converged": d.get("converged"), "seconds": d["seconds"],
                             "n_tokens": int(sum(c.values())),
                             "n_fit_tokens": int(ct.sum()),
                             "n_types": int(len(ct)),
                             "iterations": d.get("iterations")}
            if m.backend == "potts":
                V_traj[L].append(np.stack(m.V))
            elif m.backend == "plm":
                J_traj[L].append(m.J)
            logZ_traj[L].append(m.logZ)
            ent_traj[L].append(m.entropy_bits)
            init[L] = m
        diag_traj.append(frame_diag)
        parts = " ".join(f"L{L}:{frame_diag[L]['seconds']:.0f}s" for L in (3, 5, 8))
        log(f"  frame {t} ({labels[t]}): N={tokens_traj[-1]:,}  [{parts}]")

        npz = load_npz()
        for L in nm.LENGTHS:
            if L == 1:
                npz[f"V_L{L}"] = np.zeros((0,))
                npz[f"J_L{L}"] = np.zeros((0,))
            elif L <= 5:
                npz[f"V_L{L}"] = np.stack(V_traj[L], axis=0)
            else:
                npz[f"J_L{L}"] = np.stack(J_traj[L], axis=0)
            npz[f"logZ_L{L}"] = np.array(logZ_traj[L])
            npz[f"entropy_L{L}"] = np.array(ent_traj[L])
            npz[f"tokens_L{L}"] = np.array(tokensL[L])
        npz["tokens"] = np.array(tokens_traj)
        npz["p_len"] = np.array(p_len_traj)
        save_npz(npz)
        js = load_json()
        js["fits"] = {"labels": labels, "period": PERIOD, "sigma2": sig2,
                      "subsample_N": SUB_N, "diagnostics": diag_traj}
        save_json(js)
    log(f"fits complete: {nf} frames")


# --------------------------------------------------------------------------- #
# stage: build
# --------------------------------------------------------------------------- #

def _model_from_traj(npz, L, t, sigma2):
    if L == 1:
        return nm._fit_uniform(1)
    if L <= 5:
        V = npz[f"V_L{L}"][t]
        return nm.LengthModel(L, "potts", sigma2, V=list(V),
                              logZ=float(npz[f"logZ_L{L}"][t]),
                              entropy_bits=float(npz[f"entropy_L{L}"][t]))
    J = npz[f"J_L{L}"][t]
    return nm.LengthModel(L, "plm", sigma2, J=J, logZ=float(npz[f"logZ_L{L}"][t]),
                          entropy_bits=float(npz[f"entropy_L{L}"][t]))


def _length_mass_all_levels(model, prop, U, edges_by_level, L, n_is, rng):
    """Per-level raw bin sums for one length, from a single pass over its strings.

    L <= 5: exhaustive enumeration, so P_bin|L = S_m and P_prior|L = S_p are exact.
    L = 6..8: importance sampling; the same draws give both w_model = P_L(s)/q(s) and
    w_prior = (1/26^L)/q(s), normalised by n = n_is.  Returns
    (S_m, S2_m, S_p, S2_p, n, exact, ess) with per-level dicts keyed by G."""
    levels = sorted(edges_by_level)
    S_m = {G: np.zeros(G * G) for G in levels}
    S2_m = {G: np.zeros(G * G) for G in levels}
    S_p = {G: np.zeros(G * G) for G in levels}
    S2_p = {G: np.zeros(G * G) for G in levels}
    A_L = nm.A ** L
    if L <= 5:
        states = nm.enumerate_states(L)
        for a in range(0, len(states), 4_000_000):
            s = states[a:a + 4_000_000]
            lw = model.log_weight(s)
            w_m = np.exp(lw - model.logZ)
            w_p = np.full(len(s), 1.0 / A_L)
            x, y = nm.coords_of_states(U, s)
            for G in levels:
                b = nm.bin_index_xy(x, y, edges_by_level[G])
                S_m[G] += np.bincount(b, weights=w_m, minlength=G * G)
                S_p[G] += np.bincount(b, weights=w_p, minlength=G * G)
        return S_m, S2_m, S_p, S2_p, A_L, True, None
    sm_sum = 0.0
    s2m_sum = 0.0
    for off in range(0, n_is, 200_000):
        nb = min(200_000, n_is - off)
        s = prop.sample(nb, rng)
        lq = prop.log_q(s)
        w_m = np.exp(model.log_weight(s) - model.logZ - lq)
        w_p = (1.0 / A_L) * np.exp(-lq)
        sm_sum += float(w_m.sum())
        s2m_sum += float((w_m * w_m).sum())
        x, y = nm.coords_of_states(U, s)
        for G in levels:
            b = nm.bin_index_xy(x, y, edges_by_level[G])
            S_m[G] += np.bincount(b, weights=w_m, minlength=G * G)
            S2_m[G] += np.bincount(b, weights=w_m * w_m, minlength=G * G)
            S_p[G] += np.bincount(b, weights=w_p, minlength=G * G)
            S2_p[G] += np.bincount(b, weights=w_p * w_p, minlength=G * G)
    ess = sm_sum ** 2 / s2m_sum if s2m_sum > 0 else 0.0
    return S_m, S2_m, S_p, S2_p, n_is, False, ess


def _normalize_levels(S_m, S2_m, S_p, S2_p, n, exact):
    """Normalise raw per-level sums into (p_bin, p_bin_se, p_prior, p_prior_se)."""
    p_bin, p_bin_se, p_prior, p_prior_se = {}, {}, {}, {}
    for G in S_m:
        if exact:
            p_bin[G] = S_m[G]
            p_bin_se[G] = np.zeros(G * G)
            p_prior[G] = S_p[G]
            p_prior_se[G] = np.zeros(G * G)
        else:
            p_bin[G], p_bin_se[G] = nm.mean_and_se(S_m[G], S2_m[G], n)
            p_prior[G], p_prior_se[G] = nm.mean_and_se(S_p[G], S2_p[G], n)
    return p_bin, p_bin_se, p_prior, p_prior_se


def _frame_props(fr):
    out = {}
    for L in nm.LENGTHS:
        c = Counter({w: n for w, n in fr.items() if len(w) == L})
        st, ct = counter_to_arrays(c, L)
        out[L] = (st, ct)
    return out


def _string_minima(models, pL, fr, U, edges, z_grid, include_plm=True):
    """Strict full-move minima: exact for L<=5 via edits.landscape, found for L>=6.
    Each entry carries the word's projected (x, y) and its cell's height z so the
    overlay can be drawn on the surface."""
    out = []

    def _emit(w, L, E, states):
        x, y = nm.coords_of_states(U, states)
        cell = int(nm.bin_index_xy(x, y, edges)[0])
        z = float(z_grid[cell]) if np.isfinite(z_grid[cell]) else None
        out.append({"word": w, "L": L, "E": float(E),
                    "x": float(x[0]), "y": float(y[0]), "z": z})

    space = edits.Space((1, 2, 3, 4, 5))
    E = np.full(space.size, np.inf)
    for L in (1, 2, 3, 4, 5):
        lw = models[L].dense_log_weights()
        blk = space.block(E, L)
        np.multiply(lw, -1.0 / nm.LN2, out=blk)
        blk -= (np.log2(pL[L - nm.L_MIN]) - models[L].logZ / nm.LN2)
        del lw
    ls = edits.landscape(space, E, edits.MOVES)
    for i in np.flatnonzero(ls.minima):
        w = space.word(int(i))
        L = len(w)
        _emit(w, L, E[i], nm.states_of_words([w]))
    if include_plm:
        for L in (6, 7, 8):
            c = Counter({w: n for w, n in fr.items() if len(w) == L})
            st, ct = counter_to_arrays(c, L)
            if len(st) == 0:
                continue
            term = longwords.descend(models[L]._plm(), st)
            uniq = np.unique(term, axis=0)
            mins = longwords.is_strict_minimum(models[L]._plm(), uniq)
            cand = uniq[mins]
            full = nm.full_move_strict_minima(models, pL, L, cand)
            for s in cand[full]:
                w = nm.words_of(s[None])[0]
                _emit(w, L, nm.word_energy_bits(models, pL, L, s[None])[0], s[None])
    out.sort(key=lambda d: d["E"])
    return out


def _surface_rows(E, digits=3, G=GRID):
    g = E.reshape(G, G)
    return [[round(float(v), digits) if np.isfinite(v) else None for v in row]
            for row in g]


def _cell_top_words_all_levels(models, pL, fr, U, edges_by_level, k, rng,
                               n_is_cand=300_000):
    """Top-k words per cell at every resolution, by grand-canonical P(w)=P(L)P_L(w).

    L <= 5: exhaustive enumeration (ranking exact).  L = 6..8: candidates are the
    observed types plus importance samples; each word's P is exact given the exact
    ln Z, but a high-P word may be missed by the sampler, so those entries carry
    ``approx=True``.  Returns {G: {cell: [{w, L, p, E, approx}]}} sorted by p."""
    import heapq
    levels = sorted(edges_by_level)
    heaps = {G: [[] for _ in range(G * G)] for G in levels}

    def _process(s, P, L, approx):
        x, y = nm.coords_of_states(U, s)
        bl = {G: nm.bin_index_xy(x, y, edges_by_level[G]) for G in levels}
        for i in range(len(s)):
            w = None                                    # created only on acceptance
            Pi = float(P[i])
            for G in levels:
                h = heaps[G][int(bl[G][i])]
                if len(h) < k or Pi > h[0][0]:
                    if w is None:
                        w = nm.words_of(s[i:i + 1])[0]
                    if len(h) < k:
                        heapq.heappush(h, (Pi, w, L, approx))
                    else:
                        heapq.heapreplace(h, (Pi, w, L, approx))

    for L in (1, 2, 3, 4, 5):
        states = nm.enumerate_states(L)
        for a in range(0, len(states), 4_000_000):
            s = states[a:a + 4_000_000]
            lw = models[L].log_weight(s)
            P = pL[L - nm.L_MIN] * np.exp(lw - models[L].logZ)
            _process(s, P, L, False)
    for L in (6, 7, 8):
        c = Counter({w: n for w, n in fr.items() if len(w) == L})
        st, ct = counter_to_arrays(c, L)
        if len(st) == 0:
            continue
        prop = nm.Proposal.from_data(st, ct, L)
        s_is, _ = nm.importance_samples(models[L], prop, n_is_cand, rng)
        cand = np.unique(np.concatenate([st, s_is]), axis=0)
        for a in range(0, len(cand), 200_000):
            s = cand[a:a + 200_000]
            lw = models[L].log_weight(s)
            P = pL[L - nm.L_MIN] * np.exp(lw - models[L].logZ)
            _process(s, P, L, True)

    out = {}
    for G in levels:
        out[G] = {}
        for cell in range(G * G):
            top = sorted(heaps[G][cell], reverse=True)
            out[G][cell] = [{"w": w, "L": L, "p": float(P),
                             "E": float(-np.log2(P)) if P > 0 else float("inf"),
                             "approx": bool(a)} for (P, w, L, a) in top]
    return out


def _minimax_path(z, adj, src, dst):
    """Bottleneck path between ``src`` and ``dst`` on the cell graph: the path that
    minimises the maximum z along it.  Dijkstra with edge distance =
    max(current bottleneck, z[neighbour]); non-finite cells are unreachable.  Returns
    (list of cells, bottleneck z)."""
    import heapq
    n = len(adj)
    zf = np.where(np.isfinite(z), z, np.inf)
    dist = np.full(n, np.inf)
    prev = np.full(n, -1, dtype=np.int64)
    dist[src] = zf[src]
    pq = [(zf[src], int(src))]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == dst:
            break
        for v in adj[u]:
            v = int(v)
            nd = max(d, zf[v])
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if not np.isfinite(dist[dst]):
        return [], None
    path = []
    u = int(dst)
    while True:
        path.append(u)
        if u == src:
            break
        u = int(prev[u])
        if u < 0:
            break
    path.reverse()
    return path, float(dist[dst])


def _build_adjacency(models, U, edges, fr):
    n_cells = GRID * GRID
    adjmat = np.zeros((n_cells, n_cells), dtype=bool)
    rng = np.random.default_rng(SEED)
    for L in nm.LENGTHS:
        c = Counter({w: n for w, n in fr.items() if len(w) == L})
        st, ct = counter_to_arrays(c, L)
        if len(st) == 0:
            continue
        if L <= 5:
            states = nm.enumerate_states(L)
        else:
            prop = nm.Proposal.from_data(st, ct, L)
            states, _ = nm.importance_samples(models[L], prop, 400_000, rng)
        x, y = nm.coords_of_states(U, states)
        b = nm.bin_index_xy(x, y, edges)

        def bin_fn(s):
            xx, yy = nm.coords_of_states(U, s)
            return nm.bin_index_xy(xx, yy, edges)
        nm.move_adjacency_from_states(states, b, L, bin_fn, adjmat)
    return nm.adjacency_list(adjmat)


def build() -> None:
    js = load_json()
    npz = load_npz()
    labels = js["fits"]["labels"]
    sig2 = js["fits"]["sigma2"]
    nf = len(labels)
    log(f"build: {nf} frames in the fixed basis")

    _, frames, per_length = nm.cumulative_frames(MONTHLY, PERIOD)
    final_t = nf - 1
    models_final = {L: _model_from_traj(npz, L, final_t, sig2[str(L)])
                    for L in nm.LENGTHS}
    p_len_final = npz["p_len"][final_t]
    fr_final = frames[-1]
    rng = np.random.default_rng(SEED)

    # ---- basis: importance-sampled bigram covariance under the final model ----
    mean = np.zeros(nm.N_BIGRAMS)
    outer = np.zeros((nm.N_BIGRAMS, nm.N_BIGRAMS))
    wsum = 0.0
    props = {}
    for L in nm.LENGTHS:
        c = Counter({w: n for w, n in fr_final.items() if len(w) == L})
        st, ct = counter_to_arrays(c, L)
        props[L] = nm.Proposal.from_data(st, ct, L)
    for L in nm.LENGTHS:
        s, w = nm.importance_samples(models_final[L], props[L], nm.N_BASIS_PER_LENGTH, rng)
        wsum += nm.accumulate_bigram_moments(s, w * p_len_final[L - nm.L_MIN], mean, outer)
    C = outer / wsum - np.outer(mean / wsum, mean / wsum)
    lam, V = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, V = np.clip(lam[order], 0.0, None), V[:, order]
    U = V[:, :2].T.reshape(2, nm.S, nm.S)

    # grid extent from the final frame's string coordinates (L<=5 exact + samples),
    # then per-level edges (same projection and extent at every resolution)
    xs, ys = [], []
    for L in nm.LENGTHS:
        if L <= 5:
            sall = nm.enumerate_states(L)
        else:
            sall, _ = nm.importance_samples(models_final[L], props[L], 200_000, rng)
        x, y = nm.coords_of_states(U, sall)
        xs.append(x)
        ys.append(y)
    xr = np.concatenate([np.asarray(c).ravel() for c in xs])
    yr = np.concatenate([np.asarray(c).ravel() for c in ys])
    edges_by_level = {G: np.stack([np.linspace(xr.min(), xr.max(), G + 1),
                                   np.linspace(yr.min(), yr.max(), G + 1)])
                      for G in LEVELS}
    edges = edges_by_level[GRID]
    centres = nm.cell_centres(edges)

    axis_report = [nm.axis_report(U, lam, k) for k in range(2)]
    basis_meta = {"axes": axis_report,
                  "cumulative_share": float(lam[:2].sum() / lam.sum()),
                  "total_variance": float(lam.sum()),
                  "x": centres[0].round(5).tolist(),
                  "y": centres[1].round(5).tolist(),
                  "cell": [float(np.diff(e)[0]) for e in edges]}

    # cell adjacency (fixed) and per-cell top words (for well labels, ranked by P)
    log("  building move-set cell adjacency ...")
    adj = _build_adjacency(models_final, U, edges, fr_final)
    log("  computing per-cell top words at all resolutions ...")
    cell_words_all = _cell_top_words_all_levels(models_final, p_len_final, fr_final, U,
                                                edges_by_level, 5, rng)
    cell_top_words_32 = cell_words_all[GRID]
    centres = nm.cell_centres(edges)

    z_all, z_se_all, mask_all = [], [], []
    wells_all, barrier_all, ent_all, n_min_all = [], [], [], []
    traj_levels = {G: {"z": [], "z_se": [], "mask": []} for G in LEVELS}
    final_res_by_G = None
    for t in range(nf):
        models = {L: _model_from_traj(npz, L, t, sig2[str(L)]) for L in nm.LENGTHS}
        pL = npz["p_len"][t]
        fr = frames[t]
        props_t = _frame_props(fr)
        p_bin = {G: {} for G in LEVELS}
        p_se = {G: {} for G in LEVELS}
        p_prior = {G: {} for G in LEVELS}
        p_prior_se = {G: {} for G in LEVELS}
        ess_by_L = {}
        for L in nm.LENGTHS:
            st, ct = props_t[L]
            ess = None
            if L <= 5:
                S_m, S2_m, S_p, S2_p, n, exact, ess = _length_mass_all_levels(
                    models[L], None, U, edges_by_level, L, 0, rng)
                pb, pbs, pp, pps = _normalize_levels(S_m, S2_m, S_p, S2_p, n, exact)
            elif len(st) == 0:
                # uniform model (V = 0) = the prior: P(bin|L) = P_prior = |bin|/26^L,
                # estimated by uniform sampling (26^L cannot be enumerated for L >= 6).
                us = nm.uniform_samples(L, nm.N_UNIFORM_PER_LENGTH, rng)
                xu, yu = nm.coords_of_states(U, us)
                pb, pbs, pp, pps = {}, {}, {}, {}
                for G in LEVELS:
                    bu = nm.bin_index_xy(xu, yu, edges_by_level[G])
                    f = np.bincount(bu, minlength=G * G) / len(us)
                    pb[G] = f
                    pbs[G] = np.sqrt(np.maximum(f * (1 - f), 0.0) / len(us))
                    pp[G] = f
                    pps[G] = pbs[G]
            else:
                prop = nm.Proposal.from_data(st, ct, L)
                S_m, S2_m, S_p, S2_p, n, exact, ess = _length_mass_all_levels(
                    models[L], prop, U, edges_by_level, L, nm.N_IS_PER_LENGTH, rng)
                pb, pbs, pp, pps = _normalize_levels(S_m, S2_m, S_p, S2_p, n, exact)
            for G in LEVELS:
                p_bin[G][L], p_se[G][L] = pb[G], pbs[G]
                p_prior[G][L], p_prior_se[G][L] = pp[G], pps[G]
            if t == final_t:
                ess_by_L[L] = ess
        res_by_G = {G: nm.combine_bin_energy(pL, p_bin[G], p_se[G], p_prior[G],
                                             p_prior_se[G], ess_by_L, G * G)
                    for G in LEVELS}
        if t == final_t:
            final_res = res_by_G[GRID]
            final_res_by_G = res_by_G
        # store the 32-level results (backward compat) and per-level z/z_se/mask
        res = res_by_G[GRID]
        z = np.where(np.isposinf(res["z_bin"]), Z_CAP, res["z_bin"])
        z_all.append(z)
        z_se_all.append(res["z_se"])
        valid = np.isfinite(res["z_bin"])
        confident = valid & (res["z_se"] <= 0.5) & ~res["low_ess"] & ~res["one_sided"]
        m = confident | ~valid                       # gaps are empty, not low-confidence
        mask_all.append(m)
        for G in LEVELS:
            rG = res_by_G[G]
            zG = np.where(np.isposinf(rG["z_bin"]), Z_CAP, rG["z_bin"])
            validG = np.isfinite(rG["z_bin"])
            confG = validG & (rG["z_se"] <= 0.5) & ~rG["low_ess"] & ~rG["one_sided"]
            traj_levels[G]["z"].append(zG)
            traj_levels[G]["z_se"].append(rG["z_se"])
            traj_levels[G]["mask"].append(confG | ~validG)
        wells, barrier, lab = nm.projected_wells(z, adj, PERSIST_BITS,
                                                 cell_top_words_32, centres,
                                                 res["P_bin"])
        # per-well confidence: fraction of the basin's model mass in confident cells
        lab_flat = lab.ravel()
        for w in wells:
            basin = lab_flat == w["cell"]
            conf = float(res["P_bin"][basin & confident].sum())
            w["confident"] = bool(w["basin_mass"] > 0 and conf / w["basin_mass"] >= 0.5)
        wells_all.append(wells)
        barrier_all.append(barrier)
        ent = sum(pL[L - nm.L_MIN] * models[L].entropy_bits for L in nm.LENGTHS)
        ent += float(-np.sum(pL * np.log2(pL)))
        ent_all.append(ent)
        n_min_all.append(int(_count_l5_minima(models, pL)))
        log(f"  frame {t} ({labels[t]}): N={int(npz['tokens'][t]):,}  z in "
            f"[{np.nanmin(res['z_bin']):.2f}, {np.nanmax(res['z_bin']):.2f}]  "
            f"max z_se {res['z_se'].max():.3f} bits  wells {len(wells)}  minima {n_min_all[-1]}")

    z_all = np.array(z_all)
    z_se_all = np.array(z_se_all)
    mask_all = np.array(mask_all)
    # Quantise the trajectory surface.  The -32768 sentinel marks a gap (no string);
    # it must survive the clip, so only the finite (rounded) values are clipped.
    q = nm.quantize_surface(z_all, CENTIBITS)
    q_levels = {G: nm.quantize_surface(np.array(traj_levels[G]["z"]), CENTIBITS)
                for G in LEVELS}

    # final-frame string minima (all lengths) and well metadata
    string_minima_all = _string_minima(models_final, p_len_final, fr_final, U, edges,
                                       z_all[final_t], include_plm=True)
    log(f"  final string minima: {len(string_minima_all)} (top shown)")
    string_minima = string_minima_all[:nm.TOP_STRING_MINIMA]

    cell_words_grid = [cell_words_all[GRID][c] for c in range(GRID * GRID)]
    # global string table, so the per-level cell_words store word indices, not strings
    vocab = sorted({e["w"] for G in LEVELS for lst in cell_words_all[G].values()
                    for e in lst})
    w2i = {w: i for i, w in enumerate(vocab)}

    # per-level grid-adjacency wells (move-set adjacency is too expensive per level),
    # plus per-level confidence summary
    levels_out = {}
    minimax_32 = []
    for G in LEVELS:
        rG = final_res_by_G[G]
        zG = np.where(np.isposinf(rG["z_bin"]), Z_CAP, rG["z_bin"])
        validG = np.isfinite(rG["z_bin"])
        confG = validG & (rG["z_se"] <= 0.5) & ~rG["low_ess"] & ~rG["one_sided"]
        maskG = confG | ~validG
        se_fin = rG["z_se"][np.isfinite(rG["z_se"])]
        centresG = nm.cell_centres(edges_by_level[G])
        adjG = adj if G == GRID else pj.grid_adjacency((G, G))
        if G == GRID:
            # reuse the static tab's move-set-adjacency wells (identical objects)
            wellsG = wells_all[final_t]
            barrierG = barrier_all[final_t]
            wells_adj = "move_set"
        else:
            wellsG, barrierG, labG = nm.projected_wells(zG, adjG, PERSIST_BITS,
                                                        cell_words_all[G], centresG,
                                                        rG["P_bin"])
            labG_flat = labG.ravel()
            for w in wellsG:
                basin = labG_flat == w["cell"]
                conf = float(rG["P_bin"][basin & confG].sum())
                w["confident"] = bool(w["basin_mass"] > 0 and conf / w["basin_mass"] >= 0.5)
            wells_adj = "grid8"
        # minimax (bottleneck) path between the barrier's two wells, on this level's
        # adjacency; the saddle cell is the path's maximum-z cell.
        barrier_path = None
        if barrierG is not None:
            c1, c2 = barrierG["wells"]
            path, bottleneck = _minimax_path(zG, adjG, c1, c2)
            rec = []
            for c in path:
                top = cell_words_all[G].get(int(c), [])
                rec.append({"cell": int(c),
                            "x": float(centresG[0][int(c) % G]),
                            "y": float(centresG[1][int(c) // G]),
                            "z": float(zG[int(c)]),
                            "word": top[0]["w"] if top else None})
            barrier_path = {"cells": rec,
                            "bottleneck": (float(bottleneck) if bottleneck is not None
                                           else None),
                            "saddle": (int(rec[np.argmax([r["z"] for r in rec])]["cell"])
                                       if rec else None)}
        levels_out[G] = {
            "z": _surface_rows(zG, G=G),
            "z_se": _surface_rows(rG["z_se"], G=G),
            "mask": maskG.astype(int).reshape(G, G).tolist(),
            # [word_index, L, p] triplets into cell_words_vocab
            "cell_words": [[[w2i[e["w"]], e["L"], e["p"]]
                            for e in cell_words_all[G].get(c, [])]
                           for c in range(G * G)],
            "wells": wellsG, "barrier": barrierG,
            "barrier_path": barrier_path,
            "wells_adjacency": wells_adj,
            "n_gaps": int((~validG).sum()),
            "n_low_confidence": int((maskG == 0).sum()),
            "median_se": float(np.median(se_fin)) if se_fin.size else None,
        }
        log(f"  level G={G}: gaps {levels_out[G]['n_gaps']} low-conf "
            f"{levels_out[G]['n_low_confidence']} median_SE "
            f"{levels_out[G]['median_se']} wells {len(wellsG)}")

    # top-5 wells (by basin mass) pairwise minimax paths at the reference level 32
    z32 = z_all[final_t]
    centres32 = centres
    wells32 = sorted(wells_all[final_t], key=lambda w: -w.get("basin_mass", 0.0))[:5]
    for i in range(len(wells32)):
        for j in range(i + 1, len(wells32)):
            c1, c2 = wells32[i]["cell"], wells32[j]["cell"]
            path, bottleneck = _minimax_path(z32, adj, c1, c2)
            rec = []
            for c in path:
                top = cell_words_all[GRID].get(int(c), [])
                rec.append({"cell": int(c),
                            "x": float(centres32[0][int(c) % GRID]),
                            "y": float(centres32[1][int(c) // GRID]),
                            "z": float(z32[int(c)]),
                            "word": top[0]["w"] if top else None})
            minimax_32.append({"pair": [int(c1), int(c2)],
                               "cells": rec,
                               "bottleneck": (float(bottleneck) if bottleneck is not None
                                              else None)})

    # trajectory hover uses the final-frame ranking (per-frame L<=5 enumeration would
    # add ~19 min; words evolve slowly), stored compactly with the same string table.
    traj_cell_words = [[{"w": w2i[e["w"]], "L": e["L"], "p": e["p"]}
                        for e in cell_words_all[GRID][c][:3]] for c in range(GRID * GRID)]

    counts_final = final_res["P_bin"]
    out = {
        "model": "new-model grand-canonical landscape (variable length L=1..8)",
        "models": {"final": {
            "label": "New model (variable length, grand-canonical), Twitter 2006-2023",
            "tokens": float(npz["tokens"][final_t]),
            "S2": ent_all[final_t],
            "z": _surface_rows(z_all[final_t]),
            "z_se": _surface_rows(z_se_all[final_t]),
            "mask": mask_all[final_t].astype(int).reshape(GRID, GRID).tolist(),
            "P_bin": _surface_rows(final_res["P_bin"], 6),
            "P_prior": _surface_rows(final_res["P_prior"], 6),
            "cell_words": cell_words_grid,
            "wells": wells_all[final_t], "barrier": barrier_all[final_t],
            "n_string_minima": len(string_minima_all),
            "string_minima": string_minima,
            "n_wells_all": len(wells_all[final_t]),
            "n_wells_confident": int(sum(w["confident"] for w in wells_all[final_t])),
            "ess_by_L": {str(L): (ess_by_L.get(L) if ess_by_L.get(L) is not None
                                 else None) for L in nm.LENGTHS},
            "n_low_confidence": int((mask_all[final_t] == 0).sum()),
            "n_low_ess": int(final_res["low_ess"].sum()),
            "n_gaps": int((~np.isfinite(z_all[final_t])).sum()),
            **basis_meta}},
        "traj": {"months": labels, "tokens": npz["tokens"].tolist(),
                 "period": PERIOD, "sigma2": sig2,
                 "S2": ent_all,
                 "n_string_minima": n_min_all,
                 "n_string_minima_scope": "L<=5 exact (L>=6 minima found only for the final frame)",
                 "per_length_tokens": {str(L): npz[f"tokens_L{L}"].tolist()
                                       for L in nm.LENGTHS},
                 "cell_words_vocab": vocab,
                 "cell_words": traj_cell_words,
                 "cell_words_scope": "final-frame ranking only (per-frame L<=5 "
                                     "enumeration would add ~19 min; words evolve slowly; "
                                     "L>=6 entries are approximate)",
                 "zb64": base64.b64encode(q.tobytes()).decode(),
                 "z_se_max": [float(np.nanmax(r[np.isfinite(r)]))
                              if np.isfinite(r).any() else None
                              for r in z_se_all],
                 "n_confident": [int(np.sum(m)) for m in mask_all],
                 "scale": CENTIBITS,
                 "zmin": float(q[q > -32768].min()) / CENTIBITS,
                 "zmax": float(q[q > -32768].max()) / CENTIBITS,
                 "x": centres[0].round(5).tolist(),
                 "y": centres[1].round(5).tolist(),
                 "wells": wells_all, "barrier": barrier_all},
        "grid": GRID, "persist_bits": PERSIST_BITS,
        "basis": basis_meta,
        "levels_order": list(LEVELS),
        "levels": {str(G): levels_out[G] for G in LEVELS},
        "cell_words_vocab": vocab,
        "minimax_paths": minimax_32,
        "min_basin_mass_default": 1e-3,
        "prior": js.get("prior"), "measure": js.get("measure"),
    }

    js["summary"] = out
    save_json(js)
    npz2 = load_npz()
    npz2["z_traj"] = q
    npz2["z_se_traj"] = z_se_all.astype(np.float32)
    npz2["mask_traj"] = mask_all.astype(np.int8)
    npz2["entropy_traj"] = np.array(ent_all)
    npz2["basis_U"] = U
    for G in LEVELS:
        npz2[f"z_traj_L{G}"] = q_levels[G]
        npz2[f"z_se_traj_L{G}"] = np.array(traj_levels[G]["z_se"]).astype(np.float16)
        npz2[f"mask_traj_L{G}"] = np.array(traj_levels[G]["mask"]).astype(np.int8)
    save_npz(npz2)
    log(f"wrote {CACHE_NPZ.name} and {CACHE_JSON.name}")


def _count_l5_minima(models, pL):
    space = edits.Space((1, 2, 3, 4, 5))
    E = np.full(space.size, np.inf)
    for L in (1, 2, 3, 4, 5):
        lw = models[L].dense_log_weights()
        blk = space.block(E, L)
        np.multiply(lw, -1.0 / nm.LN2, out=blk)
        blk -= (np.log2(pL[L - nm.L_MIN]) - models[L].logZ / nm.LN2)
        del lw
    ls = edits.landscape(space, E, edits.MOVES)
    return int(np.count_nonzero(ls.minima))


STAGES = {"prior": stage_prior, "measure": stage_measure, "fits": stage_fits,
          "build": build}


def main(argv) -> None:
    for s in argv or ["prior", "measure", "fits", "build"]:
        if s == "data":
            labels, frames, per_length = nm.cumulative_frames(MONTHLY, PERIOD)
            log(f"data: {len(frames)} frames ({PERIOD}), final N={sum(frames[-1].values()):,}")
            continue
        log(f"stage {s}")
        STAGES[s]()


if __name__ == "__main__":
    main(sys.argv[1:])
