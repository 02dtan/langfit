"""Pairwise versus three-body maximum-entropy models of four-letter words.

For L = 4 the maximum-entropy hierarchy has four levels (Schneidman, Still, Berry &
Bialek, PRL 91, 238701, 2003; Amari, IEEE Trans. Inf. Theory 47, 1701, 2001):

    S_1 >= S_2 >= S_3 >= S_4 = S_full,

with P_k the maximum-entropy law matching all k-letter marginals. The k-th connected
information is I^(k) = S_{k-1} - S_k = D(P_k || P_{k-1}) and the multi-information is
I_N = S_1 - S_full = sum_{k>=2} I^(k). This script measures I^(3), the information
carried by three-body (triplet-clique) interactions beyond all pairwise ones, on every
corpus of the study, and asks whether it is a finite-sample artefact or generalises.

Per corpus:
  curves     S_1, S_2, S_3, S_full (exact mode, in-sample) on a half-decade grid of
             sample sizes N anchored at the common size of the four-letter study
             (the Austen token count), from two decades below it to the full corpus;
             REPS hypergeometric subsamples per grid point below the full corpus.
  triplets   at the common N and at the full N: for each of the 4 triplet cliques c,
             D(mu_c || P_2|c) (the KL between the empirical triplet marginal and the
             pairwise model's prediction of it), the entropy drop S_2 - S_{2+c} when
             only clique c is added to all pairs (Della Pietra et al., IEEE PAMI 19,
             380, 1997: the gain of inducing one feature), and the letter triples that
             carry the divergence.
  landscape  exact-mode energy landscapes of P_2 and P_3 at the same N: strict local
             minima under single-letter substitution, minima gained / lost, energy
             ranks of probe words (acronyms, slang, controls, typos).
  held-out   nested cross-validation (map mode, Gaussian prior chosen by inner CV) of
             the code lengths H_1, H_2, H_3, and the held-out gain of each single
             triplet clique (pairs + c, at the order-3 prior width chosen on the same
             outer training fold), on a subset of corpora (see CV_* below).
"""

from __future__ import annotations

import gc
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
sys.path.insert(0, str(ROOT / "scripts"))

from langfit import corpus, maxent as me, potts, social  # noqa: E402
import maxent_study as ms  # noqa: E402  (frozen; reused for corpora and probes)

L = 4
REPS = ms.REPS                      # subsamples per grid point up to the common N, as in
                                    # the four-letter study
REPS_LARGE = 1                      # above it one subsample: the sampling SD of the
                                    # entropies falls as N^-1/2 (measured at N <= N_common)
                                    # while fit cost grows, on a machine shared at 1 worker
OUT = corpus.DATA / "derived" / "ternary_study.json"
CACHE = corpus.DATA / "derived" / "ternary_study.jobs.jsonl"   # per-job results, resumable
TWITTER = os.environ.get("TERNARY_TWITTER", "twitter_year_L1-20_all.pkl.gz")
LN2 = np.log(2.0)
OUTER, INNER = ms.OUTER, ms.INNER   # nested-CV folds, as in the four-letter study
SEED = ms.SEED
WORKERS = int(os.environ.get("TERNARY_WORKERS", "3"))   # shared 8-core machine
GRID_STEP = 10.0 ** 0.5             # half-decade grid in N (same convention as sigma2)
GRID_LO = -4                        # lowest grid point: two decades below the common N
TOP = 30                            # length of reported word / cell lists (display only)

# Probe forms fixed before any fit: the four-letter study's probes plus acronyms whose
# letter triples are atypical of English spelling.
ACRONYMS = ["tmrw", "tldr", "html", "lmao", "asap", "nsfw", "imho", "omfg", "rofl", "smth"]
PROBES = list(dict.fromkeys(ACRONYMS + [w for g in ms.PROBES.values() for w in g]))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def finite(x):
    return float(x) if np.isfinite(x) else None


# --------------------------------------------------------------------------- #
# corpora
# --------------------------------------------------------------------------- #

def load_corpora() -> dict[str, dict]:
    """Austen, Gutenberg and Bluesky 2024 exactly as in the four-letter study; Twitter
    years and pooled Twitter from the all-shard yearly ingest."""
    base = ms.load_corpora()
    out = {k: base[k] for k in ("Austen", "Gutenberg", "Bluesky")}
    tw = social.load_periods(TWITTER)
    pooled = Counter()
    for y, rec in sorted(tw["periods"].items()):
        c = Counter(rec["counts"].get(L, {}))
        pooled.update(c)
        out[f"Twitter {y}"] = {"counter": c,
                               "meta": {k: v for k, v in rec.items() if k != "counts"}}
    out["Twitter"] = {"counter": pooled,
                      "meta": {"source": f"enryu43/twitter100m_tweets, {len(tw['files'])} "
                                         f"shards ({TWITTER})"}}
    del tw, base            # the all-length pickle is large; keep only the L = 4 counts
    gc.collect()
    return out


def grid(n_common: int, n_total: int) -> list[int]:
    """Half-decade grid anchored at the common N, up to (and including) the corpus."""
    out, k = [], GRID_LO
    while True:
        n = int(round(n_common * GRID_STEP ** k))
        if n >= n_total:
            break
        out.append(n)
        k += 1
    return out + [int(n_total)]


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #

def label(c, cell) -> str:
    """Letter triple on clique c as a 4-letter pattern, e.g. (0,1,3),(t,h,e) -> 'th_e'."""
    s = ["_"] * L
    for ax, k in zip(c, cell):
        s[ax] = me.ALPHABET[k]
    return "".join(s)


def triplet_divergences(m2: potts.Potts, mus3) -> list[dict]:
    """Per triplet clique: D(mu_c || P_2|c) in bits and the cells that carry it."""
    cl3 = potts.cliques(3, L)
    out = []
    for c, mu, q in zip(cl3, mus3, m2.marginals(cl3)):
        m = mu > 0
        contrib = np.zeros_like(mu)
        contrib[m] = mu[m] * np.log2(mu[m] / q[m])
        order = np.argsort(-contrib, axis=None)
        lr = np.full(mu.shape, np.inf)
        lr[m] = np.log2(mu[m] / q[m])
        under = np.argsort(lr, axis=None)
        # mass the pairwise model puts on triples never observed (forbidden in P_3)
        out.append({
            "clique": list(c), "kl_bits": float(contrib.sum()),
            "p2_mass_on_unseen_triples": float(q[~m].sum()),
            "top_cells": [(label(c, np.unravel_index(i, mu.shape)), float(mu.flat[i]),
                           float(q.flat[i]), float(contrib.flat[i])) for i in order[:TOP]],
            "most_overpredicted_seen": [
                (label(c, np.unravel_index(i, mu.shape)), float(mu.flat[i]), float(q.flat[i]))
                for i in under[:TOP] if m.flat[i]],
        })
    return out


def landscape(m2: potts.Potts, m3: potts.Potts, st, ct) -> dict:
    """Exact-mode energy landscapes of P_2 and P_3 under single-letter substitution."""
    flat = potts.flat_index(st, L)
    w = ct / ct.sum()
    seen = np.zeros(me.A ** L, dtype=bool)
    seen[flat] = True
    res = {}
    E, P, mins, rank = {}, {}, {}, {}
    for k, m in (("2", m2), ("3", m3)):
        Eb = m.energy_bits()
        E[k] = Eb.ravel()
        P[k] = np.exp2(-E[k])
        mins[k] = me.local_minima(Eb).ravel()
        r = np.empty(E[k].size, dtype=np.int64)
        r[np.argsort(E[k], kind="stable")] = np.arange(1, E[k].size + 1)
        rank[k] = r
        res[f"n_minima_{k}"] = int(mins[k].sum())
        res[f"n_minima_seen_{k}"] = int((mins[k] & seen).sum())
        res[f"minima_token_share_{k}"] = float(w[mins[k][flat]].sum())
        res[f"unseen_model_mass_{k}"] = float(P[k][~seen].sum())
        res[f"n_allowed_{k}"] = int(np.isfinite(E[k]).sum())
        # residual of usage against the model, token-weighted rms over seen words (bits)
        res[f"rms_residual_bits_{k}"] = float(np.sqrt((w * (np.log2(w) - np.log2(P[k][flat])) ** 2).sum()))
    both = mins["2"] & mins["3"]
    gained = np.flatnonzero(mins["3"] & ~mins["2"])
    lost = np.flatnonzero(mins["2"] & ~mins["3"])
    res["n_minima_both"] = int(both.sum())
    res["n_minima_gained"] = int(gained.size)
    res["n_minima_lost"] = int(lost.size)

    def rows(idx, key):
        idx = idx[np.argsort(-P[key][idx])][:TOP]
        return [(potts.unravel(i, L), float(P["3"][i]), float(P["2"][i]), bool(seen[i]))
                for i in idx]

    res["gained_minima"] = rows(gained, "3")
    res["lost_minima"] = rows(lost, "2")
    res["top_minima_3"] = rows(np.flatnonzero(mins["3"]), "3")
    res["probes"] = {}
    for f in PROBES:
        i = potts.ravel(f)
        hit = np.flatnonzero(flat == i)
        res["probes"][f] = {
            "count": float(ct[hit].sum()) if hit.size else 0.0,
            **{f"energy_bits_{k}": finite(E[k][i]) for k in ("2", "3")},
            # rank among all A^L strings by energy; None for forbidden (infinite) energy
            **{f"rank_{k}": int(rank[k][i]) if np.isfinite(E[k][i]) else None
               for k in ("2", "3")},
            **{f"is_min_{k}": bool(mins[k][i]) for k in ("2", "3")}}
    return res


def job_curve(args) -> dict:
    """In-sample hierarchy at one grid point, one subsample."""
    name, st_all, ct_all, N, seed, rep, detail = args
    st, ct = potts.subsample(st_all, ct_all, N, seed)
    t0 = time.time()
    m1, d1 = potts.fit(st, ct, L, 1, "exact")
    m2, d2 = potts.fit(st, ct, L, 2, "exact")
    m3, d3 = potts.fit(st, ct, L, 3, "exact", init=m2)
    p = ct / ct.sum()
    out = {"name": name, "N": int(N), "rep": rep, "types": int(ct.size),
           "S1": m1.entropy_bits(), "S2": m2.entropy_bits(), "S3": m3.entropy_bits(),
           "S_full": float(-(p * np.log2(p)).sum()),
           "S_full_bias": float((ct.size - 1) / (2 * N * LN2)),   # Miller (1955) scale
           "converged": bool(d1["converged"] and d2["converged"] and d3["converged"]),
           "fit": {k: {"marg_in_se": d["max_gradient_in_se"],
                       "entropy_err_in_se": d["entropy_error_in_se"],
                       "params": d.get("n_params")} for k, d in (("2", d2), ("3", d3))}}
    mus3 = potts.marginals(potts.empirical(st, ct, L), potts.cliques(3, L))
    tri = triplet_divergences(m2, mus3)
    out["kl_clique_bits"] = [t["kl_bits"] for t in tri]
    if detail:
        cl2 = potts.cliques(2, L)
        for t, c in zip(tri, potts.cliques(3, L)):
            mc, dc = potts.fit(st, ct, L, 0, "exact", init=m2, cliques_=cl2 + [c])
            t["S_2plus"] = mc.entropy_bits()
            t["drop_bits"] = out["S2"] - t["S_2plus"]
            t["converged"] = dc["converged"]
        out["triplets"] = tri
        out["landscape"] = landscape(m2, m3, st, ct)
    out["seconds"] = time.time() - t0
    return out


def job_cv(args) -> dict:
    """One outer fold of nested CV at one order; order 3 also scores pairs + one clique."""
    name, st_all, ct_all, N, sub_seed, order, outer = args
    t0 = time.time()
    st, ct = potts.subsample(st_all, ct_all, N, sub_seed)
    r = potts.nested_cv_fold(st, ct, L, order, outer, OUTER, INNER, SEED)
    out = {"name": name, "N": int(N), "order": order, **{k: v for k, v in r.items()}}
    if order == 3:
        test = potts.split(ct, OUTER, np.random.default_rng(SEED))[outer]
        train = ct - test
        cl2 = potts.cliques(2, L)
        out["single_clique_bits"] = {}
        for c in potts.cliques(3, L):
            m, d = potts.fit(st, train, L, 0, "map", sigma2=r["sigma2"], cliques_=cl2 + [c])
            out["single_clique_bits"]["".join(map(str, c))] = m.cross_entropy_bits(st, test)
    out["seconds"] = time.time() - t0
    return out


def job_map_landscape(args) -> dict:
    """Regularised (map) order-2 and order-3 models on the whole CV subsample, each at
    the median of the prior widths its nested CV selected: every string has a finite
    energy, so ranks of unseen probes (acronyms) are defined."""
    name, st_all, ct_all, N, sub_seed, s2_2, s2_3 = args
    t0 = time.time()
    st, ct = potts.subsample(st_all, ct_all, N, sub_seed)
    m2, d2 = potts.fit(st, ct, L, 2, "map", sigma2=s2_2)
    m3, d3 = potts.fit(st, ct, L, 3, "map", sigma2=s2_3, init=m2)
    out = {"name": name, "N": int(N), "sigma2_2": s2_2, "sigma2_3": s2_3,
           "S2": m2.entropy_bits(), "S3": m3.entropy_bits(),
           "converged": bool(d2["converged"] and d3["converged"]),
           "landscape": landscape(m2, m3, st, ct), "seconds": time.time() - t0}
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

# A worker killed from outside (the memory manager of the shared machine) breaks the
# pool; the pool is recreated this many times, resuming from the cache. Operational.
POOL_RESTARTS = 5


def run(jobs: list[tuple[str, callable, tuple]]) -> dict[str, dict]:
    """Run (key, fn, args) jobs on WORKERS processes, caching each result in CACHE."""
    from concurrent.futures import as_completed
    from concurrent.futures.process import BrokenProcessPool

    done = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            rec = json.loads(line)
            done[rec["key"]] = rec["result"]
    for attempt in range(POOL_RESTARTS + 1):
        todo = [j for j in jobs if j[0] not in done]
        log(f"{len(jobs)} jobs, {len(jobs) - len(todo)} cached, {len(todo)} to run")
        try:
            with ProcessPoolExecutor(WORKERS) as ex, CACHE.open("a") as fh:
                futs = {ex.submit(fn, args): key for key, fn, args in todo}
                for i, f in enumerate(as_completed(futs), 1):
                    key = futs[f]
                    done[key] = f.result()
                    fh.write(json.dumps({"key": key, "result": done[key]}) + "\n")
                    fh.flush()
                    log(f"  [{i}/{len(todo)}] {key} ({done[key].get('seconds', 0):.0f}s)")
            return done
        except BrokenProcessPool:
            log(f"worker died (attempt {attempt + 1}); restarting the pool")
    raise RuntimeError("process pool broke repeatedly")


def summarise_curves(rows: list[dict]) -> dict:
    """Mean and SE over subsamples, per corpus and grid point."""
    out: dict = {}
    for r in rows:
        out.setdefault(r["name"], {}).setdefault(r["N"], []).append(r)
    res = {}
    for name, by_n in out.items():
        pts = []
        for N in sorted(by_n):
            rs = by_n[N]
            pt = {"N": N, "reps": len(rs), "converged": all(r["converged"] for r in rs)}
            for key, f in (("S1", lambda r: r["S1"]), ("S2", lambda r: r["S2"]),
                           ("S3", lambda r: r["S3"]), ("S_full", lambda r: r["S_full"]),
                           ("S_full_bias", lambda r: r["S_full_bias"]),
                           ("types", lambda r: r["types"]),
                           ("I2", lambda r: r["S1"] - r["S2"]),
                           ("I3", lambda r: r["S2"] - r["S3"]),
                           ("I4", lambda r: r["S3"] - r["S_full"]),
                           ("I_multi", lambda r: r["S1"] - r["S_full"]),
                           ("I3_share", lambda r: (r["S2"] - r["S3"]) / (r["S1"] - r["S_full"])),
                           ("I3_over_I2", lambda r: (r["S2"] - r["S3"]) / (r["S1"] - r["S2"]))):
                v = np.array([f(r) for r in rs], dtype=float)
                pt[key] = float(v.mean())
                pt[key + "_se"] = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else None
            kl = np.array([r["kl_clique_bits"] for r in rs])
            pt["kl_clique_bits"] = kl.mean(axis=0).tolist()
            pts.append(pt)
        res[name] = pts
    return res


def main() -> None:
    t0 = time.time()
    cs = load_corpora()
    n_common = int(sum(cs["Austen"]["counter"].values()))
    tokens = {k: int(sum(v["counter"].values())) for k, v in cs.items()}
    years = sorted(k for k in cs if k.startswith("Twitter ") and tokens[k] >= n_common)
    names = ["Austen", "Gutenberg", "Bluesky", "Twitter"] + years
    arrays = {k: potts.counts_to_arrays(cs[k]["counter"], L) for k in names}
    seed_of = {k: SEED + 1000 * i for i, k in enumerate(names)}
    meta = {"L": L, "N_common": n_common, "reps": REPS, "outer_folds": OUTER,
            "inner_folds": INNER, "seed": SEED, "grid_step": GRID_STEP, "grid_lo": GRID_LO,
            "probes": PROBES, "twitter_file": TWITTER, "workers": WORKERS,
            "sigma2_start": list(potts.SIGMA2_START),
            "corpora": {k: {"tokens": tokens[k], "types": len(cs[k]["counter"]),
                            **{m: v for m, v in cs[k]["meta"].items()}}
                        for k in cs},
            "eligible": names,
            "excluded_years": sorted(k for k in cs if k.startswith("Twitter ") and k not in years),
            "params_gauge_fixed": {str(k): int(sum(
                len(potts.cliques(j, L)) * (me.A - 1) ** j for j in range(1, k + 1)))
                for k in range(1, L + 1)}}
    del cs                  # from here on only the (states, counts) arrays are needed
    gc.collect()

    # CV design: the corpora of the four-letter study's CV (Austen, Gutenberg, first and
    # last eligible year, Bluesky) plus pooled Twitter, at the common N; and pooled
    # Twitter one decade above it, to see whether the held-out gain of the triplet
    # terms grows with data.
    cv_names = ["Austen", "Gutenberg", years[0], years[-1], "Bluesky", "Twitter"]
    big = ["Twitter"]
    n_big = [int(round(n_common * GRID_STEP ** 2))]
    meta["cv"] = {"common": cv_names, "big": big, "N_big": n_big}
    meta["curve_reps"] = {"N <= N_common": REPS, "N > N_common": REPS_LARGE,
                          "N = corpus": 1}

    jobs = []
    for k in names:
        for N in grid(n_common, tokens[k]):
            reps = 1 if N >= tokens[k] else (REPS if N <= n_common else REPS_LARGE)
            for rep in range(reps):
                detail = rep == 0 and (N == n_common or N >= tokens[k])
                jobs.append((f"curve|{k}|{N}|{rep}", job_curve,
                             (k, *arrays[k], N, seed_of[k] + rep, rep, detail)))
    for order in (1, 2, 3):
        for k in cv_names:
            for f in range(OUTER):
                jobs.append((f"cv|{k}|{n_common}|{order}|{f}", job_cv,
                             (k, *arrays[k], n_common, seed_of[k], order, f)))
    for N in n_big:
        for order in (1, 2, 3):
            for k in big:
                for f in range(OUTER):
                    jobs.append((f"cv|{k}|{N}|{order}|{f}", job_cv,
                                 (k, *arrays[k], N, seed_of[k], order, f)))

    # Order of execution, so that an interrupted run leaves complete layers: curves up
    # to the common N, CV at the common N, larger curves by increasing N, large-N CV.
    def phase(j):
        kind, N = j[0].split("|")[0], int(j[0].split("|")[2])
        if kind == "curve":
            return (0, 0) if N <= n_common else (2, N)
        return (1, 0) if N <= n_common else (3, N)
    jobs.sort(key=phase)
    done = run(jobs)
    wanted = {j[0] for j in jobs}
    done = {k: v for k, v in done.items() if k in wanted}

    curve_rows = [v for key, v in done.items() if key.startswith("curve|")]
    res = {"meta": meta, "curves": summarise_curves(curve_rows), "detail": {}}
    for r in curve_rows:
        if "triplets" in r:
            tag = "full" if r["N"] >= tokens[r["name"]] else "common"
            res["detail"].setdefault(r["name"], {})[tag] = {
                k: r[k] for k in ("N", "types", "S1", "S2", "S3", "S_full", "converged",
                                  "triplets", "landscape", "fit")}
    cv: dict = {}
    for key, r in done.items():
        if not key.startswith("cv|"):
            continue
        _, k, N, order, f = key.split("|")
        cv.setdefault(k, {}).setdefault(N, {}).setdefault(order, []).append(r)
    res["cv"] = {}
    for k, by_n in cv.items():
        for N, by_o in by_n.items():
            ent = {o: potts.summarise_nested(sorted(fs, key=lambda r: r["outer"]))
                   for o, fs in by_o.items()}
            if "3" in by_o:
                f3 = sorted(by_o["3"], key=lambda r: r["outer"])
                f2 = {r["outer"]: r["test_bits"] for r in by_o.get("2", [])}
                ent["single_clique"] = {}
                for c in f3[0]["single_clique_bits"]:
                    gain = np.array([f2[r["outer"]] - r["single_clique_bits"][c] for r in f3
                                     if r["outer"] in f2])
                    ent["single_clique"][c] = {
                        "bits": float(np.mean([r["single_clique_bits"][c] for r in f3])),
                        "gain_vs_pairs": float(gain.mean()),
                        "gain_se": float(gain.std(ddof=1) / np.sqrt(gain.size))}
            for a, b in (("1", "2"), ("2", "3")):
                if a in by_o and b in by_o:
                    fa = {r["outer"]: r["test_bits"] for r in by_o[a]}
                    d = np.array([fa[r["outer"]] - r["test_bits"] for r in by_o[b]])
                    ent[f"gain_{a}to{b}"] = {"bits": float(d.mean()),
                                             "se": float(d.std(ddof=1) / np.sqrt(d.size)),
                                             "per_fold": d.tolist()}
            res["cv"].setdefault(k, {})[N] = ent
    # Phase 2: regularised landscapes at the CV-selected prior widths.
    sub_seed = {k: seed_of[k] for k in names}
    jobs2 = []
    for k, by_n in res["cv"].items():
        for N, ent in by_n.items():
            if "2" in ent and "3" in ent:
                s2 = [float(np.median(ent[o]["sigma2"])) for o in ("2", "3")]
                jobs2.append((f"map|{k}|{N}", job_map_landscape,
                              (k, *arrays[k], int(N), sub_seed[k], *s2)))
    done2 = run(jobs2)
    res["map_landscape"] = {}
    for key, r in done2.items():
        if key.startswith("map|"):
            _, k, N = key.split("|")
            res["map_landscape"].setdefault(k, {})[N] = r
    OUT.write_text(json.dumps(res))
    log(f"wrote {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
