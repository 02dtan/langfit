"""Stephens & Bialek's pairwise maximum-entropy model of four-letter words,
applied to edited prose (Austen, Gutenberg) and short-form social media (Twitter
2010-2023, Bluesky 2024).

The model (S&B, arXiv:0801.0253, eq. 2):

    P(l1, l2, l3, l4) = exp( sum_{i<j} V_ij(l_i, l_j) ) / Z

is the maximum-entropy distribution with the six observed pairwise letter
marginals. Energy E(w) = -log2 P(w). One model class is used for every in-sample
result, fitted exactly as S&B did: exact marginal matching, so letter pairs never
seen at positions (i, j) are forbidden (E = +inf).

One comparison rule: exact-fit entropies and landscapes depend on sample size (every
unseen pair is a forbidden cell), so every corpus and every Twitter year is analysed
at the same token count N -- the size of S&B's Austen sample in our tokenisation --
over independent subsamples.

Regularisation appears in exactly one place: the held-out test of whether the
pairwise couplings generalise. An exact model gives unseen test words probability
zero, so that test needs a prior; it is a Gaussian V ~ N(0, sigma^2) with sigma^2
chosen by 5-fold cross-validation.

Stages, cached to data/derived/maxent_study.json:
  replication  S&B's own protocol on S&B's corpus (dictionary filter, full sample)
  corpora      per corpus and subsample: S_1, S_2, S_full, local minima, single-letter
               error correction, probe-word energies; for the first subsample also the
               energy-frequency scatter and the words the pairwise model fails on
  cv           held-out bits per token of the independent and pairwise models

Run:  python scripts/maxent_study.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langfit import corpus, maxent as me, social  # noqa: E402

OUT = corpus.DATA / "derived" / "maxent_study.json"
N_STATES = me.A ** 4
LN2 = np.log(2.0)
REPS = 5                            # independent subsamples per corpus
OUTER, INNER = 5, 4                 # nested cross-validation folds
SEED = 20071231                     # arXiv:0801.0253 submission date; any fixed seed
WORKERS = os.cpu_count() or 1

# Probe forms, fixed before any fit: documented four-letter internet abbreviations and
# slang, standard-spelling controls, and transposition typos of frequent words.
PROBES = {
    "slang": ["lmao", "rofl", "omfg", "imho", "smth", "tmrw", "idek", "bruh", "yeet",
              "yolo", "fomo", "smol", "yall", "gurl", "noob", "thru", "stan", "simp",
              "vibe", "meme"],
    "control": ["that", "with", "have", "love", "girl", "cool", "time", "good"],
    "typo": ["waht", "taht", "jsut", "ahve", "wiht", "thsi", "liek", "konw"],
}

# S&B published values (arXiv:0801.0253, text and Fig. 2)
SB = {"tokens": 135441, "types": 763, "S_rand": 18.802, "S_ind": 14.083, "S_2": 7.48,
      "S_full": 6.918, "captured": 0.92, "anc_minima": 136, "anc_minima_words": 118,
      "anc_minima_mass": 0.635, "unseen_model_mass": 0.15, "unseen_dict_share": 0.2}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def finite(x: float):
    return float(x) if np.isfinite(x) else None


# --------------------------------------------------------------------------- #
# corpora
# --------------------------------------------------------------------------- #

def dictionary_mask() -> np.ndarray:
    m = np.zeros(N_STATES, dtype=bool)
    for w in corpus.word_list():
        if len(w) == 4 and w.isascii():
            m[me.ravel(w)] = True
    return m


def load_corpora() -> dict[str, dict]:
    """name -> {"counter": Counter, "meta": {...}}"""
    out = {"Austen": {"counter": corpus.fixed_length_words(
        corpus.load_corpus(list(corpus.AUSTEN))),
        "meta": {"source": "8 Austen texts (S&B ref. 16)"}}}
    others = [n for n in corpus.GUTENBERG if n not in corpus.AUSTEN]
    out["Gutenberg"] = {"counter": corpus.fixed_length_words(corpus.load_corpus(others)),
                        "meta": {"source": f"{len(others)} other 19th-century novels"}}
    pooled = Counter()
    for y, rec in social.load_counts("twitter")["years"].items():
        c = Counter(rec["counts"])
        pooled.update(c)
        out[f"Twitter {y}"] = {"counter": c,
                               "meta": {k: v for k, v in rec.items() if k != "counts"}}
    out["Twitter"] = {"counter": pooled,
                      "meta": {"source": "enryu43/twitter100m_tweets, 16 of 41 shards"}}
    bs = social.load_counts("bluesky")["years"]["2024"]
    out["Bluesky"] = {"counter": Counter(bs["counts"]),
                      "meta": {k: v for k, v in bs.items() if k != "counts"}}
    return out


# --------------------------------------------------------------------------- #
# per-model analysis
# --------------------------------------------------------------------------- #

def analyse(m2: me.MaxEnt, wc: me.WordCounts, dmask: np.ndarray, probes, detail: bool) -> dict:
    """Landscape of an exact pairwise model: minima, misspelling recovery, probes."""
    E = m2.energy()                                  # nats, +inf for forbidden strings
    Eb = E.ravel() / LN2
    P = np.exp(-E.ravel())
    flat = wc.flat_index()
    N = wc.n_tokens
    w = wc.counts / N
    seen = np.zeros(N_STATES, dtype=bool)
    seen[flat] = True
    mins_mask = me.local_minima(E).ravel()
    minima = np.flatnonzero(mins_mask)
    target = me.descend(E)
    rank = np.empty(N_STATES, dtype=np.int64)
    rank[np.argsort(Eb, kind="stable")] = np.arange(1, N_STATES + 1)

    recovery = me.substitution_recovery(E, wc, target)

    unseen_mass = float(P[~seen].sum())
    out = {
        "n_minima": int(minima.size),
        "n_minima_dict": int(dmask[minima].sum()),
        "n_minima_seen": int(seen[minima].sum()),
        "minima_model_mass": float(P[minima].sum()),
        "minima_token_share": float(w[mins_mask[flat]].sum()),
        "unseen_model_mass": unseen_mass,
        "unseen_dict_share": float(P[~seen & dmask].sum() / unseen_mass) if unseen_mass else 0.0,
        "recovery": float((w * recovery).sum()),
        "probes": {},
    }
    for f in probes:
        i = me.ravel(f)
        hit = np.flatnonzero(flat == i)
        out["probes"][f] = {"energy_bits": finite(Eb[i]), "rank": int(rank[i]),
                            "is_min": bool(mins_mask[i]), "target": me.unravel(target[i]),
                            "count": float(wc.counts[hit].sum()) if hit.size else 0.0}
        if detail:
            out["probes"][f]["path"] = [(s, finite(e)) for s, e in me.descent_path(Eb.reshape(E.shape), f)]
    if not detail:
        return out

    # Residual between usage and the pairwise prediction, per seen word (bits).
    # Positive: used more often than its letter pairs predict, i.e. structure beyond
    # pairwise. Ranked by the edge of the exact 95% Poisson interval of the count
    # nearest zero, so a word ranks high only if the residual survives sampling noise.
    n = wc.counts
    lp2 = np.log2(P[flat])
    resid = np.log2(n / N) - lp2
    r_lo = np.log2(chi2.ppf(0.025, 2 * n) / 2 / N) - lp2
    r_hi = np.log2(chi2.ppf(0.975, 2 * (n + 1)) / 2 / N) - lp2
    words = wc.words
    in_dict = dmask[flat]
    order_min = minima[np.argsort(-P[minima])]
    unseen = np.flatnonzero(~seen & (P > 0))
    out.update({
        "top_minima": [(me.unravel(i), float(P[i]), bool(seen[i]), bool(dmask[i]))
                       for i in order_min[:60]],
        "nondict_minima": [(me.unravel(i), float(P[i]), bool(seen[i]))
                           for i in order_min if not dmask[i]][:40],
        "under_predicted": [(words[j], float(n[j]), float(resid[j]), float(r_lo[j]),
                             bool(in_dict[j])) for j in np.argsort(-r_lo)[:40]],
        "over_predicted": [(words[j], float(n[j]), float(resid[j]), float(r_hi[j]),
                            bool(in_dict[j])) for j in np.argsort(r_hi)[:40]],
        "invented": [(me.unravel(i), float(P[i]), bool(dmask[i]))
                     for i in unseen[np.argsort(-P[unseen])][:40]],
        "scatter": {"words": words, "p_emp": w.tolist(), "p2": P[flat].tolist()},
    })
    return out


def subsample(states, counts, N, seed) -> me.WordCounts:
    """N tokens drawn without replacement (multivariate hypergeometric)."""
    rng = np.random.default_rng(seed)
    c = rng.multivariate_hypergeometric(counts.astype(np.int64), N).astype(float)
    keep = c > 0
    return me.WordCounts(states[keep], c[keep], 4)


def job_corpus(args) -> dict:
    """One subsample of one corpus at the common token count N."""
    name, states, counts, N, seed, rep, dmask, probes = args
    wc = subsample(states, counts, N, seed)
    p = wc.counts / N
    m1, d1 = me.fit(wc, 1, "exact")
    m2, d2 = me.fit(wc, 2, "exact")
    out = {"name": name, "rep": rep, "types": int(wc.counts.size),
           "S1": m1.entropy_bits(), "S2": m2.entropy_bits(),
           "S_full": float(-(p * np.log2(p)).sum()),
           "S_full_bias": float((wc.counts.size - 1) / (2 * N * LN2)),  # Miller-Madow scale
           "converged": bool(d1["converged"] and d2["converged"]),
           "fit_marginal_in_se": d2["max_gradient_in_se"],
           "fit_entropy_error_in_se": d2["entropy_error_in_se"]}
    out.update(analyse(m2, wc, dmask, probes, detail=(rep == 0)))
    if rep == 0:
        out["scatter"]["p1"] = m1.prob().ravel()[wc.flat_index()].tolist()
    return out


def job_cv(args):
    """One outer fold of nested CV on the same subsample as the corpus's rep 0."""
    name, states, counts, N, seed, order, outer = args
    wc = subsample(states, counts, N, seed)
    return name, order, me.nested_cv_fold(wc, order, outer, OUTER, INNER, SEED)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def replication(cs: dict, dmask: np.ndarray, probes) -> dict:
    """S&B protocol: Austen, dictionary-filtered, exact fits on the full sample."""
    dic = {me.unravel(i) for i in np.flatnonzero(dmask)}
    wc = me.WordCounts.from_counter(
        Counter({w: n for w, n in cs["Austen"]["counter"].items() if w in dic}))
    p = wc.counts / wc.n_tokens
    m1, d1 = me.fit(wc, 1, "exact")
    m2, d2 = me.fit(wc, 2, "exact")
    m3, d3 = me.fit(wc, 3, "exact")
    out = {"tokens": wc.n_tokens, "types": int(wc.counts.size), "S_rand": 4 * np.log2(26),
           "S_ind": m1.entropy_bits(), "S_2": m2.entropy_bits(), "S_3": m3.entropy_bits(),
           "S_full": float(-(p * np.log2(p)).sum()),
           "converged": bool(d1["converged"] and d2["converged"] and d3["converged"])}
    out["captured"] = (out["S_ind"] - out["S_2"]) / (out["S_ind"] - out["S_full"])
    a = analyse(m2, wc, dmask, probes, detail=True)
    a.pop("scatter")
    out.update(a)
    return out


def main() -> None:
    t0 = time.time()
    dmask = dictionary_mask()
    cs = load_corpora()
    probes = [w for g in PROBES.values() for w in g]
    N = int(sum(cs["Austen"]["counter"].values()))
    res = {"meta": {"N": N, "reps": REPS, "outer_folds": OUTER, "inner_folds": INNER,
                    "seed": SEED, "probes": PROBES, "sigma2_start": list(me.SIGMA2_START),
                    "sb_published": SB,
                    "dictionary": "/usr/share/dict/words (Webster's 2nd)",
                    "corpora": {k: {"tokens": int(sum(v["counter"].values())),
                                    "types": len(v["counter"]), **v["meta"]}
                                for k, v in cs.items()}}}

    log("replication: S&B protocol on Austen")
    res["replication"] = replication(cs, dmask, probes)
    r = res["replication"]
    log(f"  S_ind {r['S_ind']:.3f}  S_2 {r['S_2']:.3f}  S_3 {r['S_3']:.3f}  "
        f"S_full {r['S_full']:.3f}  captured {r['captured']:.3f}  conv {r['converged']}")

    names = ["Austen", "Gutenberg", "Bluesky"] + sorted(
        k for k in cs if k.startswith("Twitter ") and sum(cs[k]["counter"].values()) >= N)
    wcs = {k: me.WordCounts.from_counter(cs[k]["counter"]) for k in names}
    seed_of = {k: SEED + 1000 * i for i, k in enumerate(names)}
    log(f"corpora at N = {N:,} tokens x {REPS} subsamples: {len(names)} corpora")
    jobs = [(k, wcs[k].states, wcs[k].counts, N, seed_of[k] + rep, rep, dmask, probes)
            for k in names for rep in range(REPS)]
    with ProcessPoolExecutor(WORKERS) as ex:
        rows = []
        for row in ex.map(job_corpus, jobs):
            rows.append(row)
            if row["rep"] == REPS - 1:
                log(f"  {row['name']} done")
    res["corpora"] = rows

    years = [k for k in names if k.startswith("Twitter ")]
    cv_names = ["Austen", "Gutenberg", years[0], years[len(years) // 2], years[-1], "Bluesky"]
    log(f"nested CV ({OUTER} outer x {INNER} inner folds) at N = {N:,}: {cv_names}")
    jobs = [(k, wcs[k].states, wcs[k].counts, N, seed_of[k], o, f)
            for o in (2, 1) for k in cv_names for f in range(OUTER)]
    with ProcessPoolExecutor(WORKERS) as ex:
        raw = list(ex.map(job_cv, jobs))
    cv = {}
    for k in cv_names:
        cv[k] = {str(o): me.summarise_nested(sorted((r for n, oo, r in raw if n == k and oo == o),
                                                    key=lambda r: r["outer"]))
                 for o in (1, 2)}
        log(f"  {k}: independent {cv[k]['1']['bits']:.4f} +- {cv[k]['1']['se']:.4f}  pairwise "
            f"{cv[k]['2']['bits']:.4f} +- {cv[k]['2']['se']:.4f} bits/token  "
            f"sigma^2 {cv[k]['2']['sigma2']}  interior {cv[k]['2']['all_interior']}")
    res["cv"] = cv

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res))
    log(f"wrote {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
