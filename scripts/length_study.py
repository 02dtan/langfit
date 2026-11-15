"""Pairwise maximum-entropy models of Twitter words of every length L = 2 ... 20.

Extends the frozen four-letter study (``scripts/maxent_study.py``) to lengths where the
26^L grid cannot be enumerated at every optimisation step. Model: S&B's pairwise
model (``langfit.longwords.PairModel``); fit: maximum pseudo-likelihood with a
Gaussian prior; partition function: exact split enumeration for L <= 8, annealed
importance sampling beyond. See ``docs/first_pass/length.md``.

Two parts, both written to ``data/derived/length_study.json``:
validation   PLM against exact maximum likelihood (``langfit.potts``) at L = 4
             (N = 10^4, N_COMMON, 10^6) and L = 5 (N = 10^4, N_COMMON); AIS against
             exact ln Z at L = 4 ... 8 for PLM fits at N_COMMON. These numbers define
             where the approximation is trustworthy.
lengths      every length 2 ... 20 of the pooled 2006-2023 Twitter counts, at the
             common N_COMMON tokens (all tokens where fewer exist): S_1, the S_2 upper
             bound, model entropy, multi-information captured (lower bound), held-out
             gain, local minima found by descent, and data-adequacy flags.

Run:  .venv/bin/python scripts/length_study.py [--workers 1]
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")     # one BLAS thread per worker process

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langfit import corpus, longwords as lw, maxent as me, potts, social  # noqa: E402

OUT = corpus.DATA / "derived" / "length_study.json"
SOURCE = "twitter_year_L1-20_all.pkl.gz"
LN2 = np.log(2.0)
SEED = 20071231                 # same fixed seed as the frozen study
LENGTHS = range(2, 21)
# Common token count per length: the N of the frozen four-letter study (Austen's
# dictionary-filtered four-letter tokens, S&B's sample size). Holding N fixed across
# lengths makes the finite-sample biases (plug-in entropy, PLM, prior) comparable
# between lengths; lengths with fewer tokens (L >= 17) use all of them and are
# flagged. The validation part measures how every quantity moves with N.
N_COMMON = 142_614
VALIDATION = ((4, 10_000), (4, N_COMMON), (4, 1_000_000), (5, 10_000), (5, N_COMMON))
AIS_CHECK_L = (4, 5, 6, 7, 8)
AIS_SCHEDULE_L = (10, 14, 18)   # beyond exact enumeration: early, middle, late
# Prior-width ablation (does the over-weighting of the deepest well depend on sigma^2?)
# at the first AIS-only length where it appears strongly (L = 10, selected sigma^2 3.2)
# and at one where sigma^2 = 0.32 was selected (L = 14); grid = the frozen study's
# half-decade grid, from 0.32 up to 31.6 (one step past the largest selected value).
ABLATION_L = (10, 14)
ABLATION_SIGMA2 = tuple(10.0 ** (k / 2) for k in range(-1, 4))
AIS_ABLATION_RUNS = 2           # two runs give a between-run spread at half the cost
OUTER_FOLDS = 5                 # test = 1/5 of the tokens, as one outer fold of the frozen study
INNER_FOLDS = 4                 # validation = 1/4 of the remaining tokens
# Selection fits stop at a looser fraction of a standard error than final fits: the
# validation-score differences between half-decade grid points are orders of magnitude
# larger than the change from tightening the optimiser further.
RES_SELECT = 0.1
RES_FINAL = 0.01
# AIS: independent runs x chains x temperatures. The 4000-step schedule comes from a
# pilot at L = 5-8 (docs/first_pass/length.md, sec. 2): 100 steps gave errors up to
# 0.6 nats, 4000 brought them within ~2 SE; four runs give a between-run spread to
# check the delta-method SE against.
AIS_RUNS, AIS_CHAINS, AIS_TEMPS = 4, 500, 4000
# Significance convention for a held-out gain: two standard errors (~95%, one-sided
# 97.7%). A stated convention.
Z_SIG = 2.0
EXHAUSTIVE_L = 5                # dense grid for exhaustive minima (26^5 states)
TOP = 25                        # words listed per category in the JSON


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

_POOLED: dict = {}


def pooled(L: int) -> tuple[np.ndarray, np.ndarray]:
    """All years pooled: (states, counts) of words of length L."""
    if not _POOLED:
        d = social.load_periods(SOURCE)
        for r in d["periods"].values():
            for ln, c in r["counts"].items():
                _POOLED.setdefault(ln, Counter()).update(c)
    return potts.counts_to_arrays(_POOLED[L], L)


def prepare(L: int, N: int | None, seed: int) -> dict:
    """Subsample to N tokens (all if None or more than available) and split tokens into
    test (1/5), validation (1/4 of the rest) and inner training."""
    st, ct = pooled(L)
    avail = float(ct.sum())
    if N is not None and N < avail:
        s, c = potts.subsample(st, ct, N, seed)
    else:
        s, c = st, ct
    rng = np.random.default_rng(seed)
    test = potts.split(c, OUTER_FOLDS, rng)[0]
    train = c - test
    val = potts.split(train, INNER_FOLDS, rng)[0]
    return {"L": L, "states": s, "counts": c, "train": train, "test": test,
            "val": val, "inner": train - val, "available": avail}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def bits_per_token(logp: np.ndarray, counts: np.ndarray) -> float:
    m = counts > 0
    return float(-(counts[m] * logp[m]).sum() / counts[m].sum() / LN2)


def paired_gain(lp2: np.ndarray, lp1: np.ndarray, counts: np.ndarray) -> tuple[float, float]:
    """Mean per-token gain log2 P2 - log2 P1 on the given tokens and its SE, treating
    tokens as independent draws."""
    m = counts > 0
    d = (lp2[m] - lp1[m]) / LN2
    w = counts[m]
    n = w.sum()
    mean = float((w * d).sum() / n)
    var = float((w * (d - mean) ** 2).sum() / (n - 1))
    return mean, float(np.sqrt(var / n))


def plugin_entropy(counts: np.ndarray) -> tuple[float, float]:
    """Plug-in entropy (bits) and the Miller-Madow first-order bias scale (K-1)/(2N ln 2)."""
    c = counts[counts > 0]
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum()), float((c.size - 1) / (2 * c.sum() * LN2))


def log_partition(model: lw.PairModel, base: lw.Independent, seed: int,
                  with_ais: bool) -> dict:
    """Exact ln Z where enumerable, AIS otherwise (and also, when ``with_ais``, AIS next
    to the exact value as a check)."""
    out = {}
    if model.L <= lw.L_EXACT_MAX:
        t = time.time()
        out["exact"] = lw.exact_log_partition(model)
        out["exact"]["seconds"] = time.time() - t
    if model.L > lw.L_EXACT_MAX or with_ais:
        out["ais"] = ais(model, base, seed)
    best = out.get("exact") or out["ais"]
    out["logZ"] = best["logZ"]
    out["logZ_se"] = 0.0 if "exact" in out else out["ais"]["se"]
    out["entropy_bits"] = best["entropy_bits"]
    out["entropy_se_bits"] = 0.0 if "exact" in out else out["ais"]["entropy_se_bits"]
    if "exact" in out and "ais" in out:
        out["ais_error"] = out["ais"]["logZ"] - out["exact"]["logZ"]
        out["ais_error_in_se"] = out["ais_error"] / out["ais"]["se"]
        out["ais_entropy_error_bits"] = out["ais"]["entropy_bits"] - out["exact"]["entropy_bits"]
    return out


def ais(model: lw.PairModel, base: lw.Independent, seed: int, runs: int = AIS_RUNS) -> dict:
    """``runs`` independent AIS runs pooled into one estimate; the spread of the per-run
    estimates is reported next to the pooled delta-method SE as a consistency check.

    Entropy: each run gives its own estimate ln Z_r - <ln w>_r; the replicated
    uncertainty is the between-run SD / sqrt(runs). Validation against exact entropies
    (L <= 8) shows errors up to ~3.5 of the delta-method SE, so neither is a calibrated
    error bar; both are reported as numerical diagnostics."""
    t = time.time()
    rr = [lw.ais_log_partition(model, base, AIS_CHAINS, AIS_TEMPS, seed=seed + r)
          for r in range(runs)]
    # pool: ln mean over all chains = logsumexp_r (ln Z_r) - ln R (equal chains per run)
    lzr = np.array([r["logZ"] for r in rr])
    lz = float(np.logaddexp.reduce(lzr) - np.log(len(rr)))
    w = np.exp(lzr - lz)
    se_runs = float(np.std(w, ddof=1) / np.sqrt(len(rr))) if len(rr) > 1 else float("nan")
    se_pool = float(np.sqrt(np.sum([(wi * r["se"]) ** 2 for wi, r in zip(w, rr)])) / len(rr))
    mlw = float(np.sum([wi * r["mean_lw"] for wi, r in zip(w, rr)]) / w.sum())
    se_mlw = float(np.sqrt(np.sum([(wi * r["se_mean_lw"]) ** 2 for wi, r in zip(w, rr)])) / w.sum())
    se = float(np.nanmax([se_pool, se_runs]))
    ent_r = np.array([r["entropy_bits"] for r in rr])
    return {"logZ": lz, "se": se, "se_pooled_delta": se_pool, "se_between_runs": se_runs,
            "runs_logZ": lzr.tolist(), "ess": [r["ess"] for r in rr],
            "runs_entropy_bits": ent_r.tolist(),
            "entropy_se_between_runs_bits": (float(ent_r.std(ddof=1) / np.sqrt(len(rr)))
                                             if len(rr) > 1 else None),
            "chains": AIS_CHAINS, "temps": AIS_TEMPS, "runs": len(rr),
            "mean_lw": mlw, "entropy_bits": (lz - mlw) / LN2,
            "entropy_se_bits": float(np.hypot(se, se_mlw) / LN2), "seconds": time.time() - t}


def landscape(model: lw.PairModel, d: dict, lz: float) -> dict:
    """Strict local minima reached by steepest descent from every training type."""
    s, c = d["states"], d["train"]
    keep = c > 0
    s, c = s[keep], c[keep]
    N = c.sum()
    is_min = lw.is_strict_minimum(model, s)
    end = lw.descend(model, s)
    uniq, inv = np.unique(end, axis=0, return_inverse=True)
    inv = inv.ravel()
    umin = lw.is_strict_minimum(model, uniq)
    seen_index = {tuple(r): k for k, r in enumerate(s)}
    useen = np.array([tuple(r) in seen_index for r in uniq])
    ulw = model.log_weight(uniq)
    flow = np.bincount(inv, weights=c, minlength=len(uniq)) / N   # token share draining to each end
    words = lw.words_of(uniq)
    order = np.argsort(-ulw)
    out = {
        "types_that_are_minima": float(is_min.mean()),
        "tokens_on_minima": float(c[is_min].sum() / N),
        "terminal_states": int(len(uniq)),
        "minima_found": int(umin.sum()),
        "minima_found_seen": int((umin & useen).sum()),
        "minima_found_unseen": int((umin & ~useen).sum()),
        "plateau_terminals": int((~umin).sum()),
        "minima_found_model_mass": float(np.exp(ulw[umin] - lz).sum()),
        "token_share_draining_to_unseen_minima": float(flow[umin & ~useen].sum()),
        "top_minima": [(words[k], float(-(ulw[k] - lz) / LN2), bool(useen[k]), float(flow[k]))
                       for k in order if umin[k]][:TOP],
        "unseen_minima": [(words[k], float(-(ulw[k] - lz) / LN2), float(flow[k]))
                          for k in order if umin[k] and not useen[k]][:TOP],
    }
    if model.L <= EXHAUSTIVE_L:
        E = -model.dense_log_weights()
        allmin = me.local_minima(E).ravel()
        found = np.zeros(allmin.size, dtype=bool)
        found[np.ravel_multi_index(uniq[umin].T, (lw.A,) * model.L)] = True
        p = np.exp(-E.ravel() - lz)
        out["exhaustive_minima"] = int(allmin.sum())
        out["exhaustive_minima_mass"] = float(p[allmin].sum())
        out["descent_recall_minima"] = float(found[allmin].sum() / allmin.sum())
        out["descent_recall_mass"] = float(p[found & allmin].sum() / p[allmin].sum())
    return out


# --------------------------------------------------------------------------- #
# per-length analysis
# --------------------------------------------------------------------------- #

def analyse_length(L: int, N: int | None, seed: int, with_ais: bool,
                   exact_reference: bool) -> tuple[dict, dict]:
    """Fit and characterise one length; returns (results, fitted objects)."""
    t0 = time.time()
    d = prepare(L, N, seed)
    s, train, test = d["states"], d["train"], d["test"]
    Ntr, Nte = float(train.sum()), float(test.sum())
    n_par = L * (L - 1) // 2 * lw.A ** 2
    res = {"L": L, "available_tokens": d["available"], "tokens": float(d["counts"].sum()),
           "train_tokens": Ntr, "test_tokens": Nte, "train_types": int((train > 0).sum()),
           "parameters": n_par, "train_tokens_per_parameter": Ntr / n_par}

    # independent baselines: exact (S&B) for S_1, MAP for held-out code length
    ind_exact = lw.fit_independent(s, train, L)
    sel1 = lw.select_sigma2_independent(s, d["inner"], d["val"], L)
    ind = lw.fit_independent(s, train, L, sel1["best_sigma2"])
    res["independent_selection"] = sel1
    res["S1"] = ind_exact.entropy_bits()
    res["S_full_plugin"], res["S_full_mm_bias"] = plugin_entropy(train)

    # pairwise by PLM: prior width by held-out pseudo-likelihood
    sel = lw.select_sigma2(s, d["inner"], d["val"], L, score="pll", resolution=RES_SELECT)
    model, diag = lw.fit_plm(s, train, L, sel["best_sigma2"], init=sel.pop("model"),
                             resolution=RES_FINAL)
    res["selection_pll"] = sel
    res["fit"] = diag
    res["sigma2"] = sel["best_sigma2"]
    if L <= 6:
        # the same selection by held-out exact likelihood, to test the PLL proxy
        sel_ll = lw.select_sigma2(s, d["inner"], d["val"], L, score="ll", resolution=RES_SELECT)
        sel_ll.pop("model")
        res["selection_ll"] = sel_ll

    Z = log_partition(model, ind, seed, with_ais)
    lz = Z["logZ"]
    res["partition"] = {k: v for k, v in Z.items()}
    lw_s = model.log_weight(s)
    lp2 = lw_s - lz
    lp1 = ind.log_prob(s)
    res["S_model"] = Z["entropy_bits"]
    res["S_model_se"] = Z["entropy_se_bits"]
    res["logZ_se_bits"] = Z["logZ_se"] / LN2
    res["train_bits_pairwise"] = bits_per_token(lp2, train)
    res["train_bits_independent_map"] = bits_per_token(lp1, train)
    # held out
    res["test_bits_independent"] = bits_per_token(lp1, test)
    res["test_bits_pairwise"] = bits_per_token(lp2, test)
    g, g_se = paired_gain(lp2, lp1, test)
    res["heldout_gain_bits"] = g
    res["heldout_gain_se_tokens"] = g_se
    res["heldout_gain_se"] = float(np.hypot(g_se, res["logZ_se_bits"]))
    res["heldout_gain_significant"] = bool(g > Z_SIG * res["heldout_gain_se"])
    res["heldout_pll_pairwise"] = lw.pseudo_loglik(model, s, test)
    res["heldout_pll_independent"] = float(-res["test_bits_independent"])   # PLL = LL for q
    res["heldout_pll_gain_bits"] = res["heldout_pll_pairwise"] - res["heldout_pll_independent"]
    # how much of the model's probability lies on words seen in training / test
    res["model_mass_on_train_types"] = float(np.exp(lp2[train > 0]).sum())
    res["model_mass_on_any_seen_type"] = float(np.exp(lp2).sum())
    res["landscape"] = landscape(model, d, lz)

    if exact_reference and L <= EXHAUSTIVE_L:
        res["exact_reference"] = exact_reference_fits(model, ind, d, lz)
    label_entropies(res)
    res["seconds"] = time.time() - t0
    return res, {"model": model, "data": d}


def label_entropies(r: dict) -> dict:
    """Attach the S_2 and captured-information quantities with their status.

    The training code length of any pairwise Gibbs model is >= the sample maximum-
    entropy S_2 (ML / max-ent duality), so with exact ln Z it is a rigorous upper
    bound on S_2 and the captured fraction computed from it a rigorous lower bound.
    With AIS ln Z (L > L_EXACT_MAX), ln Z_hat is biased low in expectation, which
    lowers the code length; it is then an estimate with a numerical SE, not a bound.
    Idempotent; also converts records written before this labelling (``S2_upper``)."""
    if "S2_upper" in r:
        r["train_bits_pairwise"] = r.pop("S2_upper")
    r.pop("captured_lower", None)
    exact = "exact" in r["partition"]
    se = r.get("logZ_se_bits", 0.0)
    denom = r["S1"] - r["S_full_plugin"]
    r["S2_train_code_bits"] = r["train_bits_pairwise"]
    r["S2_train_code_is_rigorous_upper_bound"] = bool(exact)
    r["S2_train_code_numerical_se_bits"] = float(se)
    r["captured"] = (r["S1"] - r["train_bits_pairwise"]) / denom
    r["captured_is_rigorous_lower_bound"] = bool(exact)
    r["captured_numerical_se"] = float(se / denom)
    return r


def exact_reference_fits(model: lw.PairModel, ind: lw.Independent, d: dict, lz: float) -> dict:
    """Exact maximum-likelihood fits (``potts.fit``) on the same training tokens: MAP at
    the PLM-selected prior width (same estimand family, different estimator) and the
    exact-mode S_2 of S&B. Compared on couplings, entropy, code lengths, KL and minima."""
    L, s, train, test = model.L, d["states"], d["train"], d["test"]
    keep = train > 0
    out = {}
    t = time.time()
    ex, dx = potts.fit(s[keep], train[keep], L, 2, "map", sigma2=model.sigma2)
    out["map_seconds"] = time.time() - t
    out["map_converged"] = bool(dx["converged"])
    Vex = [np.asarray(v) for v in ex.V]
    out["couplings"] = lw.coupling_agreement(model.to_V(), Vex)
    lp_ex = np.asarray(ex.log_prob()).ravel()
    lp_pl = model.dense_log_weights().ravel() - lz
    flat = np.ravel_multi_index(s.T, (lw.A,) * L)
    p_ex = np.exp(lp_ex)
    out["entropy_exact_map"] = float(ex.entropy_bits())
    out["entropy_plm"] = float(-(np.exp(lp_pl) * lp_pl).sum() / LN2)
    out["kl_exact_to_plm_bits"] = float((p_ex * (lp_ex - lp_pl)).sum() / LN2)
    out["train_bits_exact_map"] = bits_per_token(lp_ex[flat], train)
    out["train_bits_plm"] = bits_per_token(lp_pl[flat], train)
    out["test_bits_exact_map"] = bits_per_token(lp_ex[flat], test)
    out["test_bits_plm"] = bits_per_token(lp_pl[flat], test)
    E_ex = -lp_ex.reshape((lw.A,) * L)
    E_pl = -lp_pl.reshape((lw.A,) * L)
    m_ex = me.local_minima(E_ex).ravel()
    m_pl = me.local_minima(E_pl).ravel()
    out["minima_exact"] = int(m_ex.sum())
    out["minima_plm"] = int(m_pl.sum())
    out["minima_shared"] = int((m_ex & m_pl).sum())
    out["minima_jaccard"] = float((m_ex & m_pl).sum() / (m_ex | m_pl).sum())
    out["exact_minima_mass_shared"] = float(p_ex[m_ex & m_pl].sum() / p_ex[m_ex].sum())
    del E_ex, E_pl, lp_pl
    t = time.time()
    exm, dxm = potts.fit(s[keep], train[keep], L, 2, "exact")
    out["exact_mode_seconds"] = time.time() - t
    out["S2_exact"] = float(exm.entropy_bits())
    out["exact_mode_converged"] = bool(dxm["converged"])
    return out


def ais_check(L: int, N: int, seed: int) -> dict:
    """AIS against exact ln Z for a PLM fit at (L, N), prior width by held-out PLL."""
    d = prepare(L, N, seed)
    s = d["states"]
    sel = lw.select_sigma2(s, d["inner"], d["val"], L, score="pll", resolution=RES_SELECT)
    model, diag = lw.fit_plm(s, d["train"], L, sel["best_sigma2"], init=sel.pop("model"),
                             resolution=RES_FINAL)
    ind = lw.fit_independent(s, d["train"], L, sel["best_sigma2"])
    Z = log_partition(model, ind, seed, with_ais=True)
    return {"L": L, "N": float(d["counts"].sum()), "sigma2": sel["best_sigma2"],
            "converged": diag["converged"], "logZ": Z["exact"]["logZ"],
            "ais_logZ": Z["ais"]["logZ"], "ais_se": Z["ais"]["se"],
            "ais_error": Z["ais_error"], "ais_error_in_se": Z["ais_error_in_se"],
            "ais_ess": Z["ais"]["ess"], "ais_runs_logZ": Z["ais"]["runs_logZ"],
            "entropy_exact": Z["exact"]["entropy_bits"], "entropy_ais": Z["ais"]["entropy_bits"],
            "entropy_ais_se": Z["ais"]["entropy_se_bits"],
            "exact_seconds": Z["exact"]["seconds"], "ais_seconds": Z["ais"]["seconds"]}


def ais_schedule(L: int, lengths: dict) -> dict:
    """ln Z of the length-study model at L > L_EXACT_MAX against the annealing length.

    No exact value exists here, so two weaker checks: (i) AIS at 1000, 4000 and 16000
    temperatures (one run of AIS_CHAINS chains each): ln Z_hat is biased low by an amount
    that shrinks as the schedule lengthens, so a rise with T flags an underestimate;
    (ii) a certificate: Z >= sum of w over any set of distinct strings, here the
    training types and every terminal state of steepest descent from them.
    The model is refitted at the prior width recorded by the length job on the same
    subsample (the objective is strictly convex, so this reproduces it to the stopping
    tolerance)."""
    r = lengths[str(L)]
    d = prepare(L, N_COMMON, SEED + L)
    s, train = d["states"], d["train"]
    model, diag = lw.fit_plm(s, train, L, r["sigma2"], resolution=RES_FINAL)
    sel1 = r["independent_selection"]["best_sigma2"]
    ind = lw.fit_independent(s, train, L, sel1)
    keep = train > 0
    pts = np.unique(np.concatenate([s[keep], lw.descend(model, s[keep])]), axis=0)
    lower = float(np.logaddexp.reduce(model.log_weight(pts)))
    out = {"L": L, "sigma2": r["sigma2"], "converged": diag["converged"],
           "logZ_lower_certificate": lower, "certificate_points": int(len(pts)),
           "length_job_logZ": r["partition"]["logZ"], "schedule": []}
    for T in (1000, 4000, 16000):
        a = lw.ais_log_partition(model, ind, AIS_CHAINS, T, seed=SEED + 7 * T + L)
        out["schedule"].append({"temps": T, "logZ": a["logZ"], "se": a["se"], "ess": a["ess"],
                                "entropy_bits": a["entropy_bits"],
                                "entropy_se_bits": a["entropy_se_bits"]})
        log(f"  ais schedule L={L} T={T}: lnZ {a['logZ']:.3f} +- {a['se']:.3f} "
            f"(certificate {lower:.3f})")
    return out


def _refit(L: int, r: dict) -> tuple:
    """Refit the length-study model on the same subsample and split at its recorded
    prior width, from zero (the length job warm-started from the selection path; the
    objective is strictly convex, so the two agree to the stopping tolerance)."""
    d = prepare(L, N_COMMON, SEED + L)
    s, train = d["states"], d["train"]
    model, diag = lw.fit_plm(s, train, L, r["sigma2"], resolution=RES_FINAL)
    ind = lw.fit_independent(s, train, L, r["independent_selection"]["best_sigma2"])
    return d, model, diag, ind


def entropy_replicates(L: int, lengths: dict) -> dict:
    """Replicated AIS entropy for an AIS-only length: AIS_RUNS runs, each with its own
    ln Z_r and entropy estimate, so the between-run spread of the *entropy* (not only of
    ln Z) is available. Validation (``ais_check``) shows the delta-method entropy SE is
    not calibrated; the between-run spread is a replicated numerical diagnostic, and
    it too cannot see wells that no run visits."""
    r = lengths[str(L)]
    d, model, diag, ind = _refit(L, r)
    a = ais(model, ind, SEED + 31 * L)
    a.pop("samples", None)
    return {"L": L, "sigma2": r["sigma2"], "converged": diag["converged"],
            "length_job_logZ": r["partition"]["logZ"],
            "length_job_entropy_bits": r["S_model"], **a}


def prior_ablation(L: int, lengths: dict) -> dict:
    """Same data, same split, prior widths across ABLATION_SIGMA2: does the deepest
    well's over-weighting and the duality gap depend on sigma^2? Tests the hypothesis
    that weakly identified relative well depths are being set by the prior. Each fit is
    scored by held-out code length and PLL, AIS ln Z (AIS_ABLATION_RUNS runs), entropy,
    duality gap, and calibration of the most probable training types."""
    r = lengths[str(L)]
    d = prepare(L, N_COMMON, SEED + L)
    s, train, test = d["states"], d["train"], d["test"]
    ind = lw.fit_independent(s, train, L, r["independent_selection"]["best_sigma2"])
    keep = train > 0
    st, ct = s[keep], train[keep]
    pooled(L)
    n_all = sum(_POOLED[L].values())
    words = lw.words_of(st)
    f_pool = np.array([_POOLED[L].get(w, 0) / n_all for w in words])
    top = np.argsort(-ct)[:TOP]
    rows, init = [], None
    for s2 in ABLATION_SIGMA2:
        t = time.time()
        model, diag = lw.fit_plm(s, train, L, s2, init=init, resolution=RES_FINAL)
        init = model
        a = ais(model, ind, SEED + 53 * L, runs=AIS_ABLATION_RUNS)
        lz = a["logZ"]
        lp_t = model.log_weight(st) - lz
        train_bits = float(-(ct * lp_t).sum() / ct.sum() / LN2)
        keep_te = test > 0
        lp_te = model.log_weight(s[keep_te]) - lz
        mins = lw.is_strict_minimum(model, st)
        deepest = int(np.argmax(np.where(mins, lp_t, -np.inf)))
        rows.append({
            "sigma2": s2, "converged": diag["converged"], "iterations": diag["iterations"],
            "logZ": lz, "logZ_se": a["se"], "ess": a["ess"],
            "entropy_bits": a["entropy_bits"], "runs_entropy_bits": a["runs_entropy_bits"],
            "train_bits": train_bits,
            "duality_gap_bits": train_bits - a["entropy_bits"],
            "test_bits": float(-(test[keep_te] * lp_te).sum() / test[keep_te].sum() / LN2),
            "heldout_pll_bits": lw.pseudo_loglik(model, s, test),
            "model_mass_on_train_types": float(np.exp(lp_t).sum()),
            "deepest_training_minimum": words[deepest],
            "deepest_log2_ratio_to_pooled_freq": float(lp_t[deepest] / LN2
                                                       - np.log2(f_pool[deepest])),
            "top_types_mean_log2_ratio": float(np.mean(lp_t[top] / LN2 - np.log2(f_pool[top]))),
            "top_types_max_log2_ratio": float(np.max(lp_t[top] / LN2 - np.log2(f_pool[top]))),
            "seconds": time.time() - t})
        log(f"  ablation L={L} s2={s2:.3g}: gap {rows[-1]['duality_gap_bits']:+.2f} "
            f"test {rows[-1]['test_bits']:.3f} deepest {words[deepest]} "
            f"{rows[-1]['deepest_log2_ratio_to_pooled_freq']:+.2f}")
    return {"L": L, "selected_sigma2": r["sigma2"], "rows": rows}


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #

def job(spec: tuple) -> tuple:
    kind = spec[0]
    if kind == "length":
        _, L = spec
        res, _ = analyse_length(L, N_COMMON, SEED + L, with_ais=False, exact_reference=False)
        return spec, res
    if kind == "validate":
        _, L, N = spec
        res, _ = analyse_length(L, N, SEED + 100 * L + int(np.log10(N)), with_ais=True,
                                exact_reference=True)
        return spec, res
    if kind == "ais":
        _, L, N = spec
        return spec, ais_check(L, N, SEED + 1000 * L + int(np.log10(N)))
    raise ValueError(kind)


def _prefix(lengths: dict, key: str) -> int | None:
    """Largest L such that every length 2..L is present and passes ``key``; None if
    length 2 is absent or fails."""
    L_ok = None
    L = 2
    while str(L) in lengths and lengths[str(L)][key]:
        L_ok = L
        L += 1
    return L_ok


def summarise(out: dict) -> dict:
    """Heuristic data-adequacy screens and language coverage.

    These are *screens*, not validated limits: exact PLM-vs-ML comparisons exist only
    at L = 4, 5, and their thresholds are transferred to other lengths. Two nested
    screens. Local screen: (a) held-out gain of the pairwise over the independent model
    > Z_SIG standard errors (token sampling and ln Z error combined), (b) final fit met
    its stopping rule, (c) training tokens (the 4/5 refit) per coupling parameter >= the
    smallest ratio at which PLM was compared with an exact fit, (d) ln Z respects the
    certificate ln Z >= ln sum of w over training types (model mass on them <= 1).
    All-quantity screen adds (e) |D| <= the largest |D| in the validation fits, D =
    training code length - model entropy. For any Gibbs model D = theta . (mu_model -
    mu_emp) / ln 2: one signed contraction, small at the ML/MAP optimum, not a bound
    on marginal errors, KL, minima or descent. Each verdict is the contiguous prefix
    of passing lengths starting at L = 2."""
    lengths = out["lengths"]
    d = social.load_periods(SOURCE)
    tok = Counter()
    for r in d["periods"].values():
        for ln, c in r["counts"].items():
            tok[ln] += sum(c.values())
    total = sum(tok.values())
    for v in out["validation"]:
        label_entropies(v)
    ratio_min = min((v["train_tokens_per_parameter"] for v in out["validation"]),
                    default=None)
    gap_max = max((abs(v["train_bits_pairwise"] - v["S_model"]) for v in out["validation"]),
                  default=None)
    calibration = {}
    for L in sorted(lengths, key=int):
        r = label_entropies(lengths[L])
        r["duality_gap_bits"] = r["train_bits_pairwise"] - r["S_model"]
        r["screen"] = {
            "gain_significant": bool(r["heldout_gain_significant"]),
            "converged": bool(r["fit"]["converged"]),
            "within_compared_tokens_per_parameter": bool(
                ratio_min is not None and r["train_tokens_per_parameter"] >= ratio_min),
            "logZ_certificate_respected": bool(r["model_mass_on_train_types"] <= 1.0)}
        ok = all(r["screen"].values())
        r["screen"]["duality_gap_within_compared"] = bool(
            gap_max is not None and abs(r["duality_gap_bits"]) <= gap_max)
        r["passes_local_screen"] = bool(ok)
        r["passes_all_quantity_screen"] = bool(ok and r["screen"]["duality_gap_within_compared"])
        for k in ("adequacy", "adequate_local", "adequate_global"):
            r.pop(k, None)
        # calibration of the deepest well: model probability against the frequency
        # of the same word in all pooled tokens of that length
        w, e_bits, _, _ = r["landscape"]["top_minima"][0]
        pooled(int(L))
        n_all = sum(_POOLED[int(L)].values())
        f = _POOLED[int(L)].get(w, 0) / n_all
        calibration[L] = {"word": w, "sigma2": r["sigma2"], "model_p": 2.0 ** -e_bits,
                          "pooled_freq": f,
                          "log2_ratio": float(np.log2(2.0 ** -e_bits / f)) if f else None}
    L_local = _prefix(lengths, "passes_local_screen")
    L_all = _prefix(lengths, "passes_all_quantity_screen")

    def share(hi, lo=1):
        return sum(v for k, v in tok.items() if lo <= k <= hi) / total if hi else None

    return {"status": "heuristic screens: thresholds from exact comparisons at L = 4, 5 "
                      "transferred to other lengths; not a validation of any observable "
                      "at L >= 6",
            "L_screen_all_quantities": L_all, "L_screen_local": L_local,
            "criterion": "contiguous prefix from L = 2 of lengths passing: held-out gain "
                         f"> {Z_SIG} SE, converged, training tokens per parameter >= "
                         "smallest ratio compared with an exact fit, ln Z certificate; "
                         "the all-quantity screen adds |duality gap| <= largest in the "
                         "compared fits",
            "exactly_computed_lengths": f"2..{lw.L_EXACT_MAX} (ln Z, model entropy, "
                                        "training and test code lengths exact for the "
                                        "fitted model)",
            "validated_against_exact_ML_lengths": "4, 5 (PLM vs exact MAP / exact S_2)",
            "compared_tokens_per_parameter_min": ratio_min,
            "compared_duality_gap_max_bits": gap_max,
            "top_minimum_calibration": calibration,
            "tokens_by_length": {int(k): int(v) for k, v in sorted(tok.items())},
            "token_share_upto_L_screen_all": share(L_all),
            "token_share_2_to_L_screen_all": share(L_all, 2),
            "token_share_upto_L_screen_local": share(L_local),
            "token_share_2_to_L_screen_local": share(L_local, 2),
            "token_share_exact_logZ_2_to_8": share(lw.L_EXACT_MAX, 2),
            "token_share_validated_4_5": share(5, 4),
            "total_tokens_L1_20": int(total)}


def _done(out: dict) -> set:
    done = {("length", int(L)) for L in out["lengths"]}
    done |= {("validate", r["L"], r["N_requested"]) for r in out["validation"]}
    done |= {("ais", r["L"], r["N_requested"]) for r in out["ais_check"]}
    return done


def _record(out: dict, sp: tuple, res: dict) -> None:
    if sp[0] == "length":
        out["lengths"][str(sp[1])] = res
        log(f"L={sp[1]}: S1 {res['S1']:.3f} train code {res['train_bits_pairwise']:.3f} "
            f"gain {res['heldout_gain_bits']:.3f}+-{res['heldout_gain_se']:.3f} "
            f"minima {res['landscape']['minima_found']} ({res['seconds']:.0f}s)")
    elif sp[0] == "validate":
        res["N_requested"] = sp[2]
        out["validation"].append(res)
        r = res["exact_reference"]
        log(f"validate L={sp[1]} N={sp[2]}: pearson {r['couplings']['pearson']:.4f} "
            f"dS {r['entropy_plm'] - r['entropy_exact_map']:+.4f} "
            f"S2 {r['S2_exact']:.3f} vs train code {res['train_bits_pairwise']:.3f}")
    else:
        res["N_requested"] = sp[2]
        out["ais_check"].append(res)
        log(f"ais L={sp[1]} N={sp[2]}: err {res['ais_error']:+.4f} se {res['ais_se']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--lengths", default=None, help="comma list, default 2..20")
    ap.add_argument("--skip-validation", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    lengths = [int(x) for x in a.lengths.split(",")] if a.lengths else list(LENGTHS)
    specs = [("length", L) for L in lengths]
    if not a.skip_validation:
        specs += [("validate", L, N) for L, N in VALIDATION]
        specs += [("ais", L, N_COMMON) for L in AIS_CHECK_L]
    partial = OUT.with_suffix(".partial.json")
    # Resume: jobs already recorded in the partial file (or a finished output) are not
    # rerun. Delete data/derived/length_study.json to recompute from scratch.
    src = partial if partial.exists() else OUT
    if src.exists():
        out = json.loads(src.read_text())
    else:
        out = {"meta": {}, "lengths": {}, "validation": [], "ais_check": []}
    out["meta"] = {"source": SOURCE, "pooled_years": "all (2006-2023)", "seed": SEED,
                   "N_common": N_COMMON, "outer_folds": OUTER_FOLDS,
                   "inner_folds": INNER_FOLDS, "res_select": RES_SELECT,
                   "res_final": RES_FINAL,
                   "ais": {"runs": AIS_RUNS, "chains": AIS_CHAINS, "temps": AIS_TEMPS},
                   "L_exact_max": lw.L_EXACT_MAX, "z_sig": Z_SIG,
                   "sigma2_start": list(me.SIGMA2_START)}
    done = _done(out)
    specs = [sp for sp in specs if sp not in done]
    log(f"{len(specs)} jobs to run on {a.workers} worker(s); {len(done)} already done")
    if a.workers <= 1:
        for sp in specs:
            _record(out, *job(sp))
            partial.write_text(json.dumps(out, default=float))
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            futs = [ex.submit(job, sp) for sp in specs]
            for f in as_completed(futs):
                _record(out, *f.result())
                partial.write_text(json.dumps(out, default=float))
    out.setdefault("ais_schedule", [])
    out.setdefault("entropy_replicates", [])
    out.setdefault("prior_ablation", [])
    if not a.skip_validation:
        for key, Ls, fn in (("ais_schedule", AIS_SCHEDULE_L, ais_schedule),
                            ("prior_ablation", ABLATION_L, prior_ablation),
                            ("entropy_replicates",
                             [L for L in LENGTHS if L > lw.L_EXACT_MAX], entropy_replicates)):
            have = {r["L"] for r in out[key]}
            for L in Ls:
                if L not in have and str(L) in out["lengths"]:
                    out[key].append(fn(L, out["lengths"]))
                    partial.write_text(json.dumps(out, default=float))
    if out["lengths"]:
        out["summary"] = summarise(out)
    out["meta"]["seconds_last_invocation"] = time.time() - t0
    OUT.write_text(json.dumps(out, default=float))
    partial.unlink(missing_ok=True)
    log(f"wrote {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
