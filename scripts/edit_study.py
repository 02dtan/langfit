"""Edit-move landscape of Twitter words across lengths 3-5 (substitution, adjacent
transposition, insertion, deletion), compared with S&B's substitution-only landscape.

Model. For each length L the pairwise maximum-entropy (S&B) model is fitted by exact
enumeration (``langfit.potts``) to the pooled Twitter counts of all years; P(L) is the
token share of length L among all tokens of length 1-20 in the same data. The cross-
length energy is the grand-canonical E(w) = H_L(w) - mu_L = -log2 P(w) of
``langfit.edits`` (bits). Two fitting modes, as in the four-letter study:

  exact  S&B's protocol (unseen letter pairs forbidden, E = +inf); the primary model.
  map    Gaussian prior on couplings, sigma^2 by 4-fold CV (half-decade grid extended
         until interior); every string has finite energy. Robustness check, run on
         the L = 3, 4 space only (``MAP_LENGTHS``), against exact on the same space.

Stages (cached):
  fits         data/derived/edit_models/twitter_pooled_L{L}_{mode}.npz
  landscape    minima, lost wells and the move that kills them, basins, boundary checks
  recovery     exact enumeration of all single-edit corruptions of every observed word
  probes       descent paths of waht, taht, jsut, ahve, wiht, thsi, liek, konw
  channel      KCG-style EM estimate of per-type error rates; MAP decoder vs descent
  closure      lengths 1-5 (L = 1, 2 models are exact empirical joints): lower boundary
  map_check    exact vs map landscapes on L = 3, 4
Output: data/derived/edit_study.json (exact on L = 3-5 under key "exact").

Run:  python scripts/edit_study.py [--workers 1]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langfit import corpus, edits as ed, potts, social  # noqa: E402

DERIVED = corpus.DATA / "derived"
OUT = DERIVED / "edit_study.json"
MODELS = DERIVED / "edit_models"
SOURCE = "twitter_year_L1-20_all.pkl.gz"
LENGTHS = (3, 4, 5)
CLOSURE = (1, 2, 3, 4, 5)
# Map-mode robustness check: L = 3, 4 only. At L = 5 the 4-fold CV of sigma^2 needs
# ~20 fits on the 26^5 grid, too costly on the shared machine; the exact L = 5 model
# forbids only 0.73% of states, and forbidden strings still relax under descent.
MAP_LENGTHS = (3, 4)
SEED = 19640301                       # Damerau, CACM 7(3), March 1964; any fixed seed
PROBES = ["waht", "taht", "jsut", "ahve", "wiht", "thsi", "liek", "konw"]
# Monte Carlo sample per error type for the decoder comparison. Numerical tolerance:
# the binomial standard error of a recovery rate is <= 0.5/sqrt(K) = 0.005.
K_MC = 10_000
TOP = 25                              # display only: rows listed per table


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def pooled_counts() -> tuple[dict[int, Counter], dict, dict[int, float]]:
    """Pooled counts of every year. Word types are kept only for the lengths analysed
    (1-5) and L = 6 (upper-boundary check); other lengths contribute token totals
    (for P(L)) only, which keeps memory to what the study uses."""
    data = social.load_periods(SOURCE)
    keep = set(CLOSURE) | {max(LENGTHS) + 1}
    by_len: dict[int, Counter] = {}
    tot: Counter = Counter()
    for rec in data["periods"].values():
        for L, c in rec["counts"].items():
            L = int(L)
            tot[L] += sum(c.values())
            if L in keep:
                by_len.setdefault(L, Counter()).update(c)
    meta = {"source": SOURCE, "files": data["files"], "years": sorted(data["periods"]),
            "tokens_all_lengths": float(sum(tot.values())),
            "tokens_by_length": {str(L): float(tot[L]) for L in sorted(tot)},
            "types_by_length_pooled": {str(L): len(by_len[L]) for L in sorted(by_len)}}
    del data
    return by_len, meta, {L: float(n) for L, n in tot.items()}


def dictionary_words() -> set[str]:
    """Reference lexicon for the KCG channel estimate: the system word list (Webster's
    2nd, lemmas) united with the vocabulary of the edited Gutenberg prose already used
    in this project (supplies inflections). Edited text is the closest available proxy
    for intended spellings."""
    words = set(corpus.word_list())
    for name in corpus.GUTENBERG:
        try:
            txt = corpus.load_text(name).lower().replace("_", " ")
        except Exception as e:                               # noqa: BLE001
            log(f"  dictionary: skip {name}: {e}")
            continue
        words |= {t for t in corpus._TOKEN.findall(txt) if t.isascii() and t.isalpha()}
    return words


# --------------------------------------------------------------------------- #
# fits
# --------------------------------------------------------------------------- #

def _model_path(L: int, mode: str) -> Path:
    return MODELS / f"twitter_pooled_L{L}_{mode}.npz"


def _fit_job(args) -> dict:
    L, mode, states, counts, sel = args
    t0 = time.time()
    order = min(2, L)
    info: dict = {"L": L, "mode": mode, "order": order, "N": float(counts.sum()),
                  "types": int(counts.size)}
    if mode == "exact":
        m, d = potts.fit(states, counts, L, order, "exact")
    else:
        m, d = potts.fit(states, counts, L, order, "map", sigma2=sel["best_sigma2"])
        info["cv"] = sel
        info["sigma2"] = sel["best_sigma2"]
    info["diagnostics"] = {k: v for k, v in d.items() if isinstance(v, (int, float, bool, str))}
    info["entropy_bits"] = m.entropy_bits()
    info["seconds"] = time.time() - t0
    MODELS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_model_path(L, mode), cliques=np.array(m.cliques).reshape(len(m.cliques), -1),
                        V=np.stack(m.V), info=json.dumps(info))
    return info


def load_model(L: int, mode: str) -> tuple[potts.Potts, dict]:
    z = np.load(_model_path(L, mode))
    cl = [tuple(int(x) for x in c) for c in z["cliques"]]
    V = [v for v in z["V"]]
    info = json.loads(str(z["info"]))
    return potts.Potts(L, cl, V, sigma2=info.get("sigma2")), info


def _cv_job(args):
    states, counts, L, grid, k, seed = args
    return k, potts.cv_fold(states, counts, L, min(2, L), grid, k, 4, seed)


def _select_prior_parallel(ex, states, counts, L) -> dict:
    """``potts.select_prior`` (4 folds, same seed stream, same grid extension rule)
    with the folds of each grid round run in parallel."""
    seed, n_folds = SEED + L, 4
    grid = list(potts.SIGMA2_START)
    per_fold = [None] * n_folds
    for k, r in ex.map(_cv_job, [(states, counts, L, grid, k, seed) for k in range(n_folds)]):
        per_fold[k] = r
    agg = potts.aggregate_cv(per_fold)
    for _ in range(potts.SIGMA2_MAX_EXTENSIONS):
        if agg["interior"]:
            break
        lo = agg["best_sigma2"] == agg["sigma2"][0]
        new = agg["sigma2"][0] / potts.SIGMA2_STEP if lo else agg["sigma2"][-1] * potts.SIGMA2_STEP
        for k, r in ex.map(_cv_job, [(states, counts, L, [new], k, seed) for k in range(n_folds)]):
            per_fold[k] = per_fold[k] + r
        agg = potts.aggregate_cv(per_fold)
        log(f"    L={L} cv grid {['%.3g' % s for s in agg['sigma2']]} best {agg['best_sigma2']:.3g}")
    return agg


def fit_all(by_len, workers: int, mode: str) -> dict:
    Ls = sorted(set(LENGTHS) | set(CLOSURE)) if mode == "exact" else list(MAP_LENGTHS)
    todo = sorted((L for L in Ls if not _model_path(L, mode).exists()), reverse=True)
    if todo:
        log(f"fitting {mode} {todo}")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            if mode == "exact":                       # longest first balances the workers
                args = [(L, mode, *potts.counts_to_arrays(by_len[L], L), None) for L in todo]
                for info in ex.map(_fit_job, args):
                    log(f"  L={info['L']} exact: {info['seconds']:.0f}s "
                        f"converged={info['diagnostics'].get('converged')}")
            else:
                for L in todo:
                    st, c = potts.counts_to_arrays(by_len[L], L)
                    t0 = time.time()
                    sel = _select_prior_parallel(ex, st, c, L)
                    info = ex.submit(_fit_job, (L, mode, st, c, sel)).result()
                    log(f"  L={L} map: sigma2={sel['best_sigma2']:.3g} interior={sel['interior']} "
                        f"{time.time() - t0:.0f}s converged={info['diagnostics'].get('converged')}")
    return {f"L{L}_{mode}": load_model(L, mode)[1] for L in Ls}


# --------------------------------------------------------------------------- #
# analysis helpers
# --------------------------------------------------------------------------- #

def energy(space: ed.Space, mode: str, p_len: dict[int, float]) -> tuple[np.ndarray, dict]:
    """Grand-canonical energy, built one length block at a time (each block's
    log-weights are freed once written)."""
    lw = {L: (lambda L=L: load_model(L, mode)[0].log_weights()) for L in space.lengths}
    return ed.grand_canonical_energy(space, lw, p_len)


class Tokens:
    """Observed token counts on the space, stored sparsely (sorted global indices)."""

    def __init__(self, space: ed.Space, by_len):
        idx, cnt = [], []
        for L in space.lengths:
            st, c = potts.counts_to_arrays(by_len[L], L)
            idx.append(space.flat(st, L))
            cnt.append(c)
        idx, cnt = np.concatenate(idx), np.concatenate(cnt)
        o = np.argsort(idx)
        self.idx, self.cnt = idx[o], cnt[o]
        self.total = float(self.cnt.sum())
        self.space = space

    def at(self, i: int) -> float:
        k = np.searchsorted(self.idx, i)
        return float(self.cnt[k]) if k < self.idx.size and self.idx[k] == i else 0.0

    def share(self, mask: np.ndarray, L: int) -> float:
        """Token share of length-L tokens on the states where ``mask`` (global) holds."""
        inL = self.space.length_of(self.idx) == L
        return float(self.cnt[inL & mask[self.idx]].sum() / self.cnt[inL].sum())


def minima_summary(space, sub, full, tokens: Tokens) -> dict:
    out = {}
    for L in space.lengths:
        o = space.offset(L)
        sl = slice(o, o + ed.A ** L)
        ms, mf = sub.minima[sl], full.minima[sl]
        lost = np.flatnonzero(ms & ~mf) + o
        kept = np.flatnonzero(mf) + o
        killers = Counter(ed.classify_edit(space.word(i), space.word(full.best_i[i]))
                          for i in lost)
        out[str(L)] = {
            "minima_sub": int(ms.sum()), "minima_full": int(mf.sum()),
            "full_subset_of_sub": bool(not np.any(mf & ~ms)),
            "lost": int(lost.size), "lost_by_killing_move": dict(killers),
            "model_mass_at_minima_sub": float(np.exp2(-full.E[np.flatnonzero(ms) + o]).sum()),
            "model_mass_at_minima_full": float(np.exp2(-full.E[kept]).sum()),
            "token_share_at_minima_sub": tokens.share(sub.minima, L),
            "token_share_at_minima_full": tokens.share(full.minima, L),
            "plateau_terminals_full": int((full.terminal[sl] == np.arange(o, o + ed.A ** L)).sum()
                                          - mf.sum()),
        }
        # every lost well, energy order: [word, lower neighbour, killing move]
        out[str(L)]["all_lost_wells"] = [
            [space.word(i), space.word(full.best_i[i]),
             ed.classify_edit(space.word(i), space.word(full.best_i[i]))]
            for i in lost[np.argsort(full.E[lost], kind="stable")]]
        top_lost = lost[np.argsort(full.E[lost], kind="stable")][:TOP]
        out[str(L)]["top_lost_wells"] = [
            {"word": space.word(i), "E_bits": float(full.E[i]), "tokens": tokens.at(i),
             "lower_neighbour": space.word(full.best_i[i]),
             "neighbour_E_bits": full.best_e(i),
             "move": ed.classify_edit(space.word(i), space.word(full.best_i[i]))}
            for i in top_lost]
        top_kept = kept[np.argsort(full.E[kept], kind="stable")][:TOP]
        out[str(L)]["top_wells_full"] = [
            {"word": space.word(i), "E_bits": float(full.E[i]), "tokens": tokens.at(i)}
            for i in top_kept]
    return out


def basin_summary(space, ls, tokens: Tokens) -> dict:
    b = ed.basins(ls, (tokens.idx, tokens.cnt))
    order = np.argsort(-b["model_mass"])
    q = lambda x: {f"q{int(k * 100)}": float(np.quantile(x, k)) for k in (0.5, 0.9, 0.99)}  # noqa: E731
    lens = space.length_of(b["index"])
    return {
        "n_basins": int(b["index"].size),
        "states_quantiles": q(b["states"]), "model_mass_quantiles": q(b["model_mass"]),
        "model_mass_in_basins": float(b["model_mass"].sum()),
        "token_mass_in_basins": float(b["token_mass"].sum() / tokens.total),
        "basins_by_length": {str(L): int((lens == L).sum()) for L in space.lengths},
        "top_by_model_mass": [
            {"word": space.word(b["index"][j]), "states": int(b["states"][j]),
             "model_mass": float(b["model_mass"][j]),
             "token_share": float(b["token_mass"][j] / tokens.total)}
            for j in order[:TOP]],
    }


def basin_comparison(space, sub, full, tokens: Tokens, words) -> list[dict]:
    """Basin of the same well under both move sets (state count, token share)."""
    tk = (tokens.idx, tokens.cnt)
    bs, bf = ed.basins(sub, tk), ed.basins(full, tk)
    ds = {int(i): j for j, i in enumerate(bs["index"])}
    df = {int(i): j for j, i in enumerate(bf["index"])}
    out = []
    for w in (w for w in words if len(w) in space.lengths):
        i = space.index(w)
        r = {"word": w, "tokens": tokens.at(i)}
        for name, b, d in (("sub", bs, ds), ("full", bf, df)):
            j = d.get(i)
            r[name] = None if j is None else {
                "states": int(b["states"][j]), "token_share": float(b["token_mass"][j] / tokens.total),
                "model_mass": float(b["model_mass"][j])}
        out.append(r)
    return out


def upper_boundary_check(space, full, by_len, N_all) -> dict:
    """L = 5 wells are minima only among modelled moves; their insertion neighbours at
    L = 6 are not modelled. Proxy: an observed six-letter word w6 whose empirical token
    probability n(w6)/N exceeds the model probability of its five-letter deletion w5
    would make w5 not a minimum if the L = 6 model reproduced frequent words."""
    Lmax = max(space.lengths)
    if Lmax + 1 not in by_len:
        return {}
    o = space.offset(Lmax)
    st, c = potts.counts_to_arrays(by_len[Lmax + 1], Lmax + 1)
    p6 = c / N_all
    f = np.ravel_multi_index(st.T, (ed.A,) * (Lmax + 1)).astype(np.int64)
    mins = np.flatnonzero(full.minima[o:o + ed.A ** Lmax])
    worst = np.zeros(mins.size)
    for p in range(Lmax + 1):
        s = ed.A ** (Lmax - p)
        g = (f // (s * ed.A)) * s + f % s
        k = np.minimum(np.searchsorted(mins, g), max(mins.size - 1, 0))
        hit = mins[k] == g if mins.size else np.zeros(g.size, bool)
        np.maximum.at(worst, k[hit], p6[hit])
    pw = np.exp2(-full.E[o + mins])
    risk = worst > pw
    return {"L": Lmax, "minima": int(mins.size), "at_risk_by_empirical_L6": int(risk.sum()),
            "at_risk_fraction": float(risk.mean()) if mins.size else None,
            "at_risk_model_mass_fraction": float(pw[risk].sum() / pw.sum()) if mins.size else None}


def recovery_table(space, lss: dict, by_len, dictionary: set[str] | None) -> dict:
    out = {}
    for L in space.lengths:
        st, c = potts.counts_to_arrays(by_len[L], L)
        if dictionary is not None:
            words = ["".join(ed.ALPHABET[k] for k in s) for s in st]
            keep = np.array([w in dictionary for w in words])
            st, c = st[keep], c[keep]
        row = {"types": int(c.size), "tokens": float(c.sum())}
        for kind in ed.ERRORS:
            row[kind] = {}
            for name, ls in lss.items():
                r = ed.recovery(ls, st, L, kind)
                v, w = (None, 0.0) if r is None else ed.token_weighted(r, c)
                row[kind][name] = v
                row[kind]["tokens_with_events"] = w
        out[str(L)] = row
    # pooled over lengths, token-weighted over words with at least one event of the
    # type inside the modelled space
    pooled = {}
    for kind in ed.ERRORS:
        pooled[kind] = {}
        for name in lss:
            num = den = 0.0
            for L in space.lengths:
                v = out[str(L)][kind][name]
                if v is not None:
                    w = out[str(L)][kind]["tokens_with_events"]
                    num += v * w
                    den += w
            pooled[kind][name] = num / den if den else None
    out["pooled"] = pooled
    return out


# Model-weighted move statistics enumerate every state of a length; done up to
# 26^4 states (a compute convention; 26^5 x 156 insertion events is not affordable).
MODEL_WEIGHTED_MAX_L = 4


def move_delta_summary(space, E, by_len) -> dict:
    """dE = E(w') - E(w) of single moves from (a) observed words, token-weighted, and
    (b) model-distributed words P(w | L) for L <= MODEL_WEIGHTED_MAX_L."""
    out = {}
    for L in space.lengths:
        st, c = potts.counts_to_arrays(by_len[L], L)
        row = {"observed_tokens": {mv: ed.move_delta(space, E, st, L, mv, c) for mv in ed.MOVES}}
        if L <= MODEL_WEIGHTED_MAX_L:
            f = np.arange(ed.A ** L)
            sa = np.stack(np.unravel_index(f, (ed.A,) * L), axis=1).astype(np.int64)
            p = np.exp2(-space.block(E, L).ravel())
            row["model_P_w_given_L"] = {mv: ed.move_delta(space, E, sa, L, mv, p)
                                        for mv in ed.MOVES}
        out[str(L)] = row
    return out


def random_corruption(w: str, kind: str, rng) -> str:
    L = len(w)
    if kind == "sub":
        p = rng.integers(L)
        a = ed.ALPHABET[(ed.ALPHABET.index(w[p]) + 1 + rng.integers(ed.A - 1)) % ed.A]
        return w[:p] + a + w[p + 1:]
    if kind == "trans":
        ps = [p for p in range(L - 1) if w[p] != w[p + 1]]
        p = ps[rng.integers(len(ps))]
        return w[:p] + w[p + 1] + w[p] + w[p + 2:]
    if kind == "ins":
        p = rng.integers(L + 1)
        return w[:p] + ed.ALPHABET[rng.integers(ed.A)] + w[p:]
    p = rng.integers(L)
    return w[:p] + w[p + 1:]


def channel_study(space, E, sub, full, by_len, dictionary, rng) -> dict:
    dmask = np.zeros(space.size, dtype=bool)
    for w in dictionary:
        if len(w) in space.lengths:
            dmask[space.index(w)] = True
    typos_type, typos_tok = {}, {}
    for L in space.lengths:
        st, c = potts.counts_to_arrays(by_len[L], L)
        nd = ~dmask[space.flat(st, L)]
        typos_type[L] = (st[nd], np.ones(int(nd.sum())))
        typos_tok[L] = (st[nd], c[nd])
    # Interior lengths have every source length (L-1, L, L+1) modelled, so no error
    # type is censored by the boundary. Primary estimate: token-weighted EM on the
    # retained sample (non-lexicon strings of the interior length), conditioned on
    # retention (``edits.selection_probabilities``): the retained sample admits each
    # type with a different probability (insertions come from L-1 sources, deletions
    # from L+1 sources, and outputs that are lexicon words are never retained).
    # Uncorrected assignment shares are reported only as a description of that sample.
    interior = [L for L in space.lengths if L - 1 in space.lengths and L + 1 in space.lengths]
    sel = ed.selection_probabilities(space, E, dmask, interior)
    em_k = ed.em_channel(space, E, {L: typos_tok[L] for L in interior}, dmask, selection=sel)
    em_t = ed.em_channel(space, E, {L: typos_type[L] for L in interior}, dmask)
    n_int = sum(float(sum(by_len[L].values())) for L in interior)
    # The token share of retained near-misses is NOT an error rate or a bound on it:
    # it counts slang/names as errors (false positives) and misses real-word errors,
    # multi-edit errors and errors without a lexicon neighbour (false negatives). It is
    # used as an uncalibrated working value of eps; decoder results are reported on a
    # half-decade grid around it (a convention: the same grid spacing as the sigma^2
    # search) to show the sensitivity.
    near_miss = em_k["weight_used"] / n_int
    eps_grid = [near_miss * 10.0 ** (k / 2) for k in range(-2, 3)]
    rho = em_k["rho"]
    log(f"  channel rho(selection-corrected, L={interior})={rho} near-miss={near_miss:.4f}")

    words, counts, lens = [], [], []
    for L in space.lengths:
        for w, n in by_len[L].items():
            words.append(w)
            counts.append(n)
            lens.append(L)
    words, counts, lens = np.array(words), np.array(counts, float), np.array(lens)
    in_lex = np.array([w in dictionary for w in words])
    keys = ["sub_descent", "full_descent"] + [f"decoder_eps{k}" for k in range(len(eps_grid))]

    def summarise(res):
        r = {k: np.array(v, float) for k, v in res.items()}
        return {k: {"rate": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(v.size))}
                for k, v in r.items()}

    # recovery of synthetic single errors, paired on the same corrupted samples
    mc = {}
    for kind in ed.ERRORS:
        ok = np.ones(words.size, dtype=bool)
        if kind == "ins":
            ok &= np.isin(lens, [L for L in space.lengths if L + 1 in space.lengths])
        if kind == "del":
            ok &= np.isin(lens, [L for L in space.lengths if L - 1 in space.lengths])
        if kind == "trans":
            ok &= np.array([any(a != b for a, b in zip(w, w[1:])) for w in words])
        pr = np.where(ok, counts, 0.0)
        pick = rng.choice(words.size, size=K_MC, p=pr / pr.sum())
        res = {k: [] for k in keys}
        for j in pick:
            w = str(words[j])
            o = random_corruption(w, kind, rng)
            i = space.index(o)
            res["sub_descent"].append(sub.terminal[i] == space.index(w))
            res["full_descent"].append(full.terminal[i] == space.index(w))
            for k, (c, _) in enumerate(ed.decode_many(o, space, E, eps_grid, rho)):
                res[f"decoder_eps{k}"].append(c == w)
        mc[kind] = summarise(res)
    # change rate on uncorrupted corpus tokens: every observed token, typos included,
    # and the subset that are lexicon words
    pick = rng.choice(words.size, size=K_MC, p=counts / counts.sum())
    fa = {k: [] for k in keys}
    lex = []
    for j in pick:
        w = str(words[j])
        i = space.index(w)
        lex.append(bool(in_lex[j]))
        fa["sub_descent"].append(sub.terminal[i] != i)
        fa["full_descent"].append(full.terminal[i] != i)
        for k, (c, _) in enumerate(ed.decode_many(w, space, E, eps_grid, rho)):
            fa[f"decoder_eps{k}"].append(c != w)
    lex = np.array(lex)
    mc["change_rate_observed_tokens"] = summarise(fa)
    mc["change_rate_observed_lexicon_tokens"] = summarise(
        {k: list(np.array(v)[lex]) for k, v in fa.items()})
    mc["lexicon_share_of_sampled_tokens"] = float(lex.mean())
    return {"dictionary_size": len(dictionary), "em_lengths": interior,
            "selection_probabilities": sel,
            "em_token_weighted_selection_corrected": em_k,
            "em_type_weighted_assignment_shares_uncorrected": em_t,
            "near_miss_token_fraction": near_miss, "eps_grid": eps_grid,
            "eps_working_index": 2, "tokens_interior": n_int,
            "monte_carlo": mc, "K": K_MC,
            "probes": {w: [dict(zip(("decoded", "posterior"), r))
                           for r in ed.decode_many(w, space, E, eps_grid, rho)]
                       for w in PROBES if len(w) in space.lengths}}


def recovery_and_paths(space, sub, full, tokens, by_len, dictionary, E) -> dict:
    lss = {"sub": sub, "full": full}
    return {
        "recovery_all_tokens": recovery_table(space, lss, by_len, None),
        "recovery_dictionary_tokens": recovery_table(space, lss, by_len, dictionary),
        "probes": {w: {"tokens": tokens.at(space.index(w)),
                       "sub": sub.path(w), "full": full.path(w)}
                   for w in PROBES if len(w) in space.lengths},
    }


def landscape_block(space, E, by_len, N_all, dictionary, probe_words) -> tuple[dict, tuple]:
    tokens = Tokens(space, by_len)
    log(f"  landscapes on {space.size:,} states")
    sub = ed.landscape(space, E, ("sub",))
    full = ed.landscape(space, E)
    res = {
        "states": space.size,
        "minima": minima_summary(space, sub, full, tokens),
        "basins_sub": basin_summary(space, sub, tokens),
        "basins_full": basin_summary(space, full, tokens),
        "basin_comparison": basin_comparison(space, sub, full, tokens, probe_words),
        "upper_boundary": upper_boundary_check(space, full, by_len, N_all),
        "widom": [ed.widom_insertion(space, E, L) for L in space.lengths
                  if L + 1 in space.lengths],
        # token-weighted mean energy of observed words per length. A difference of
        # these length-conditioned means is NOT the energy change of a move (deletion
        # parents are not distributed as P(w | L-1)); see move_delta_bits.
        "mean_token_energy_bits": {
            str(L): float((tokens.cnt * E[tokens.idx])[space.length_of(tokens.idx) == L].sum()
                          / tokens.cnt[space.length_of(tokens.idx) == L].sum())
            for L in space.lengths},
        "move_delta_bits": move_delta_summary(space, E, by_len),
    }
    log("  recovery")
    res.update(recovery_and_paths(space, sub, full, tokens, by_len, dictionary, E))
    return res, (sub, full, tokens)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    log("loading counts")
    by_len, meta, tot = pooled_counts()
    N_all = meta["tokens_all_lengths"]
    p_len = {L: tot[L] / N_all for L in tot}
    fits = fit_all(by_len, args.workers, "exact")
    log("dictionary")
    dictionary = dictionary_words()
    # wells compared across move sets: the most frequent token types of each length
    probe_words = []
    for L in LENGTHS:
        probe_words += [w for w, _ in by_len[L].most_common(TOP // len(LENGTHS) + 1)]
    out = {"meta": meta, "p_len": {str(L): p_len[L] for L in sorted(p_len)}, "fits": fits,
           "conventions": {"lengths": LENGTHS, "closure_lengths": CLOSURE, "seed": SEED,
                           "K_mc": K_MC, "top_rows_listed": TOP,
                           "tie_rule": "lowest energy; ties sub<trans<del<ins, then position, then letter",
                           "moves": ed.MOVES, "errors": ed.ERRORS}}

    def save():
        OUT.write_text(json.dumps(out, indent=1, default=float))

    space = ed.Space(LENGTHS)
    log("model exact")
    E, mu = energy(space, "exact", p_len)
    res, (sub, full, _) = landscape_block(space, E, by_len, N_all, dictionary, probe_words)
    res["chemical_potentials"] = {str(L): v for L, v in mu.items()}
    log("  channel")
    res["channel"] = channel_study(space, E, sub, full, by_len, dictionary, rng)
    out["exact"] = res
    del sub, full, E
    gc.collect()
    save()

    log("closure lengths 1-5 (exact)")
    cs = ed.Space(CLOSURE)
    E, mu = energy(cs, "exact", p_len)
    res, _ = landscape_block(cs, E, by_len, N_all, dictionary, probe_words)
    res["chemical_potentials"] = {str(L): v for L, v in mu.items()}
    out["closure_exact"] = res
    del E, res
    gc.collect()
    save()

    out["fits"].update(fit_all(by_len, args.workers, "map"))
    # like-for-like comparison on the L = 3, 4 space: exact vs map couplings
    s34 = ed.Space(MAP_LENGTHS)
    out["map_check_L34"] = {}
    for mode in ("exact", "map"):
        log(f"map check L={MAP_LENGTHS}: {mode}")
        E, mu = energy(s34, mode, p_len)
        res, _ = landscape_block(s34, E, by_len, N_all, dictionary, probe_words)
        res["chemical_potentials"] = {str(L): v for L, v in mu.items()}
        out["map_check_L34"][mode] = res
        del E
        gc.collect()
    save()
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
