import itertools

import numpy as np
import pytest

from langfit import edits as ed
from langfit import maxent as me

AL = ed.ALPHABET


def _neighbours(w, lengths, moves=ed.MOVES):
    """Brute-force single-edit neighbours of w inside the modelled lengths."""
    out = set()
    L = len(w)
    if "sub" in moves:
        out |= {w[:p] + c + w[p + 1:] for p in range(L) for c in AL if c != w[p]}
    if "trans" in moves:
        out |= {w[:p] + w[p + 1] + w[p] + w[p + 2:] for p in range(L - 1) if w[p] != w[p + 1]}
    if "del" in moves and L - 1 in lengths:
        out |= {w[:p] + w[p + 1:] for p in range(L)}
    if "ins" in moves and L + 1 in lengths:
        out |= {w[:p] + c + w[p:] for p in range(L + 1) for c in AL}
    return out


@pytest.fixture(scope="module")
def small():
    sp = ed.Space((1, 2, 3))
    rng = np.random.default_rng(0)
    E = rng.normal(size=sp.size) * 3.0
    E[rng.random(sp.size) < 0.05] = np.inf
    return sp, E


def test_space_indexing(small):
    sp, _ = small
    assert sp.size == 26 + 26 ** 2 + 26 ** 3
    for w in ["a", "z", "ab", "zz", "abc", "zzz", "qrs"]:
        i = sp.index(w)
        assert sp.word(i) == w and sp.length_of([i])[0] == len(w)
    assert sp.index("abc") - sp.offset(3) == me.ravel("abc")


@pytest.mark.parametrize("moves", [("sub",), ("sub", "trans"), ed.MOVES])
def test_best_neighbour_matches_brute_force(small, moves):
    sp, E = small
    be, bi = ed.best_neighbour(sp, E, moves)
    rng = np.random.default_rng(1)
    for i in rng.choice(sp.size, 400, replace=False):
        w = sp.word(i)
        nb = _neighbours(w, sp.lengths, moves)
        es = {v: E[sp.index(v)] for v in nb}
        ref = min(es.values()) if es else np.inf
        assert be[i] == ref
        if np.isfinite(ref):
            assert es[sp.word(bi[i])] == ref


def test_substitution_only_reproduces_maxent(small):
    sp, E = small
    ls = ed.landscape(sp, E, ("sub",))
    for L in sp.lengths:
        EL = sp.block(E, L)
        mins = sp.block(ls.minima, L)
        assert np.array_equal(mins, me.local_minima(EL))
        term = sp.block(ls.terminal, L).ravel() - sp.offset(L)
        assert np.array_equal(term, me.descend(EL))


def test_full_minima_subset_and_paths(small):
    sp, E = small
    sub = ed.landscape(sp, E, ("sub",))
    full = ed.landscape(sp, E)
    assert not np.any(full.minima & ~sub.minima)       # more moves -> fewer wells
    for i in np.flatnonzero(full.minima)[:50]:
        w = sp.word(i)
        assert all(E[sp.index(v)] > E[i] for v in _neighbours(w, sp.lengths))
    # descent path agrees with terminal map and strictly decreases
    for i in np.random.default_rng(2).choice(sp.size, 50, replace=False):
        p = full.path(sp.word(i))
        assert sp.index(p[-1]["word"]) == full.terminal[i]
        es = [s["E_bits"] for s in p]
        assert all(b < a for a, b in zip(es, es[1:]))
        assert all(s["move"] in ed.MOVES for s in p[1:])


def test_basins_partition(small):
    sp, E = small
    ls = ed.landscape(sp, E)
    b = ed.basins(ls)
    plateau = (ls.terminal == np.arange(sp.size)) & ~ls.minima
    assert b["states"].sum() + np.isin(ls.terminal, np.flatnonzero(plateau)).sum() == sp.size
    assert np.all(b["states"] >= 1)


def test_corruptions_and_recovery_match_maxent():
    sp = ed.Space((2, 3, 4))
    rng = np.random.default_rng(3)
    E = rng.normal(size=sp.size)
    ls = ed.landscape(sp, E, ("sub",))
    words = ["abc", "the", "aab", "zzz"]
    states = np.array([[ord(c) - 97 for c in w] for w in words])
    rec = ed.recovery(ls, states, 3, "sub")
    wc = me.WordCounts(states, np.ones(len(words)), 3)
    assert np.allclose(rec, me.substitution_recovery(sp.block(E, 3), wc))
    # event sets equal brute force, per type
    for kind, gen in [("sub", lambda w: [w[:p] + c + w[p + 1:] for p in range(3) for c in AL if c != w[p]]),
                      ("trans", lambda w: [w[:p] + w[p + 1] + w[p] + w[p + 2:] for p in range(2) if w[p] != w[p + 1]]),
                      ("ins", lambda w: [w[:p] + c + w[p:] for p in range(4) for c in AL]),
                      ("del", lambda w: [w[:p] + w[p + 1:] for p in range(3)])]:
        got = [[] for _ in words]
        for tgt, ok in ed.corruptions(sp, states, 3, kind):
            for j in range(len(words)):
                if ok[j]:
                    got[j].append(sp.word(tgt[j]))
        for j, w in enumerate(words):
            assert sorted(got[j]) == sorted(gen(w)), (kind, w)
            assert len(got[j]) == ed.n_events(w, kind)
    assert ed.corruptions(sp, np.array([[0, 1, 2, 3]]), 4, "ins") is None   # L=5 not modelled
    full = ed.landscape(sp, E)
    r = ed.recovery(full, states, 3, "trans")
    assert np.isnan(r[3]) and np.all((r[:3] >= 0) & (r[:3] <= 1))


def test_preimages_multiplicity():
    pre = ed.preimages("wil")
    assert pre["will"] == ("del", 2)            # 'will' lost one of its two l's
    assert pre["wi"] == ("ins", 1)
    assert pre["wli"] == ("trans", 1)
    assert pre["wit"] == ("sub", 1)
    # consistency with forward events: m(c -> o) = number of events of c giving o
    for o in ["wil", "aab", "taht"]:
        for c, (k, m) in ed.preimages(o).items():
            sp = ed.Space(tuple(range(1, 7)))
            st = np.array([[ord(x) - 97 for x in c]])
            it = ed.corruptions(sp, st, len(c), k)
            n = sum(int(ok[0] and tgt[0] == sp.index(o)) for tgt, ok in it)
            assert n == m, (o, c, k)


def test_grand_canonical_energy():
    sp = ed.Space((2, 3))
    rng = np.random.default_rng(4)
    lw = {L: rng.normal(size=(26,) * L) + 7.0 for L in sp.lengths}
    pL = {2: 0.2, 3: 0.3}
    E, mu = ed.grand_canonical_energy(sp, {2: lw[2], 3: lambda: lw[3]}, pL)
    for L in sp.lengths:
        assert np.isclose(np.exp2(-sp.block(E, L)).sum(), pL[L])
    assert np.isclose(mu[3]["mu_bits"], np.log2(0.3) - np.log2(np.exp(lw[3]).sum()))
    w = "ab"
    assert np.isclose(E[sp.index(w)], -lw[2][0, 1] / np.log(2) - mu[2]["mu_bits"])


def test_em_recovers_planted_rates():
    sp = ed.Space((3, 4, 5))
    rng = np.random.default_rng(5)
    # a source of 40 random 4-letter "dictionary" words with Zipf weights
    words = sorted({"".join(rng.choice(list(AL), 4)) for _ in range(40)})
    p = 1.0 / np.arange(1, len(words) + 1)
    p /= p.sum()
    E = np.full(sp.size, 60.0)
    dic = np.zeros(sp.size, dtype=bool)
    for w, q in zip(words, p):
        E[sp.index(w)] = -np.log2(q)
        dic[sp.index(w)] = True
    rho = {"sub": 0.5, "trans": 0.1, "ins": 0.25, "del": 0.15}
    counts = {}
    for _ in range(20000):
        c = words[rng.choice(len(words), p=p)]
        t = rng.choice(ed.ERRORS, p=[rho[k] for k in ed.ERRORS])
        L = len(c)
        if t == "sub":
            q = rng.integers(L); a = AL[(AL.index(c[q]) + 1 + rng.integers(25)) % 26]
            o = c[:q] + a + c[q + 1:]
        elif t == "trans":
            q = rng.integers(L - 1); o = c[:q] + c[q + 1] + c[q] + c[q + 2:]
        elif t == "ins":
            q = rng.integers(L + 1); o = c[:q] + AL[rng.integers(26)] + c[q:]
        else:
            q = rng.integers(L); o = c[:q] + c[q + 1:]
        if o != c and o not in words:
            counts[o] = counts.get(o, 0) + 1
    typos = {}
    for L in (3, 4, 5):
        ws = [o for o in counts if len(o) == L]
        typos[L] = (np.array([[ord(x) - 97 for x in o] for o in ws]).reshape(-1, L),
                    np.array([counts[o] for o in ws], dtype=float))
    fit = ed.em_channel(sp, E, typos, dic)
    assert fit["converged"]
    for k in ed.ERRORS:
        assert abs(fit["rho"][k] - rho[k]) < 0.03, fit["rho"]


def test_decode_prefers_transposition_source():
    sp = ed.Space((3, 4, 5))
    E = np.full(sp.size, 40.0)
    E[sp.index("what")] = 5.0
    E[sp.index("waht")] = 30.0
    rho = {"sub": 0.5, "trans": 0.1, "ins": 0.2, "del": 0.2}
    c, post = ed.decode("waht", sp, E, eps=0.05, rho=rho)
    assert c == "what" and 0.5 < post <= 1.0
    c, _ = ed.decode("what", sp, E, eps=0.05, rho=rho)
    assert c == "what"


def test_widom_identity():
    sp = ed.Space((2, 3))
    rng = np.random.default_rng(6)
    lw = {L: rng.normal(size=(26,) * L) for L in sp.lengths}
    E, _ = ed.grand_canonical_energy(sp, lw, {2: 0.4, 3: 0.6})
    w = ed.widom_insertion(sp, E, 2)
    assert np.isclose(w["mean_insertion_factor"], w["identity_rhs"])
    assert np.isclose(w["identity_rhs"], 3 * 0.6 / 0.4)
    E[sp.index("ab")] = np.inf                   # forbidden parent -> deficit
    w = ed.widom_insertion(sp, E, 2)
    assert w["mean_insertion_factor"] < w["identity_rhs"]


def test_token_weighted_skips_types_without_events():
    v, w = ed.token_weighted(np.array([1.0, np.nan, 0.0]), np.array([3.0, 5.0, 1.0]))
    assert v == 0.75 and w == 4.0
    assert ed.token_weighted(np.array([np.nan]), np.array([2.0])) == (None, 0.0)


def test_em_selection_correction_reviewer_case():
    """Sources P(ab)=0.8, P(cde)=0.1, P(fghi)=0.1, planted rates 0.25 each, only
    non-lexicon outputs of the interior length 3 retained, exact expected counts:
    assignment shares are biased toward insertions (from the frequent 2-letter source);
    the retention-conditioned EM recovers the planted rates."""
    sp = ed.Space((2, 3, 4))
    src = {"ab": 0.8, "cde": 0.1, "fghi": 0.1}
    E = np.full(sp.size, 30.0)
    dic = np.zeros(sp.size, dtype=bool)
    for c, p in src.items():
        E[sp.index(c)] = -np.log2(p)
        dic[sp.index(c)] = True
    rho = dict.fromkeys(ed.ERRORS, 0.25)
    w = {}
    for c, p in src.items():
        L = len(c)
        evs = {"sub": [c[:q] + a + c[q + 1:] for q in range(L) for a in AL if a != c[q]],
               "trans": [c[:q] + c[q + 1] + c[q] + c[q + 2:] for q in range(L - 1) if c[q] != c[q + 1]],
               "ins": [c[:q] + a + c[q:] for q in range(L + 1) for a in AL],
               "del": [c[:q] + c[q + 1:] for q in range(L)]}
        for t, outs in evs.items():
            for o in outs:
                if len(o) == 3 and o not in src:
                    w[o] = w.get(o, 0.0) + p * rho[t] / len(outs)
    ws = sorted(w)
    typos = {3: (np.array([[ord(x) - 97 for x in o] for o in ws]), np.array([w[o] for o in ws]))}
    biased = ed.em_channel(sp, E, typos, dic)
    assert abs(biased["rho"]["ins"] - 0.8 / 1.1) < 1e-6
    sel = ed.selection_probabilities(sp, E, dic, [3])
    fit = ed.em_channel(sp, E, typos, dic, selection=sel)
    assert fit["converged"]
    for t in ed.ERRORS:
        assert abs(fit["rho"][t] - 0.25) < 1e-6, fit["rho"]
    with pytest.raises(ValueError):
        ed.selection_probabilities(sp, E, dic, [4])


@pytest.mark.parametrize("c", ["aaa", "ab", "a", "what", "aab"])
def test_channel_rows_sum_to_one(c):
    eps, rho = 0.1, {"sub": 0.4, "trans": 0.1, "ins": 0.3, "del": 0.2}
    L = len(c)
    outs = set()
    outs |= {c[:q] + a + c[q + 1:] for q in range(L) for a in AL if a != c[q]}
    outs |= {c[:q] + c[q + 1] + c[q] + c[q + 2:] for q in range(L - 1) if c[q] != c[q + 1]}
    outs |= {c[:q] + a + c[q:] for q in range(L + 1) for a in AL}
    outs |= {c[:q] + c[q + 1:] for q in range(L)}
    total = np.exp2(ed.channel_log2(c, c, None, 1, eps, rho))
    for o in outs:
        k, m = ed.preimages(o)[c]
        total += np.exp2(ed.channel_log2(o, c, k, m, eps, rho))
    assert np.isclose(total, 1.0)


def test_decode_many_matches_decode():
    sp = ed.Space((3, 4, 5))
    rng = np.random.default_rng(7)
    E = rng.normal(size=sp.size) * 4 + 30
    E[sp.index("what")] = 5.0
    rho = {"sub": 0.5, "trans": 0.1, "ins": 0.2, "del": 0.2}
    grid = [1e-3, 0.05, 0.5]
    many = ed.decode_many("waht", sp, E, grid, rho)
    for eps, r in zip(grid, many):
        assert r == ed.decode("waht", sp, E, eps, rho)


def test_move_delta_brute_force():
    sp = ed.Space((2, 3, 4))
    rng = np.random.default_rng(8)
    E = rng.normal(size=sp.size)
    E[rng.random(sp.size) < 0.1] = np.inf
    words = ["abc", "zzq", "the"]
    st = np.array([[ord(x) - 97 for x in w] for w in words])
    wt = np.array([1.0, 2.0, 3.0])
    for mv in ("sub", "trans", "del", "ins"):
        r = ed.move_delta(sp, E, st, 3, mv, wt)
        nbrs = {"sub": lambda w: [w[:p] + a + w[p + 1:] for p in range(3) for a in AL if a != w[p]],
                "trans": lambda w: [w[:p] + w[p + 1] + w[p] + w[p + 2:] for p in range(2) if w[p] != w[p + 1]],
                "del": lambda w: [w[:p] + w[p + 1:] for p in range(3)],
                "ins": lambda w: [w[:p] + a + w[p:] for p in range(4) for a in AL]}[mv]
        num = den = 0.0
        for w, x in zip(words, wt):
            if not np.isfinite(E[sp.index(w)]):
                continue
            d = np.array([E[sp.index(v)] - E[sp.index(w)] for v in nbrs(w)])
            num += x * (d < 0).mean()
            den += x
        assert np.isclose(r["frac_downhill"], num / den)
