"""Tests of ``langfit.nm_landscape``: the grand-canonical variable-length landscape.

Each check is against an independent exact computation on a grid small enough to
enumerate (L <= 3), or a closed-form property of the definitions.
"""

from collections import Counter

import numpy as np
import pytest
from scipy.special import logsumexp

from langfit import nm_landscape as nm
from langfit import edits, longwords, potts

A = nm.A
LN2 = np.log(2.0)


# --------------------------------------------------------------------------- #
# length prior
# --------------------------------------------------------------------------- #

def test_length_prior_empty_is_uniform():
    p = nm.length_prior({})
    assert p.shape == (8,)
    assert np.allclose(p, 1.0 / 8.0)


def test_length_prior_normalises_and_pulls_toward_uniform():
    p = nm.length_prior({2: 900.0, 4: 100.0})
    assert np.isclose(p.sum(), 1.0)
    # posterior mean lies between the empirical share and the prior 1/8
    emp = (900.0 + 100.0 + 8 * 1.0)          # denominator
    assert np.isclose(p[1], (900.0 + 1.0) / emp)


# --------------------------------------------------------------------------- #
# bigram profile and coordinates
# --------------------------------------------------------------------------- #

def test_bigram_ids_boundary_marking():
    # "ab" -> bigrams "#a", "ab", "b#"
    st = np.array([[0, 1]])
    ids = nm.bigram_ids(st)[0]
    B = nm.BOUNDARY_IDX
    expect = [B * nm.S + 0, 0 * nm.S + 1, 1 * nm.S + B]
    assert ids.tolist() == expect


def test_coords_are_sums_of_bigram_scores():
    rng = np.random.default_rng(0)
    U = rng.normal(size=(2, nm.S, nm.S))
    st = np.array([[2, 1, 5], [0, 3, 3]])
    x, y = nm.coords_of_states(U, st)
    ids = nm.bigram_ids(st)
    assert np.allclose(x, U[0].ravel()[ids].sum(axis=1))
    assert np.allclose(y, U[1].ravel()[ids].sum(axis=1))


def test_enumerate_states_counts():
    for L in (1, 2, 3):
        s = nm.enumerate_states(L)
        assert s.shape == (A ** L, L)
        assert len(np.unique(s, axis=0)) == A ** L


# --------------------------------------------------------------------------- #
# uniform model and grand-canonical energy
# --------------------------------------------------------------------------- #

def test_uniform_length_one():
    m = nm._fit_uniform(1)
    assert np.isclose(m.logZ, np.log(26.0))
    assert np.isclose(m.entropy_bits, np.log2(26.0))
    assert np.allclose(m.log2_prob(np.arange(26).reshape(-1, 1)), -np.log2(26.0))


# --------------------------------------------------------------------------- #
# bin energy combination (closed-form properties)
# --------------------------------------------------------------------------- #

def test_empty_frame_is_flat():
    n_cells = 9
    # uniform model = prior: every cell's model mass equals its prior mass per length
    p_bin = {L: np.ones(n_cells) / (A ** L) for L in nm.LENGTHS}
    p_prior = {L: np.ones(n_cells) / (A ** L) for L in nm.LENGTHS}
    se = {L: np.zeros(n_cells) for L in nm.LENGTHS}
    ess = {L: None for L in nm.LENGTHS}
    pL = nm.length_prior({})                        # uniform prior = empty frame
    r = nm.combine_bin_energy(pL, p_bin, se, p_prior, se, ess, n_cells)
    assert np.allclose(r["z_bin"], 0.0)
    assert np.all(np.isfinite(r["z_bin"]))


def test_z_bin_is_log_prior_over_model():
    # z_bin = log2 P_prior - log2 P_bin, closed form, no |bin| needed
    n_cells = 1
    p_bin = {1: np.array([0.5])}
    p_prior = {1: np.array([2.0])}                  # arbitrary (not 1/26)
    se = {1: np.zeros(1)}
    ess = {1: None}
    pL = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])       # only length 1
    r = nm.combine_bin_energy(pL, p_bin, se, p_prior, se, ess, n_cells)
    pi = 1.0 / 8.0
    assert np.isclose(r["z_bin"][0], np.log2(pi * 2.0) - np.log2(0.5))


def test_combine_bin_energy_gap_and_onesided():
    n_cells = 2
    p_bin = {1: np.array([0.5, 0.1])}
    p_prior = {1: np.array([1.0 / A, 0.0])}         # cell 1 has no prior mass
    se = {1: np.zeros(2)}
    ess = {1: None}
    pL = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    r = nm.combine_bin_energy(pL, p_bin, se, p_prior, se, ess, n_cells)
    assert np.isfinite(r["z_bin"][0])               # both masses positive
    assert np.isnan(r["z_bin"][1])                  # one-sided: no invented z
    assert r["one_sided"][1]
    assert not r["gap"][0]


def test_low_ess_flag_marks_l8_dominated_cells():
    n_cells = 1
    p_bin = {8: np.array([0.5])}
    p_prior = {8: np.array([1.0 / (26 ** 8)])}
    se = {8: np.zeros(1)}
    ess = {8: 14.0}                                  # the L=8 sampler's ESS ~ 14 < 100
    pL = np.array([0, 0, 0, 0, 0, 0, 0, 1.0])
    r = nm.combine_bin_energy(pL, p_bin, se, p_prior, se, ess, n_cells)
    assert r["low_ess"][0]                           # all mass from a low-ESS length


# --------------------------------------------------------------------------- #
# importance sampling is unbiased
# --------------------------------------------------------------------------- #

def test_importance_sampling_matches_exact_bin_probability():
    rng = np.random.default_rng(1)
    L = 2
    # a small non-trivial model (dense, L <= 5)
    V = [np.array([[1.5, 0.0, 0.0] + [0.0] * 23,
                   [0.0, -0.5, 0.0] + [0.0] * 23,
                   [0.0, 0.0, 0.3] + [0.0] * 23] + [[0.0] * 26 for _ in range(23)])]
    # make it 26x26 properly
    Vv = np.zeros((A, A))
    Vv[0, 0] = 1.5
    Vv[1, 1] = -0.5
    Vv[2, 2] = 0.3
    Vv[0, 1] = Vv[1, 0] = 0.2
    model = nm.LengthModel(L, "potts", 1.0, V=[Vv])
    lw = model.dense_log_weights()
    model.logZ = float(logsumexp(lw))
    model.entropy_bits = float((lw - model.logZ).mean() / LN2)  # not used

    # empirical data = all 676 states weighted by the model (a good proposal)
    states = nm.enumerate_states(L)
    p = np.exp(lw.ravel() - model.logZ)
    counts = np.maximum((p * 1e6).astype(np.int64), 1).astype(float)
    prop = nm.Proposal.from_data(states, counts, L)

    U = rng.normal(size=(2, nm.S, nm.S))
    s, w = nm.importance_samples(model, prop, 200_000, rng)
    x, y = nm.coords_of_states(U, s)
    edges = nm.grid_from_coords([x], [y], 4)
    b = nm.bin_index_xy(x, y, edges)
    m1, m2 = nm.bin_accumulate(b, w, 16)
    est, se = nm.mean_and_se(m1, m2, len(s))
    # exact
    xa, ya = nm.coords_of_states(U, states)
    ba = nm.bin_index_xy(xa, ya, edges)
    exact = np.bincount(ba, weights=p, minlength=16)
    # within 4 SE (generous: the proposal here is near-perfect, ESS ~ N)
    assert np.allclose(est, exact, atol=4 * se.max() + 1e-6)
    assert (se > 0).any()


def test_importance_weights_unbiased_mean_one():
    rng = np.random.default_rng(2)
    L = 3
    states = nm.enumerate_states(L)
    counts = rng.integers(10, 1000, len(states)).astype(float)
    model = nm.LengthModel(L, "potts", 1.0, V=[np.zeros((A, A))] * 3)
    model.logZ = float(L * np.log(A))
    model.entropy_bits = float(L * np.log2(A))
    prop = nm.Proposal.from_data(states, counts, L)
    s, w = nm.importance_samples(model, prop, 50_000, rng)
    assert np.isclose(w.mean(), 1.0, atol=0.05)      # E_q[P/q] = 1


# --------------------------------------------------------------------------- #
# full-move strict minima
# --------------------------------------------------------------------------- #

def test_uniform_single_letter_is_not_a_minimum():
    # a single letter under a uniform model is never strict: it has a deletion-free
    # substitution neighbour at equal energy (or an insertion at lower length mass).
    models = {1: nm._fit_uniform(1),
              2: nm.LengthModel(2, "potts", 1.0, V=[np.zeros((A, A))],
                                logZ=2 * np.log(A), entropy_bits=2 * np.log2(A))}
    pL = nm.length_prior({})
    mask = nm.full_move_strict_minima(models, pL, 1, np.array([[0]]))
    # every single-letter string is equivalent; none is strictly below all neighbours
    # (substitution to another letter is equal energy, so not strict).
    assert not mask[0]


def test_deep_well_is_a_strict_minimum():
    # a length-1 model with a single strongly favoured letter is a strict minimum
    # against substitution (only move type available within length 1), provided the
    # favoured letter's energy is below all 25 substitutions.
    pL = nm.length_prior({})
    # length-1 block is uniform (no couplings); give it a site field through the
    # grand-canonical scale is not possible, so instead test length 2 with a deep
    # diagonal coupling favouring "aa".
    Vv = np.zeros((A, A))
    Vv[0, 0] = 5.0
    model = nm.LengthModel(2, "potts", 1.0, V=[Vv], logZ=0.0, entropy_bits=0.0)
    lw = model.dense_log_weights()
    model.logZ = float(logsumexp(lw))
    models = {1: nm._fit_uniform(1),
              2: model,
              3: nm.LengthModel(3, "potts", 1.0, V=[np.zeros((A, A))] * 3,
                                logZ=3 * np.log(A), entropy_bits=3 * np.log2(A))}
    pL2 = np.array([0.0, 1.0, 0, 0, 0, 0, 0, 0])
    st = np.array([[0, 0]])                          # "aa"
    assert nm.full_move_strict_minima(models, pL2, 2, st)[0]


# --------------------------------------------------------------------------- #
# fitting (smoke)
# --------------------------------------------------------------------------- #

def test_fit_length_one_is_uniform():
    m, d = nm.fit_length(np.zeros((0, 1)), np.zeros(0), 1, None)
    assert m.backend == "uniform"
    assert d["uniform"] is True


def test_fit_length_two_potts_converges():
    rng = np.random.default_rng(3)
    words = ["th", "he", "in", "er", "an", "re", "on", "at", "en", "nd", "ti", "es"]
    st = np.array([[ord(c) - 97 for c in w] for w in words])
    counts = rng.integers(100, 10_000, len(words)).astype(float)
    m, d = nm.fit_length(st, counts, 2, 1.0)
    assert m.backend == "potts"
    assert np.isfinite(m.logZ)
    assert d["converged"]


def test_empty_frame_is_uniform():
    for L in (2, 5, 8):
        m, d = nm.fit_length(np.zeros((0, L), dtype=np.int64), np.zeros(0), L, 3.16)
        assert d["uniform"] is True
        assert np.isclose(m.logZ, L * np.log(26.0))
        assert np.isclose(m.entropy_bits, L * np.log2(26.0))


# --------------------------------------------------------------------------- #
# quantisation sentinel and count floor
# --------------------------------------------------------------------------- #

def test_quantize_surface_gap_sentinel_survives():
    z = np.array([1.234, np.nan, -16.9, np.inf, 0.0])
    q = nm.quantize_surface(z, 100.0)
    assert q.dtype == np.int16
    # the NaN gap maps to exactly -32768, not -32767
    assert q[1] == -32768
    # finite values are finite centibits
    assert q[0] == 123
    assert q[2] == -1690
    assert q[4] == 0
    # +inf is a gap too (non-finite)
    assert q[3] == -32768
    # and the -32768 sentinel is preserved on a round-trip
    dec = np.where(q == -32768, np.nan, q / 100.0)
    assert np.isnan(dec[1]) and np.isnan(dec[3])
    assert np.isclose(dec[0], 1.23)


# --------------------------------------------------------------------------- #
# per-cell top words (hover feature)
# --------------------------------------------------------------------------- #

def test_cell_top_words_argmax():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import build_nm_landscape as B

    rng = np.random.default_rng(0)
    L = 2
    Vv = np.zeros((A, A))
    Vv[0, 0] = 4.0                              # favour "aa"
    Vv[5, 5] = 2.0                              # and "ff"
    model = nm.LengthModel(2, "potts", 1.0, V=[Vv])
    lw = model.dense_log_weights()
    model.logZ = float(logsumexp(lw))
    import math
    models = {1: nm._fit_uniform(1), 2: model}
    for L in (3, 4, 5):
        npairs = L * (L - 1) // 2
        models[L] = nm.LengthModel(L, "potts", 1.0, V=[np.zeros((A, A))] * npairs,
                                   logZ=L * np.log(A), entropy_bits=L * np.log2(A))
    pL = np.array([0.0, 1.0, 0, 0, 0, 0, 0, 0])  # only length 2
    fr = Counter({"aa": 5, "ff": 3})
    U = rng.normal(size=(2, nm.S, nm.S))
    states = nm.enumerate_states(2)
    x, y = nm.coords_of_states(U, states)
    edges = nm.grid_from_coords([x], [y], 4)
    n_cells = 16
    cw_all = B._cell_top_words_all_levels(models, pL, fr, U, {4: edges}, 5, rng)
    cw = cw_all[4]
    # brute force: per cell, the top-1 word is the argmax of P over all L=2 strings
    P = np.exp(lw.ravel() - model.logZ)          # P(w | L=2), P(L=2)=1
    b = nm.bin_index_xy(x, y, edges)
    checked = 0
    for c in range(n_cells):
        mask = b == c
        if not mask.any():
            # no L=2 string projects here; the other lengths have P(L)=0, so any
            # returned word has zero grand-canonical probability
            if cw[c]:
                assert cw[c][0]["p"] == 0.0
            continue
        assert cw[c]
        # top-1 must attain the maximum P (ties broken arbitrarily), and the list is
        # sorted by P descending
        assert np.isclose(cw[c][0]["p"], float(P[mask].max()))
        ps = [e["p"] for e in cw[c]]
        assert ps == sorted(ps, reverse=True)
        checked += 1
    assert checked > 0


def test_level32_wells_match_static_tab():
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "data" / "derived" / "nm_landscape.json"
    if not p.exists():
        pytest.skip("nm_landscape.json not built")
    s = json.load(open(p))["summary"]
    assert s["levels"]["32"]["wells"] == s["models"]["final"]["wells"]
    assert s["levels"]["32"]["barrier"] == s["models"]["final"]["barrier"]
    assert s["levels"]["32"]["wells_adjacency"] == "move_set"
    assert s["levels"]["8"]["wells_adjacency"] == "grid8"


def test_well_labels_rank_by_probability_not_cell_index():
    # A planted basin: the minimum cell holds three counter-ordered words, the
    # highest-index cell holds the single most probable word.  The well's name must be
    # the basin's most probable string (in the last cell), not the minimum cell's word
    # and not cell/corpus order; the minimum cell's word is stored separately.
    n = 3
    E = np.array([0.0, 1.0, 2.0] + [np.inf] * (n * n - 3))
    adj = [np.array([1]), np.array([0, 2]), np.array([1])] + \
          [np.array([], dtype=np.int64) for _ in range(n * n - 3)]
    cell_top = {
        0: [{"w": "x1", "L": 2, "p": 0.3, "approx": False},
            {"w": "x2", "L": 2, "p": 0.2, "approx": False},
            {"w": "x3", "L": 2, "p": 0.1, "approx": False}],
        1: [{"w": "bb", "L": 2, "p": 0.5, "approx": False}],
        2: [{"w": "cc", "L": 2, "p": 0.9, "approx": False}],
    }
    centres = [np.array([0.5, 1.5, 2.5]), np.array([0.5, 1.5, 2.5])]
    wells, barrier, lab = nm.projected_wells(E, adj, 0.0, cell_top, centres)
    assert len(wells) == 1
    labels = [w["w"] for w in wells[0]["words"]]
    assert labels == ["cc", "bb", "x1"]   # basin ranked by P
    assert wells[0]["min_cell_top"]["w"] == "x1"
    assert wells[0]["min_cell_top"]["p"] == 0.3


def test_name_is_basin_max_p_and_ge_min_cell_top():
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "data" / "derived" / "nm_landscape.json"
    if not p.exists():
        pytest.skip("nm_landscape.json not built")
    s = json.load(open(p))["summary"]
    for G in (32, 96):
        lvl = s["levels"][str(G)]
        for well in lvl["wells"]:
            if not well["words"]:
                continue
            name_p = well["words"][0]["p"]
            for e in well["words"][1:]:                    # sorted by P
                assert name_p >= e["p"], f"G={G} cell {well['cell']} not sorted"
            mct = well.get("min_cell_top")
            if mct:
                assert name_p >= mct["p"], \
                    f"G={G} cell {well['cell']}: name P {name_p} < min_cell_top P {mct['p']}"
            for e in well["words"]:                        # share field present
                assert "share" in e


def test_persistence_saddle_equals_minimax_bottleneck():
    # Two basins joined by a chain whose peak is cell 5 (z = 5).  The persistence
    # merge saddle between the two minima must equal the minimax (bottleneck) height.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import build_nm_landscape as B
    from langfit import projection as pj

    z = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=float)
    adj = [np.array([1]), np.array([0, 2]), np.array([1, 3]), np.array([2, 4]),
           np.array([3, 5]), np.array([4, 6]), np.array([5, 7]), np.array([6, 8]),
           np.array([7, 9]), np.array([8])]
    wells = pj.persistence(z, adj)
    # the two minima are 0 (z=0) and 9 (z=1); they merge at the chain peak cell 5 (z=5)
    saddle = {w["cell"]: w for w in wells if not w["essential"] and w["saddle"] == 5}
    assert saddle, "expected a merge at cell 5"
    path, bottleneck = B._minimax_path(z, adj, 0, 9)
    assert path == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert bottleneck == 5.0


def test_real_model_barrier_saddle_equals_bottleneck():
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "data" / "derived" / "nm_landscape.json"
    if not p.exists():
        pytest.skip("nm_landscape.json not built")
    s = json.load(open(p))["summary"]
    for G in (32, 96):
        lvl = s["levels"][str(G)]
        bar = lvl.get("barrier")
        bp = lvl.get("barrier_path")
        if bar is None or bp is None or bp.get("bottleneck") is None:
            continue
        # the minimax bottleneck must equal the persistence saddle height (both on the
        # same adjacency and the same clipped z)
        assert abs(bp["bottleneck"] - bar["E"]) < 1e-6, \
            f"G={G}: bottleneck {bp['bottleneck']} != saddle {bar['E']}"
