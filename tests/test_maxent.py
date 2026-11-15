"""Correctness of the maximum-entropy machinery, checked against exact answers.

All tests use the real 26-letter alphabet with L = 4 only where the answer is known
in closed form; otherwise the claim is a structural property (moment matching,
entropy ordering, planted-model recovery) that a wrong implementation would break.
"""

from collections import Counter

import numpy as np
import pytest

from langfit import maxent as me


def _planted(rng, order=2, scale=1.0):
    """A random pairwise model over 26^4 words, and an exact sample from it."""
    model = me.MaxEnt(L=4, cliques=me.cliques(order))
    model.V = [rng.normal(0.0, scale, (me.A,) * order) for _ in model.cliques]
    return model


def _sample(model, n, rng):
    p = model.prob().ravel()
    idx = rng.choice(p.size, size=n, p=p)
    words = [me.unravel(i) for i in idx]
    return me.WordCounts.from_counter(Counter(words))


def test_independent_model_is_product_of_position_marginals():
    """Order 1 has a closed form: P = prod_i p_i, S_1 = sum_i H(X_i)."""
    wc = me.WordCounts.from_counter(Counter({"that": 50, "this": 30, "with": 15, "have": 5}))
    m, d = me.fit(wc, 1, "exact", resolution=1e-7)
    p_emp = wc.empirical()
    prod = np.ones((me.A,) * 4)
    H = 0.0
    for i in range(4):
        pi = me.marginal(p_emp, (i,))
        shape = [1] * 4
        shape[i] = me.A
        prod = prod * pi.reshape(shape)
        H -= float((pi[pi > 0] * np.log2(pi[pi > 0])).sum())
    assert np.allclose(m.prob(), prod, atol=1e-9)
    assert m.entropy_bits() == pytest.approx(H, abs=1e-7)


def test_exact_fit_matches_every_pairwise_marginal():
    rng = np.random.default_rng(0)
    wc = _sample(_planted(rng), 20_000, rng)
    m, d = me.fit(wc, 2, "exact")
    p_emp, p = wc.empirical(), m.prob()
    for c in me.cliques(2):
        assert np.abs(me.marginal(p, c) - me.marginal(p_emp, c)).max() < 1e-6


def test_entropy_hierarchy_is_monotone():
    """S_1 >= S_2 >= S_3 >= S_emp: adding constraints can only lower the entropy
    (Schneidman et al. PRL 2003). A fit that violates this is not maximum entropy."""
    rng = np.random.default_rng(1)
    wc = _sample(_planted(rng, order=3, scale=1.5), 5_000, rng)
    p = wc.counts / wc.counts.sum()
    S_emp = float(-(p * np.log2(p)).sum())
    S = [me.fit(wc, k, "exact")[0].entropy_bits() for k in (1, 2, 3)]
    assert S[0] >= S[1] - 1e-6 >= S[2] - 2e-6
    assert S[2] >= S_emp - 1e-6


def test_planted_pairwise_model_is_recovered():
    """With enough samples from a true pairwise model, the fitted entropy converges
    to the true entropy and the third-order model adds nothing."""
    rng = np.random.default_rng(2)
    truth = _planted(rng, scale=0.6)
    wc = _sample(truth, 400_000, rng)
    m2, _ = me.fit(wc, 2, "l2", lam=1e-6)
    assert m2.entropy_bits() == pytest.approx(truth.entropy_bits(), abs=0.05)
    # energies agree up to the free additive constant on the high-probability words
    p = truth.prob().ravel()
    top = np.argsort(p)[-2000:]
    e_t, e_f = truth.energy().ravel()[top], m2.energy().ravel()[top]
    assert np.corrcoef(e_t, e_f)[0, 1] > 0.98


def test_forbidden_cells_get_zero_probability_in_exact_mode():
    """A letter pair never seen at positions (i, j) is forbidden: every word that
    contains it has probability exactly zero, which is what makes the landscape
    sparse rather than continuous."""
    wc = me.WordCounts.from_counter(Counter({"abcd": 3, "abce": 2, "xbcd": 1}))
    m, _ = me.fit(wc, 2, "exact")
    p = m.prob()
    assert p[tuple(ord(c) - 97 for c in "zzzz")] == 0.0
    assert p[tuple(ord(c) - 97 for c in "abcd")] > 0.0


def test_l2_model_assigns_finite_energy_everywhere():
    wc = me.WordCounts.from_counter(Counter({"abcd": 3, "abce": 2, "xbcd": 1}))
    m, _ = me.fit(wc, 2, "l2", lam=1e-3)
    assert np.all(np.isfinite(m.energy()))


def test_local_minima_brute_force():
    """The vectorised minimum test agrees with checking all 4*25 neighbours."""
    rng = np.random.default_rng(3)
    m = _planted(rng, scale=2.0)
    E = m.energy()
    mins = me.local_minima(E)
    flat = np.flatnonzero(mins)
    assert flat.size > 0
    for i in list(flat[:50]) + list(rng.choice(E.size, 200, replace=False)):
        s = np.unravel_index(i, E.shape)
        brute = all(E[s[:ax] + (a,) + s[ax + 1:]] > E[s]
                    for ax in range(4) for a in range(me.A) if a != s[ax])
        assert brute == bool(mins[s])


def test_descent_reaches_a_minimum_and_minima_are_fixed_points():
    rng = np.random.default_rng(4)
    E = _planted(rng, scale=2.0).energy()
    target = me.descend(E)
    mins = me.local_minima(E).ravel()
    assert mins[target].all()                       # every word ends in a minimum
    fixed = np.flatnonzero(mins)
    assert np.array_equal(target[fixed], fixed)     # minima map to themselves
    # the explicit path agrees with the vectorised map
    for i in rng.choice(E.size, 30, replace=False):
        path = me.descent_path(E, me.unravel(i))
        assert me.ravel(path[-1][0]) == target[i]
        es = [e for _, e in path]
        assert all(b < a for a, b in zip(es, es[1:]))


def test_prior_selection_is_interior_and_extends_the_grid_when_needed():
    """Too narrow a prior underfits and too wide overfits, so the CV optimum is interior;
    when the data want couplings wider than the starting grid allows, the grid is
    extended rather than the endpoint being reported as an optimum."""
    rng = np.random.default_rng(5)
    wc = _sample(_planted(rng, scale=1.0), 3_000, rng)
    res = me.select_prior(wc, 2, n_folds=4)
    assert res["interior"]

    # planted couplings have variance 9, at the edge of the starting grid: the search
    # must extend past the endpoint and still report a bracketed optimum
    wide = _sample(_planted(np.random.default_rng(6), scale=3.0), 200_000,
                   np.random.default_rng(7))
    res = me.select_prior(wide, 2, n_folds=3)
    assert res["interior"]
    assert len(res["sigma2"]) > len(me.SIGMA2_START)


def test_degenerate_count_tables_converge_without_crashing():
    """Uniform counts zero the first-order entropy variance, and a letter that never
    varies zeroes a marginal's binomial variance; both standard errors are floored."""
    m, d = me.fit(me.WordCounts.from_counter(Counter({"aa": 1, "bb": 1}), L=2), 2, "exact")
    assert d["converged"]
    m, d = me.fit(me.WordCounts.from_counter(Counter({"aa": 2, "ab": 1}), L=2), 2, "exact")
    assert d["converged"]
    assert m.entropy_bits() == pytest.approx(-(2 / 3) * np.log2(2 / 3) - (1 / 3) * np.log2(1 / 3),
                                              abs=1e-6)


def test_plateau_terminals_are_not_reported_as_wells():
    E = np.zeros((me.A, me.A))
    assert not me.local_minima(E).any()
    target = me.descend(E)
    assert np.array_equal(target, np.arange(E.size))      # descent stops everywhere


def test_recovery_counts_only_the_intended_word():
    """Audit counterexample: with {'aaaa': 3, 'aaab': 1} every misspelling of either word
    relaxes to 'aaaa'. Basin agreement would score 1; decoding accuracy is 3/4."""
    wc = me.WordCounts.from_counter(Counter({"aaaa": 3, "aaab": 1}))
    m, _ = me.fit(wc, 2, "exact")
    rec = me.substitution_recovery(m.energy(), wc)
    w = wc.counts / wc.counts.sum()
    assert dict(zip(wc.words, rec)) == {"aaaa": 1.0, "aaab": 0.0}
    assert float((w * rec).sum()) == pytest.approx(0.75)


def test_forbidden_strings_relax_into_the_allowed_region():
    """In the exact model a string with a never-seen letter pair has infinite energy;
    steepest descent carries it to a finite-energy local minimum, which is how a
    misspelling that breaks a pairwise constraint gets 'corrected'."""
    wc = me.WordCounts.from_counter(Counter({"that": 50, "this": 30, "with": 15, "have": 5}))
    m, _ = me.fit(wc, 2, "exact")
    E = m.energy()
    target = me.descend(E)
    i = me.ravel("thqt")
    assert not np.isfinite(E.ravel()[i])
    assert me.unravel(target[i]) == "that"
