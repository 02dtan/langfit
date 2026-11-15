"""Tests of the dense Potts model against the frozen sparse implementation."""

from itertools import combinations

import numpy as np
import pytest

from langfit import corpus, maxent as me, potts

# Equivalence fits run at a tighter stopping rule than the default 1% of a standard
# error (numerical tolerance), so that both solvers stop close to the optimum.
TIGHT = 1e-3


@pytest.fixture(scope="module")
def sample():
    c = corpus.fixed_length_words(corpus.load_corpus(list(corpus.AUSTEN)))
    st, ct = potts.counts_to_arrays(c, 4)
    wc = me.WordCounts.from_counter(c)
    assert np.array_equal(wc.states, st) and np.array_equal(wc.counts, ct)
    return potts.subsample(st, ct, 10_000, 7)


@pytest.fixture(scope="module")
def synthetic():
    """Tokens of a random strictly positive three-body model on a 6-letter sub-alphabet,
    with enough tokens that every triplet cell of the sub-alphabet is observed. The
    maximum-entropy optimum is then interior in the free cells and both solvers reach
    it to near machine precision, so they can be compared at 1e-6 bits."""
    rng = np.random.default_rng(11)
    k = 6
    lw = np.zeros((k,) * 4)
    for c in potts.cliques(3, 4):
        lw = lw + rng.normal(scale=0.7, size=(k,) * 3).reshape(
            tuple(k if a in c else 1 for a in range(4)))
    p = np.exp(lw - lw.max())
    p /= p.sum()
    n = rng.multinomial(200_000, p.ravel())
    idx = np.flatnonzero(n)
    st = np.stack(np.unravel_index(idx, (k,) * 4), axis=1).astype(np.int64)
    ct = n[idx].astype(float)
    for c, m in zip(potts.cliques(3, 4), potts.marginals(potts.empirical(st, ct, 4),
                                                          potts.cliques(3, 4))):
        assert (m[(slice(0, k),) * 3] > 0).all()
    return st, ct


def test_broadcast_and_marginals():
    rng = np.random.default_rng(0)
    for L, k in ((3, 2), (4, 3), (5, 2)):
        cl = potts.cliques(k, L)
        V = [rng.normal(size=(potts.A,) * k) for _ in cl]
        naive = np.zeros((potts.A,) * L)
        for c, v in zip(cl, V):
            naive = naive + v.reshape(tuple(potts.A if a in c else 1 for a in range(L)))
        assert np.allclose(potts._broadcast_sum(L, cl, V), naive)
        p = rng.random((potts.A,) * L)
        for c, m in zip(cl, potts.marginals(p, cl)):
            other = tuple(a for a in range(L) if a not in c)
            assert np.allclose(m, p.sum(axis=other))


def test_split_matches_maxent(sample):
    st, ct = sample
    a = potts.split(ct, 5, np.random.default_rng(3))
    b = me.WordCounts(st, ct, 4).split(5, np.random.default_rng(3))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def _se_S_bits(ct):
    """Sampling scale of the entropy used by the stopping rule (maxent.fit), in bits."""
    N = ct.sum()
    p = ct / N
    lnp = np.log(p)
    var = max(float((p * lnp ** 2).sum() - (p * lnp).sum() ** 2), 0.0)
    return max(np.sqrt(var / N), (p.size - 1) / (2 * N)) / np.log(2)


def _compare(st, ct, order, mode, tol):
    wc = me.WordCounts(st, ct, 4)
    s2 = 1.0
    lam = me.prior_lambda(s2, ct.sum()) if mode == "map" else 0.0
    # map mode stops on the posterior gradient only, so it is run tighter (numerical tolerance)
    res = TIGHT if mode == "exact" else TIGHT / 10
    m_ref, d_ref = me.fit(wc, order, "exact" if mode == "exact" else "l2", lam=lam,
                          resolution=res)
    m_new, d_new = potts.fit(st, ct, 4, order, mode, sigma2=s2 if mode == "map" else None,
                             resolution=res)
    assert d_ref["converged"] and d_new["converged"]
    if tol is None:
        # practical tolerance: sum of the two fits' estimated entropy errors (an
        # estimate, not a bound; see test_entropy_error_estimate_boundary)
        tol = (d_ref["entropy_error_in_se"] + d_new["entropy_error_in_se"]) * _se_S_bits(ct)
    assert abs(m_ref.entropy_bits() - m_new.entropy_bits()) <= max(tol, 1e-12)
    assert abs(m_ref.cross_entropy_bits(wc) - m_new.cross_entropy_bits(st, ct)) <= max(tol, 1e-6)
    p_ref, p_new = m_ref.prob(), m_new.prob()
    # each cell is within TIGHT * se_c of the data, se_c <= max(sqrt(1/(4N)), 1/N)
    tol_m = 2 * TIGHT * max(0.5 / np.sqrt(ct.sum()), 1.0 / ct.sum())
    for c in potts.cliques(order, 4):
        assert np.abs(me.marginal(p_ref, c) - me.marginal(p_new, c)).max() < tol_m
    for v_ref, v_new in zip(m_ref.V, m_new.V):       # forbidden cells coincide
        assert np.array_equal(np.isfinite(v_ref), np.isfinite(v_new))


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("mode", ["exact", "map"])
def test_equivalence_synthetic(synthetic, order, mode):
    """Interior optimum: the two implementations agree to 1e-6 bits."""
    _compare(*synthetic, order, mode, 1e-6)


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("mode", ["exact", "map"])
def test_equivalence_austen(sample, order, mode):
    """Real words (forbidden cells, couplings running to -inf): the two implementations
    agree within the sum of their estimated entropy errors (a practical tolerance)."""
    _compare(*sample, order, mode, 1e-6 if mode == "map" or order == 1 else None)


def test_cv_fold_matches_maxent(sample):
    st, ct = sample
    a = potts.cv_fold(st, ct, 4, 1, [0.1, 1.0], fold=2, n_folds=4, seed=5)
    b = me.cv_fold(me.WordCounts(st, ct, 4), 1, [0.1, 1.0], fold=2, n_folds=4, seed=5)
    for (s_a, x_a), (s_b, x_b) in zip(a, b):
        assert s_a == s_b and abs(x_a - x_b) < 1e-5


def test_full_clique_reproduces_empirical():
    """Order L in exact mode is the empirical distribution itself: S_L = plug-in."""
    rng = np.random.default_rng(1)
    st = rng.integers(0, 5, size=(300, 3))
    ct = rng.integers(1, 20, size=300).astype(float)
    m, d = potts.fit(st, ct, 3, 3, "exact")
    p = potts.empirical(st, ct, 3).ravel()
    p = p[p > 0]
    assert d["converged"]
    assert abs(m.entropy_bits() + (p * np.log2(p)).sum()) < 1e-6


def test_entropy_error_estimate_boundary():
    """Analytically solvable boundary case: tokens 100, 010, 001 (100 each). The pair
    marginals forbid every other string, so the order-2 maximum entropy is log2 3, with
    the optimum at infinite couplings. The fit's entropy-error figure is an estimate
    (local quadratic dual gap), not a bound: the true error is ~1.03x the estimate here.
    The test asserts the factor-2 margin quoted in the documentation."""
    st = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ct = np.full(3, 100.0)
    for res in (1e-2, 1e-3):
        m, d = potts.fit(st, ct, 3, 2, "exact", resolution=res)
        assert d["converged"]
        N = ct.sum()
        se_S_bits = ((ct.size - 1) / (2 * N)) / np.log(2)   # uniform counts: Miller floor
        estimate = d["entropy_error_in_se"] * se_S_bits
        assert abs(m.entropy_bits() - np.log2(3)) <= 2 * estimate

def test_l5_pairwise_matches_marginals():
    rng = np.random.default_rng(2)
    st = rng.integers(0, 26, size=(2000, 5))
    st[:, 1] = (st[:, 0] + rng.integers(0, 3, size=2000)) % 26     # a pair correlation
    ct = rng.integers(1, 10, size=2000).astype(float)
    p_emp = potts.empirical(st, ct, 5)
    cl = potts.cliques(2, 5)
    for mode in ("exact", "map"):
        m, d = potts.fit(st, ct, 5, 2, mode, sigma2=10.0 if mode == "map" else None)
        assert d["converged"]
        if mode == "exact":
            for a, b in zip(m.marginals(cl), potts.marginals(p_emp, cl)):
                assert np.abs(a - b).max() < 1e-4


def test_warm_start_from_lower_order(sample):
    st, ct = sample
    m2, _ = potts.fit(st, ct, 4, 2, "map", sigma2=1.0, resolution=TIGHT)
    cold, _ = potts.fit(st, ct, 4, 3, "map", sigma2=1.0, resolution=TIGHT)
    warm, d = potts.fit(st, ct, 4, 3, "map", sigma2=1.0, init=m2, resolution=TIGHT)
    assert d["converged"]
    assert abs(cold.entropy_bits() - warm.entropy_bits()) < 1e-5


def test_explicit_cliques():
    """Pairs plus one triplet: the triplet marginal is matched exactly."""
    rng = np.random.default_rng(4)
    st = rng.integers(0, 6, size=(500, 4))
    ct = rng.integers(1, 30, size=500).astype(float)
    cl = potts.cliques(2, 4) + [(0, 1, 2)]
    m, d = potts.fit(st, ct, 4, 0, "exact", cliques_=cl)
    assert d["converged"]
    emp = potts.marginals(potts.empirical(st, ct, 4), [(0, 1, 2)])[0]
    assert np.abs(m.marginals([(0, 1, 2)])[0] - emp).max() < 1e-4
    assert list(combinations(range(4), 2)) == m.cliques[:6]
