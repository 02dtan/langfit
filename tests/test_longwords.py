"""Tests of langfit.longwords: pseudo-likelihood fits, exact split enumeration, AIS,
and the per-word landscape functions, each checked against an independent exact
computation on a grid small enough to enumerate."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

from langfit import longwords as lw
from langfit import maxent as me

A = lw.A
ROOT = Path(__file__).resolve().parent.parent


def random_model(L, scale, seed):
    rng = np.random.default_rng(seed)
    return lw.PairModel(L, rng.normal(0.0, scale, (L * (L - 1) // 2, A, A)))


def exact_sample(model, N, seed):
    """Multinomial counts of N exact draws from a dense (L <= 5) model."""
    lwd = model.dense_log_weights().ravel()
    p = np.exp(lwd - logsumexp(lwd))
    c = np.random.default_rng(seed).multinomial(N, p).astype(float)
    keep = np.flatnonzero(c)
    states = np.stack(np.unravel_index(keep, (A,) * model.L), axis=1).astype(np.int64)
    return states, c[keep]


def test_plm_gradient_matches_finite_differences():
    m = random_model(3, 0.5, 1)
    s, c = exact_sample(m, 5000, 2)
    data = lw._PLData(s, c, 3)
    free = np.ones(m.J.shape, dtype=bool)
    theta = np.random.default_rng(3).normal(0, 0.3, free.sum())
    lam = 1e-3
    f, g = lw._plm_objective(theta, free, 3, data, lam)
    rng = np.random.default_rng(4)
    for k in rng.choice(theta.size, 12, replace=False):
        e = np.zeros_like(theta)
        e[k] = 1e-6
        fd = (lw._plm_objective(theta + e, free, 3, data, lam)[0]
              - lw._plm_objective(theta - e, free, 3, data, lam)[0]) / 2e-6
        assert abs(fd - g[k]) < 1e-6 * max(1.0, abs(g[k]))


def test_plm_is_consistent_on_exact_samples():
    """PLM is consistent (Hyvarinen 2006): with many exact samples from a known model,
    the gauge-invariant couplings are recovered."""
    truth = random_model(3, 0.4, 5)
    s, c = exact_sample(truth, 2_000_000, 6)
    m, d = lw.fit_plm(s, c, 3, sigma2=None)
    assert d["converged"]
    agree = lw.coupling_agreement(m.to_V(), truth.to_V())
    assert agree["pearson"] > 0.99
    assert abs(agree["slope"] - 1) < 0.02


def brute_log_partition_L6(model):
    """Independent L = 6 enumeration: fix the first letter, sum 26^5 densely."""
    rest = lw.PairModel(5, np.stack([model.J[p] for p, (i, j) in enumerate(model.pairs) if i > 0]))
    base = rest.dense_log_weights()
    lz, num = [], []
    for a in range(A):
        lwd = base.copy()
        for p, (i, j) in enumerate(model.pairs):
            if i == 0:
                shape = [1] * 5
                shape[j - 1] = A
                lwd = lwd + model.J[p][a].reshape(shape)
        flat = lwd.ravel()
        lz.append(logsumexp(flat))
        if not np.isfinite(lz[-1]):             # every string with this first letter forbidden
            num.append(0.0)
            continue
        w = np.exp(flat - lz[-1])
        num.append((w * np.where(np.isfinite(flat), flat, 0.0)).sum())
    lz = np.array(lz)
    tot = logsumexp(lz)
    mlw = float((np.exp(lz - tot) * np.array(num)).sum())
    return float(tot), mlw


def test_split_enumeration_is_exact_at_L6():
    m = random_model(6, 0.6, 7)
    m.J[0, 3, :] = -np.inf                      # forbidden cells must be handled too
    got = lw.exact_log_partition(m)
    lz, mlw = brute_log_partition_L6(m)
    assert abs(got["logZ"] - lz) < 1e-9 * abs(lz)
    assert abs(got["mean_lw"] - mlw) < 1e-8 * abs(mlw)


def test_dense_partition_matches_maxent_entropy():
    m = random_model(4, 0.8, 8)
    ref = me.MaxEnt(L=4, cliques=me.cliques(2, 4), V=m.to_V())
    assert abs(lw.exact_log_partition(m)["entropy_bits"] - ref.entropy_bits()) < 1e-9


def test_ais_brackets_exact_log_partition():
    m = random_model(4, 0.6, 9)
    s, c = exact_sample(m, 100_000, 10)
    base = lw.fit_independent(s, c, 4, sigma2=10.0)
    exact = lw.exact_log_partition(m)
    a = lw.ais_log_partition(m, base, 2000, 300, seed=11)
    # five delta-method standard errors: a loose band, because the delta method is
    # itself an estimate
    assert abs(a["logZ"] - exact["logZ"]) < 5 * a["se"] + 1e-3
    assert abs(a["entropy_bits"] - exact["entropy_bits"]) < 5 * a["entropy_se_bits"] + 1e-3


def test_landscape_matches_dense_definitions():
    m = random_model(4, 1.0, 12)
    E = -m.dense_log_weights()
    mins = me.local_minima(E).ravel()
    target = me.descend(E)
    grid = np.stack(np.unravel_index(np.arange(A ** 4), (A,) * 4), axis=1)
    pick = np.random.default_rng(13).choice(A ** 4, 20000, replace=False)
    pick = np.union1d(pick, np.flatnonzero(mins))
    s = grid[pick]
    assert np.array_equal(lw.is_strict_minimum(m, s), mins[pick])
    end = lw.descend(m, s)
    assert np.array_equal(np.ravel_multi_index(end.T, (A,) * 4), target[pick])


def test_descent_from_forbidden_states_matches_dense():
    """Two disjoint forbidden pairs: every neighbour of 'aaaa' is still forbidden, so
    the dense reference stays put; and on a random masked model every start agrees."""
    J = np.zeros((6, A, A))
    J[0, 0, 0] = -np.inf            # pair (0, 1)
    J[5, 0, 0] = -np.inf            # pair (2, 3)
    m = lw.PairModel(4, J)
    start = np.array([[0, 0, 0, 0]])
    dense = me.descend(-m.dense_log_weights())
    assert dense[0] == 0
    assert np.array_equal(lw.descend(m, start), start)

    m = random_model(4, 1.0, 21)
    m.J[np.random.default_rng(22).random(m.J.shape) < 0.3] = -np.inf
    target = me.descend(-m.dense_log_weights())
    grid = np.stack(np.unravel_index(np.arange(A ** 4), (A,) * 4), axis=1)
    pick = np.random.default_rng(23).choice(A ** 4, 20000, replace=False)
    end = lw.descend(m, grid[pick])
    assert np.array_equal(np.ravel_multi_index(end.T, (A,) * 4), target[pick])


def test_independent_closed_form():
    m = random_model(4, 0.5, 14)
    s, c = exact_sample(m, 50_000, 15)
    ind = lw.fit_independent(s, c, 4)
    p = lw.site_counts(s, c, 4) / c.sum()
    h = -np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0).sum()
    assert abs(ind.entropy_bits() - h) < 1e-12


@pytest.mark.skipif(not (ROOT / "data/derived/twitter_L4_by_year.json").exists(),
                    reason="Twitter L = 4 counts not present")
def test_plm_agrees_with_exact_fit_at_L4():
    """PLM vs exact maximum likelihood (both with the same Gaussian prior) on real
    four-letter words at the common N of the frozen study. Tolerances are those the
    length study documents as the PLM approximation error at this N (Pearson of the
    gauge-invariant couplings, entropy and training code length in bits); they are
    checks that the approximation stays as good as reported, not sampling bands."""
    d = json.loads((ROOT / "data/derived/twitter_L4_by_year.json").read_text())
    C = Counter()
    for r in d["years"].values():
        C.update(r["counts"])
    wc = me.WordCounts.from_counter(C)
    rng = np.random.default_rng(20071231)
    c = rng.multivariate_hypergeometric(wc.counts.astype(np.int64), 142_614).astype(float)
    keep = c > 0
    s, c = wc.states[keep], c[keep]
    sigma2 = 1.0
    m, dm = lw.fit_plm(s, c, 4, sigma2)
    ex, dx = me.fit(me.WordCounts(s, c, 4), 2, "l2", lam=me.prior_lambda(sigma2, c.sum()))
    assert dm["converged"] and dx["converged"]
    agree = lw.coupling_agreement(m.to_V(), ex.V)
    assert agree["pearson"] > 0.97
    Z = lw.exact_log_partition(m)
    assert abs(Z["entropy_bits"] - ex.entropy_bits()) < 0.1
    ce_plm = -(c * (m.log_weight(s) - Z["logZ"])).sum() / c.sum() / np.log(2)
    ce_ex = ex.cross_entropy_bits(me.WordCounts(s, c, 4))
    assert 0 <= ce_plm - ce_ex < 0.1          # ML minimises training code length
