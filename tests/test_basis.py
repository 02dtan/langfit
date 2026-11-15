"""Correctness of ``langfit.basis`` (projection bases and exact heat-bath kinetics).

Every test uses a small alphabet A and length L (the module's routines are general in
(A,)*L tensors), so they are fast and exact.  Claims are structural (stationarity,
reversibility, P-orthonormality, spectra in the right interval, exact C(0) recovery),
which a wrong implementation would break.
"""

import numpy as np
import pytest

from langfit import basis as bs


def _rand_P(A, L, rng, scale=1.0):
    w = np.exp(rng.normal(0.0, scale, (A,) * L))
    return w / w.sum()


def _indicators(A, L):
    """Direct, brute-force one-hot indicators of every string: (L*A, A^L)."""
    X = np.zeros((L * A, A ** L))
    for s in range(A ** L):
        digits = np.unravel_index(s, (A,) * L)
        for i, a in enumerate(digits):
            X[i * A + a, s] = 1.0
    return X


def _cov(X, w):
    mu = X @ w
    return (X - mu[:, None]) @ ((X - mu[:, None]) * w).T


# --------------------------------------------------------------------------- #
# heat-bath kinetics
# --------------------------------------------------------------------------- #

def test_heat_bath_preserves_stationary_measure():
    rng = np.random.default_rng(0)
    P = _rand_P(3, 3, rng)
    F = rng.normal(size=(2, 3, 3, 3))
    TF = bs.heat_bath_apply(P, F)
    # E_P[T f] = E_P[f]  (stationarity)
    assert np.allclose((P * TF).sum(axis=(1, 2, 3)), (P * F).sum(axis=(1, 2, 3)))


def test_heat_bath_contracts_in_l2():
    rng = np.random.default_rng(1)
    P = _rand_P(3, 3, rng)
    f = rng.normal(size=(3, 3, 3))
    g = bs.heat_bath_apply(P, f[None])[0]
    assert (P * g ** 2).sum() <= (P * f ** 2).sum() + 1e-12


def test_indicator_lagged_cov_zero_is_exact_static_covariance():
    rng = np.random.default_rng(2)
    A, L = 3, 3
    P = _rand_P(A, L, rng)
    r = bs.indicator_lagged_covariances(P, lags=[1, 3])
    w = P.ravel()
    X = _indicators(A, L)
    C0 = _cov(X, w)
    assert np.allclose(r["C"][0], C0, atol=1e-12)
    # reversibility => lagged covariance is symmetric
    assert r["max_asymmetry"] < 1e-10
    for t in (1, 3):
        assert np.allclose(r["C"][t], r["C"][t].T, atol=1e-10)


def test_lagged_covariance_decays_to_zero_correlation():
    # for a well-mixed chain the lag-tau covariance of the indicators goes to 0
    rng = np.random.default_rng(3)
    A, L = 2, 4
    P = _rand_P(A, L, rng, scale=0.3)  # weak couplings => fast mixing
    r = bs.indicator_lagged_covariances(P, lags=[1, 2, 4, 8])
    norm = np.linalg.norm(r["C"][0])
    assert np.linalg.norm(r["C"][8]) < np.linalg.norm(r["C"][1])
    assert np.linalg.norm(r["C"][8]) < 0.5 * norm


# --------------------------------------------------------------------------- #
# linear bases
# --------------------------------------------------------------------------- #

def test_whitener():
    rng = np.random.default_rng(4)
    P = _rand_P(3, 3, rng)
    X = _indicators(3, 3)
    C0 = _cov(X, P.ravel())
    W = bs.whitener(C0)
    assert np.allclose(W.T @ C0 @ W, np.eye(W.shape[1]), atol=1e-9)


def test_pca_axes_sorted_unit_eigenvectors():
    rng = np.random.default_rng(5)
    P = _rand_P(3, 3, rng)
    C0 = _cov(_indicators(3, 3), P.ravel())
    lam, U = bs.pca_axes(C0)
    assert np.all(np.diff(lam) <= 1e-12)            # descending
    assert np.allclose(np.linalg.norm(U, axis=0), 1.0)   # unit columns


def test_tica_eigenvalues_are_autocorrelations():
    rng = np.random.default_rng(6)
    A, L = 3, 3
    P = _rand_P(A, L, rng)
    r = bs.indicator_lagged_covariances(P, lags=[1])
    C0, C1 = r["C"][0], r["C"][1]
    lam, V = bs.tica_axes(C0, C1)
    assert np.all(lam <= 1.0 + 1e-9)
    assert np.all(lam >= -1.0 - 1e-9)
    # C0-orthonormality of the returned axes
    assert np.allclose(V.T @ C0 @ V, np.eye(V.shape[1]), atol=1e-9)
    # lambda is the lag-1 autocorrelation of the coordinate v^T x
    for k in range(V.shape[1]):
        v = V[:, k]
        assert lam[k] == pytest.approx(v @ C1 @ v, abs=1e-9)


def test_eckart_young_no_linear_basis_beats_pc1_plus_pc2():
    # any 2-D subspace has <= lambda1 + lambda2 of the covariance trace
    rng = np.random.default_rng(7)
    P = _rand_P(3, 3, rng)
    X = _indicators(3, 3)
    C0 = _cov(X, P.ravel())
    lam1, lam2 = np.linalg.eigvalsh(C0)[-2:]
    V = rng.normal(size=(C0.shape[0], 2))
    Q, _ = np.linalg.qr(V)
    assert np.trace(Q.T @ C0 @ Q) <= lam1 + lam2 + 1e-9


# --------------------------------------------------------------------------- #
# kinetic scores
# --------------------------------------------------------------------------- #

def test_vamp2_identity_features_saturates():
    assert bs.vamp2(np.eye(2), np.eye(2)) == pytest.approx(2.0)


def test_vamp2_constant_feature_contributes_nothing():
    # an exactly constant feature (variance 0) is a null direction of C0y and is
    # dropped by the whitening, so only the non-constant feature's r^2 counts
    C0y = np.array([[0.0, 0.0], [0.0, 1.0]])
    Cty = np.array([[0.0, 0.0], [0.0, 0.5]])
    assert bs.vamp2(C0y, Cty) == pytest.approx(0.25, abs=1e-6)


def test_implied_timescales():
    assert bs.implied_timescales([0.5], 1)[0] == pytest.approx(-1.0 / np.log(0.5))
    assert np.isinf(bs.implied_timescales([1.0], 1)[0])
    assert np.isnan(bs.implied_timescales([0.0], 1)[0])
    assert np.isnan(bs.implied_timescales([-0.5], 1)[0])


# --------------------------------------------------------------------------- #
# support and slow modes
# --------------------------------------------------------------------------- #

def test_support_components_finds_disconnected_support():
    A, L = 2, 2
    P = np.zeros((A,) * L)
    P[0, 0] = 0.5   # "00"
    P[1, 1] = 0.5   # "11"  (no single-substitution path between them in the support)
    comp = bs.support_components(P)
    assert comp.max() + 1 == 2


def test_slow_modes_constant_first_and_orthonormal():
    rng = np.random.default_rng(8)
    P = _rand_P(2, 3, rng, scale=0.5)
    lam, psi = bs.slow_modes(P, k=4)
    assert lam[0] == pytest.approx(1.0, abs=1e-8)
    assert np.all(lam <= 1.0 + 1e-8)
    # psi[0] is the constant function on the support
    sup = (P > 0).ravel()
    assert np.allclose(psi[0, sup], 1.0, atol=1e-7)
    # P-orthonormality of the eigenfunctions
    w = P.ravel()
    for i in range(4):
        for j in range(4):
            ip = (psi[i] * psi[j] * w).sum()
            assert ip == pytest.approx(1.0 if i == j else 0.0, abs=1e-7)


# --------------------------------------------------------------------------- #
# class means / LDA / coordinates
# --------------------------------------------------------------------------- #

def test_class_means_and_lda_eigenvalue_range():
    rng = np.random.default_rng(9)
    A, L = 3, 3
    P = _rand_P(A, L, rng)
    X = _indicators(A, L)
    C0 = _cov(X, P.ravel())
    mean = X @ P.ravel()
    labels = np.zeros(A ** L, dtype=int)
    labels[A ** L // 2:] = 1          # arbitrary 2-class split
    uniq, pi, mu = bs.class_means(P, labels)
    assert np.allclose(pi.sum(), 1.0)
    lam, V = bs.lda_axes(P, labels, C0, mean)
    assert np.all(lam >= -1e-9)
    assert np.all(lam <= 1.0 + 1e-9)


def test_coords_linear_matches_manual_projection():
    rng = np.random.default_rng(10)
    A, L = 2, 2
    P = _rand_P(A, L, rng)
    X = _indicators(A, L)
    mean = X @ P.ravel()
    V = rng.normal(size=(L * A, 2))
    coords = bs.coords_linear(V, mean, L, A)
    assert coords.shape == (2, A ** L)
    assert np.allclose(coords, V.T @ (X - mean[:, None]))


def test_qgram_profile_locality_and_shape():
    p = bs.qgram_profile("cat")                     # '#cat#'
    assert p.shape == (27 ** 2,)                   # boundary-marked bigrams
    assert p.sum() == 4                            # '#c','ca','at','t#'
    # substitution changes at most 4 profile mass (2q) for q = 2
    q_abcd = bs.qgram_profile("abcd")
    q_axcd = bs.qgram_profile("axcd")
    assert np.abs(q_abcd - q_axcd).sum() <= 4
    # q = 1 counts each symbol of '#cat#' (the boundary appears twice): L + 2
    assert bs.qgram_profile("cat", q=1).sum() == 5
    # insertion adds one q-gram (length + 1 q-grams for q = 2)
    assert bs.qgram_profile("abcd").sum() == 5
    assert bs.qgram_profile("abxcd").sum() == 6


def test_pca_basis_matches_manual_pca():
    rng = np.random.default_rng(11)
    A, L = 3, 3
    P = _rand_P(A, L, rng)
    b = bs.pca_basis(P, alphabet="abc")
    C0 = _cov(_indicators(A, L), P.ravel())
    lam, U = bs.pca_axes(C0)
    assert np.allclose(b["eigenvalues"], lam[:2])
    Q, _ = np.linalg.qr(b["V"])
    Qr, _ = np.linalg.qr(U[:, :2])
    s = np.linalg.svd(Q.T @ Qr, compute_uv=False)
    assert np.allclose(s, 1.0, atol=1e-8)
    a = b["annotation"][0]
    assert np.isclose(sum(a["position_share"]), 1.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# selectable bases (registry + compute_basis)
# --------------------------------------------------------------------------- #

class _FakeModel:
    """A tiny pairwise model with the same interface ``compute_basis`` needs
    (``prob``/``cliques``/``V``), over an arbitrary small alphabet."""

    def __init__(self, P, cliques, V):
        self._P, self.cliques, self.V = P, cliques, V

    def prob(self):
        return self._P


def _small_model(rng, L=4, A=2):
    cliques = [(i, j) for i in range(L) for j in range(i + 1, L)]
    V = [rng.normal(size=(A, A)) for _ in cliques]
    lw = np.zeros((A,) * L)
    for (i, j), v in zip(cliques, V):
        shape = [1] * L
        shape[i] = shape[j] = A
        lw = lw + v.reshape(shape)
    P = np.exp(lw - lw.max())
    P /= P.sum()
    return _FakeModel(P, cliques, V)


def test_compute_basis_returns_valid_axes_for_all_names():
    rng = np.random.default_rng(12)
    m = _small_model(rng, L=4, A=2)
    for name in bs.BASES:
        b = bs.compute_basis(name, model=m, L=4, A=2)
        if b["kind"] == "coords":
            assert b["coords"].shape == (2, 2 ** 4)
            continue
        U = b["U"]
        assert U.shape == (8, 2)                       # L*A = 4*2
        assert np.allclose(np.linalg.norm(U, axis=0), 1.0)         # unit columns
        piv = np.abs(U).argmax(axis=0)
        assert np.all(U[piv, np.arange(2)] > 0)                     # sign convention
        b2 = bs.compute_basis(name, model=m, L=4, A=2)
        assert np.array_equal(U, b2["U"])                           # determinism
        assert b["kind"] == name


def test_pca_compute_basis_matches_projection_byte_for_byte():
    import langfit.maxent as me
    import langfit.projection as pj
    npz = np.load("data/derived/landscape_artifact.npz")
    m = me.MaxEnt(L=4, cliques=me.cliques(2), V=[v.copy() for v in npz["V_austen"]])
    npz.close()
    P = m.prob()
    b = bs.compute_basis("pca", model=m)
    lam, U = pj.principal_axes(pj.indicator_covariance(P))
    assert np.array_equal(b["U"], U[:, :2])
    assert np.allclose(b["mean"], pj.site_marginals(P).ravel())


def test_energy_aligned_axis_tracks_planted_energy():
    # energy depends on one letter at one position -> PLS axis 1 must load it
    L, A = 4, 3
    E = np.zeros((A,) * L)
    E[0] = 5.0                                       # (position 0, letter 0) -> E=5
    Pw = np.exp(-E) / np.exp(-E).sum()
    b = bs.compute_basis("energy_aligned", P=Pw, E=E, L=L, A=A, alphabet="abc")
    u1 = b["U"][:, 0]
    assert int(np.argmax(np.abs(u1))) == 0            # (0, letter 0) is the top loading
    X = _indicators(A, L)
    y = u1 @ (X - (X @ Pw.ravel())[:, None])
    assert abs(np.corrcoef(y, E.ravel())[0, 1]) > 0.95


def test_coupling_spectral_finds_planted_pattern():
    # ferromagnetic "00" couplings -> the top J eigenvector is the all-(letter 0 vs
    # letter 1) pattern (up to global sign)
    L, A = 4, 2
    cliques = [(i, j) for i in range(L) for j in range(i + 1, L)]
    V = []
    for _ in cliques:
        Vv = np.zeros((A, A))
        Vv[0, 0] = 1.0
        V.append(Vv)
    m = _FakeModel(np.ones((A,) * L) / A ** L, cliques, V)
    b = bs.compute_basis("coupling_spectral", model=m, L=L, A=2)
    u1 = b["U"][:, 0]
    idx0 = np.array([p * A for p in range(L)])        # (position p, letter 0)
    idx1 = idx0 + 1                                    # (position p, letter 1)
    assert np.allclose(u1[idx0], u1[idx0][0])          # all letter-0 loadings equal
    assert np.allclose(u1[idx1], -u1[idx0][0])         # letter-1 = opposite
    assert abs(u1[idx0][0]) > np.abs(u1).max() * 0.5   # dominates the axis


def test_all_bases_are_distinct():
    """Every pair of the five shipped bases spans measurably different 2-D subspaces
    (principal angle > 1 deg) on a real model."""
    import langfit.maxent as me
    npz = np.load("data/derived/landscape_artifact.npz")
    m = me.MaxEnt(L=4, cliques=me.cliques(2), V=[v.copy() for v in npz["V_austen"]])
    npz.close()
    bases = {n: bs.compute_basis(n, model=m)["U"] for n in bs.BASES
             if bs.BASES[n].get("kind") != "coords"}
    names = list(bases)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            Qa, _ = np.linalg.qr(bases[names[i]])
            Qb, _ = np.linalg.qr(bases[names[j]])
            s = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1)
            ang = np.rad2deg(np.arccos(s.min()))
            assert ang > 1.0, f"{names[i]} vs {names[j]}: angle {ang:.2f} deg"


def test_lda_fisher_eigenvalues_are_correlation_ratios():
    """The LDA generalised eigenvalue lambda = between-class / total variance (the
    correlation ratio eta^2), hence lies in [0, 1] and is >= a random direction's."""
    import langfit.maxent as me
    rng = np.random.default_rng(15)
    L, A = 3, 3
    P = _rand_P(A, L, rng)
    with np.errstate(divide="ignore"):
        E = -np.log(P)
    labels = me.descend(E).astype(np.int64).ravel()
    C0 = _cov(_indicators(A, L), P.ravel())
    mean = _indicators(A, L) @ P.ravel()
    lam, V = bs.lda_axes(P, labels, C0, mean)
    assert np.all(lam >= -1e-9) and np.all(lam <= 1.0 + 1e-9)
    # a random direction's correlation ratio is <= the top LDA eigenvalue
    rnd = rng.normal(size=L * A)
    rnd = rnd / np.linalg.norm(rnd)
    _, pi, mu = bs.class_means(P, labels)
    pi = pi / pi.sum()
    SB = ((mu - mean) * pi[:, None]).T @ (mu - mean)
    eta2_rand = float(rnd @ SB @ rnd) / float(rnd @ C0 @ rnd)
    assert eta2_rand <= lam[0] + 1e-9


def test_energy_aligned_maximises_energy_covariance():
    """PLS axis 1 achieves the maximum P-weighted |covariance with E| among all unit
    directions (its stated optimality)."""
    rng = np.random.default_rng(16)
    L, A = 3, 3
    P = _rand_P(A, L, rng)
    with np.errstate(divide="ignore"):
        E = -np.log(P)
    E = np.where(np.isfinite(E), E, 0.0)
    b = bs.compute_basis("energy_aligned", P=P, E=E, L=L, A=A, alphabet="abc")
    u1 = b["U"][:, 0]
    X = _indicators(A, L)
    mean_E = float((P.ravel() * E.ravel()).sum())
    def cov(v):
        y = v @ X
        return abs(float((P.ravel() * (y - y @ P.ravel()) * (E.ravel() - mean_E)).sum()))
    # PLS axis 1 should beat any single indicator and be near the optimum
    best_indicator = max(cov(np.eye(L * A)[k]) for k in range(L * A))
    assert cov(u1) >= best_indicator - 1e-9
    # and beat a sample of unit-norm random directions
    rng2 = np.random.default_rng(0)
    sample = 0.0
    for _ in range(200):
        r = rng2.normal(size=L * A)
        sample = max(sample, cov(r / np.linalg.norm(r)))
    assert cov(u1) >= sample - 1e-9


def test_mass_weighted_scale_ignores_sparse_outlier():
    """A single sparse (tiny-mass) outlier cell must not change the robust roughness
    scales (mass-weighted SD and 5-95 IQR), while it stretches the raw range enormously."""
    E = np.array([10.0, 10.5, 11.0, 10.2, 10.8, 11.2, 10.4, 10.9, 11.1])
    mass = np.ones(9)
    occ = np.ones(9, dtype=bool)
    base = bs.mass_weighted_scale(E, mass, occ)
    E2 = np.append(E, [200.0])
    mass2 = np.append(mass, [1e-6])                   # a sparse outlier cell
    occ2 = np.ones(10, dtype=bool)
    out = bs.mass_weighted_scale(E2, mass2, occ2)
    assert abs(out["sd"] - base["sd"]) < 0.5
    assert abs(out["iqr595"] - base["iqr595"]) < 0.5
    assert (E2.max() - E2.min()) > 100 * (E.max() - E.min())   # raw range is fragile


# --------------------------------------------------------------------------- #
# energy-metric layout (sixth basis)
# --------------------------------------------------------------------------- #

def test_energy_metric_layout_recovers_grid_and_is_deterministic():
    """A 3x3 grid graph (A=3, L=2) with a smooth energy (i+j) recovers the grid: the
    layout distance is highly rank-correlated with the Manhattan (graph) distance."""
    from scipy.stats import spearmanr
    L, A = 2, 3
    N = A ** L
    digits = np.stack(np.unravel_index(np.arange(N), (A,) * L)).T
    E = (digits[:, 0] + digits[:, 1]).astype(np.float32)
    P = np.ones(N) / N
    c = bs.energy_metric_layout(E, P, L, A, lam=1.0, K=9, seed=0)
    assert c.shape == (2, N) and c.dtype == np.float32
    assert np.array_equal(c, bs.energy_metric_layout(E, P, L, A, lam=1.0, K=9, seed=0))
    dij = np.linalg.norm(c.T[:, None, :] - c.T[None, :, :], axis=2)
    man = (np.abs(digits[:, 0, None] - digits[None, :, 0])
           + np.abs(digits[:, 1, None] - digits[None, :, 1]))
    m = ~np.eye(N, dtype=bool)
    rho = spearmanr(dij[m].ravel(), man[m].ravel()).correlation
    assert rho > 0.9


def test_energy_metric_slope_is_bounded_by_lambda():
    """The realised energy slope |dE|/||dp|| over graph edges is bounded by ~lambda
    (by construction: each edge has length sqrt(1 + (dE/lam)^2) >= |dE|/lam)."""
    L, A = 2, 3
    N = A ** L
    digits = np.stack(np.unravel_index(np.arange(N), (A,) * L)).T
    E = (digits[:, 0] + digits[:, 1]).astype(np.float32)
    P = np.ones(N) / N
    lam = 1.0
    c = bs.energy_metric_layout(E, P, L, A, lam=lam, K=9, seed=0)
    slopes = []
    for i in range(N):
        for p in range(L):
            off = A ** (L - 1 - p)
            for a in range(A):
                if a != digits[i, p]:
                    j = i + (a - digits[i, p]) * off
                    d = np.linalg.norm(c[:, i] - c[:, j])
                    if d > 1e-6:
                        slopes.append(abs(E[i] - E[j]) / d)
    assert np.percentile(slopes, 95) <= lam * 1.1 + 1e-6


def test_energy_rank_layout_is_monotone_in_energy():
    """The degenerate control (boustrophedon raster of the energy rank) has a row
    coordinate monotone in E (it encodes only the energy rank)."""
    from scipy.stats import spearmanr
    L, A = 3, 4
    N = A ** L
    rng = np.random.default_rng(21)
    E = rng.random(N).astype(np.float32)
    c = bs.energy_rank_layout(E, L, A)
    assert c.shape == (2, N)
    # y (row) is rank // width, hence monotone in E
    rho = spearmanr(c[1], E).correlation
    assert rho > 0.9
    # determinism and independence from E's actual values (only its order matters)
    E2 = np.argsort(np.argsort(E)).astype(np.float32)     # same order, different values
    assert np.array_equal(c, bs.energy_rank_layout(E2, L, A))


def test_compute_basis_energy_metric_returns_coords():
    """compute_basis('energy_metric') returns kind='coords' with a (2, N) float32 array."""
    L, A = 2, 3
    rng = np.random.default_rng(22)
    P = rng.random((A,) * L)
    P = P / P.sum()
    b = bs.compute_basis("energy_metric", P=P, L=L, A=A, K=9)
    assert b["kind"] == "coords"
    assert b["coords"].shape == (2, A ** L)
    assert b["annotation"] is None


def test_occupied_components_counts_real_components():
    """The component counter must count components of the OCCUPIED cells only, not of the
    full grid (a bug that reported '1 component' always)."""
    G = 5
    # two far-apart horizontal pairs -> 2 components under both 4- and 8-connectivity
    occ = np.zeros(G * G, dtype=bool)
    occ[[0, 1, 24, 23]] = True          # (0,0),(0,1) and (4,4),(4,3)
    n4, big4 = bs.occupied_components(occ, G, 4)
    n8, big8 = bs.occupied_components(occ, G, 8)
    assert (n4, big4) == (2, 2)
    assert (n8, big8) == (2, 2)
    # a diagonal touch merges under 8-connectivity but not 4-connectivity
    occ2 = np.zeros(G * G, dtype=bool)
    occ2[[0, 6]] = True                 # (0,0) and (1,1): diagonal
    n4b, _ = bs.occupied_components(occ2, G, 4)
    n8b, _ = bs.occupied_components(occ2, G, 8)
    assert n4b == 2 and n8b == 1
    # a fully-connected 2x2 block is one component under both
    occ3 = np.zeros(G * G, dtype=bool)
    occ3[[0, 1, 5, 6]] = True
    assert bs.occupied_components(occ3, G, 4)[0] == 1
    assert bs.occupied_components(occ3, G, 8)[0] == 1


def test_artifact_bases_json_has_no_control_characters():
    """json-load scripts/artifact_bases.json and assert no control characters (other than
    \\n, \\t) in any string, catching the '\\a'-as-BEL class of bug."""
    import json
    ab = json.load(open("scripts/artifact_bases.json"))

    def scan(o):
        if isinstance(o, str):
            bad = [c for c in o if ord(c) < 32 and c != "\n"]
            assert not bad, f"control characters {[hex(ord(c)) for c in bad]} found"
        elif isinstance(o, dict):
            for v in o.values():
                scan(v)
        elif isinstance(o, list):
            for v in o:
                scan(v)

    scan(ab)
    # sanity: the KaTeX alpha survives as a literal backslash-alpha
    assert "\\alpha" in ab["bases"]["tempered_pca"]["html"]


def test_compute_basis_generic_pca_and_locality():
    # generic feature-map versions run and return valid axes on a small example
    rng = np.random.default_rng(14)
    N, d = 20, 5
    F = rng.normal(size=(N, d))
    w = rng.random(N)
    w /= w.sum()
    E = rng.normal(size=N)
    adj = np.array([[i, (i + 1) % N] for i in range(N)])
    for name in ("pca", "tempered_pca", "energy_aligned", "locality"):
        b = bs.compute_basis_generic(name, F, w, E, adj)
        assert b["U"].shape == (d, 2)
        assert np.allclose(np.linalg.norm(b["U"], axis=0), 1.0)
    try:
        bs.compute_basis_generic("coupling_spectral", F, w, E, adj)
        raise AssertionError("coupling_spectral should not generalise")
    except KeyError:
        pass
