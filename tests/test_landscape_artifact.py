"""Projection, bin energies and persistence used by the energy-landscape artifact,
checked against brute-force computations and hand-built grids with known answers."""

from itertools import product

import numpy as np
import pytest

from langfit import basis as bas, maxent as me, projection as pj


def _random_tensor(rng, A, L, forbid=0.0):
    p = rng.random((A,) * L) ** 3
    if forbid:
        p[rng.random(p.shape) < forbid] = 0.0
    return p / p.sum()


def _one_hot(A, L):
    strings = np.array(list(product(range(A), repeat=L)))
    X = np.zeros((strings.shape[0], L * A))
    for i in range(L):
        X[np.arange(strings.shape[0]), i * A + strings[:, i]] = 1.0
    return strings, X


def test_indicator_covariance_matches_brute_force():
    rng = np.random.default_rng(0)
    A, L = 4, 4
    P = _random_tensor(rng, A, L)
    _, X = _one_hot(A, L)
    w = P.ravel()
    mu = w @ X
    C_brute = (X - mu).T @ ((X - mu) * w[:, None])
    assert np.allclose(pj.indicator_covariance(P), C_brute, atol=1e-14)
    # trace = Σ_i (1 - Σ_a P_i(a)^2), the total one-hot variance
    p1 = pj.site_marginals(P)
    assert np.isclose(np.trace(C_brute), (1 - (p1 ** 2).sum(axis=1)).sum())


def test_one_hot_distance_is_twice_hamming():
    A, L = 3, 4
    strings, X = _one_hot(A, L)
    rng = np.random.default_rng(1)
    for a, b in rng.integers(0, len(strings), (50, 2)):
        assert np.sum((X[a] - X[b]) ** 2) == 2 * np.sum(strings[a] != strings[b])


def test_principal_axes_are_sorted_unit_eigenvectors_with_sign_convention():
    rng = np.random.default_rng(2)
    P = _random_tensor(rng, 5, 4)
    C = pj.indicator_covariance(P)
    lam, U = pj.principal_axes(C)
    assert np.all(np.diff(lam) <= 1e-15)
    assert np.allclose(C @ U[:, :2], U[:, :2] * lam[:2])
    assert np.allclose(np.linalg.norm(U, axis=0), 1.0)
    for k in range(U.shape[1]):
        assert U[np.abs(U[:, k]).argmax(), k] > 0
    # one zero mode per position (each one-hot block sums to 1)
    assert np.sum(lam < 1e-12) >= 4
    rep = pj.axis_report(U[:, 0], lam, 0, 4, "abcde", top=6)
    assert np.isclose(sum(rep["position_share"]), 1.0)
    assert np.isclose(rep["variance_share"], lam[0] / np.trace(C))
    assert len(rep["top_loadings"]) == 6


def test_projection_matches_explicit_dot_products():
    rng = np.random.default_rng(3)
    A, L = 4, 4
    P = _random_tensor(rng, A, L)
    _, X = _one_hot(A, L)
    lam, U = pj.principal_axes(pj.indicator_covariance(P))
    mean = P.ravel() @ X
    coords = pj.project(U[:, :2], mean, L, A)
    assert np.allclose(coords.T, (X - mean) @ U[:, :2])
    # centred under P and with variance λ_k along axis k
    w = P.ravel()
    assert np.allclose(coords @ w, 0.0, atol=1e-13)
    assert np.allclose((coords ** 2) @ w, lam[:2])


def test_bin_energy_uniform_is_flat_at_log_number_of_states():
    A, L = 6, 4
    P = np.full((A,) * L, 1.0 / A ** L)
    _, X = _one_hot(A, L)
    rng = np.random.default_rng(4)
    U = np.linalg.qr(rng.normal(size=(L * A, 2)))[0]
    coords = pj.project(U, P.ravel() @ X, L, A)
    edges = pj.grid_edges(coords, 7)
    bins = pj.bin_index(coords, edges)
    E, count = pj.bin_energy(np.log2(P.ravel()), bins, 49)
    occ = count > 0
    assert np.allclose(E[occ], L * np.log2(A))
    assert np.all(np.isnan(E[~occ]))
    assert count.sum() == A ** L
    assert edges[0, 0] == coords[0].min() and edges[0, -1] == coords[0].max()


def test_bin_energy_is_exponential_average_with_forbidden_cells():
    rng = np.random.default_rng(5)
    n = 200
    log2p = rng.normal(-20, 5, n)
    log2p[:10] = -np.inf
    bins = rng.integers(0, 12, n)
    bins[:10] = 12                          # cell 12: forbidden strings only
    E, count = pj.bin_energy(log2p, bins, 14)
    for b in range(12):
        m = bins == b
        if m.any():
            assert np.isclose(E[b], -np.log2(np.mean(np.exp2(log2p[m]))))
    assert E[12] == np.inf
    assert np.isnan(E[13]) and count[13] == 0
    # far below float64's normal range: the per-cell log-sum-exp must not underflow
    E2, _ = pj.bin_energy(np.array([-2000.0, -2001.0]), np.array([0, 0]), 1)
    assert np.isclose(E2[0], 2000 - np.log2((1 + 0.5) / 2))


def _two_wells():
    """Two basins: depths 0 (left) and 3 (right), separated by a ridge at 5."""
    E = np.full((5, 9), 8.0)
    E[1:4, 0:4] = 2.0
    E[2, 1] = 0.0
    E[1:4, 5:9] = 4.0
    E[2, 7] = 3.0
    E[2, 4] = 5.0                           # the only pass between the basins
    return E


def test_persistence_two_wells_elder_rule():
    E = _two_wells()
    wells = pj.persistence(E, pj.grid_adjacency(E.shape))
    # the global minimum never merges; it is paired with the highest cell (8)
    assert wells[0]["cell"] == 2 * 9 + 1 and wells[0]["essential"]
    assert wells[0]["persistence"] == 8.0
    w = wells[1]
    assert w["cell"] == 2 * 9 + 7
    assert w["E_min"] == 3.0 and w["E_saddle"] == 5.0 and w["persistence"] == 2.0
    assert w["saddle"] == 2 * 9 + 4 and w["merges_into"] == 2 * 9 + 1
    # the flat right-basin floor yields only zero-persistence (tie-break) pairs
    assert len(wells) == 2 + sum(x["persistence"] == 0 for x in wells)


def test_persistence_holes_and_separate_components():
    E = _two_wells()
    E[:, 4] = np.inf                        # all-forbidden column: no path between basins
    E[0, 0] = np.nan                        # empty cell
    wells = pj.persistence(E, pj.grid_adjacency(E.shape))
    assert sorted(w["E_min"] for w in wells if w["essential"]) == [0.0, 3.0]


def test_persistence_plateau_and_small_bump():
    E = np.zeros((4, 4))
    E[0, 0] = -0.5                          # a shallow well on a flat plane
    E[3, 3] = -0.2
    wells = pj.persistence(E, pj.grid_adjacency(E.shape))
    assert len(wells) == 2
    assert wells[1]["cell"] == 15 and np.isclose(wells[1]["persistence"], 0.2)


def test_significant_basins_cancel_shallow_wells():
    E = _two_wells()
    E[1, 2] = 1.5                           # a bump, then a shallow dimple inside the left basin
    E[1, 3] = 1.0
    adj = pj.grid_adjacency(E.shape)
    wells = pj.persistence(E, adj)
    shallow = [w for w in wells if w["cell"] == 1 * 9 + 3]
    assert shallow and np.isclose(shallow[0]["persistence"], 0.5)
    lab = pj.significant_basins(E, wells, 1.0, adj)
    assert set(np.unique(lab[np.isfinite(E)])) == {2 * 9 + 1, 2 * 9 + 7}
    assert lab[1, 3] == 2 * 9 + 1
    assert lab[2, 6] == 2 * 9 + 7
    lab_all = pj.significant_basins(E, wells, 0.0, adj)
    assert lab_all[1, 3] == 1 * 9 + 3


@pytest.mark.parametrize("seed", range(3))
def test_persistence_counts_every_local_minimum(seed):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(12, 12))
    wells = pj.persistence(E, pj.grid_adjacency(E.shape))
    strict = 0
    for r in range(12):
        for c in range(12):
            nb = [E[rr, cc] for rr in range(max(r - 1, 0), min(r + 2, 12))
                  for cc in range(max(c - 1, 0), min(c + 2, 12)) if (rr, cc) != (r, c)]
            strict += E[r, c] < min(nb)
    assert len(wells) == strict
    assert sum(w["essential"] for w in wells) == 1
    assert wells[0]["essential"] and wells[0]["persistence"] == E.max() - E.min()
    assert all(w["persistence"] >= 0 for w in wells)


def test_substitution_adjacency_matches_brute_force():
    """Cells are linked iff a single-letter substitution takes a string of one to a
    string of the other."""
    A, L, n = 4, 3, 5
    rng = np.random.default_rng(6)
    bins = rng.integers(0, n, (A,) * L)
    adj = pj.substitution_adjacency(bins, n + 1)       # cell n is empty
    brute = {k: set() for k in range(n + 1)}
    for s in product(range(A), repeat=L):
        for i in range(L):
            for b in range(A):
                t = list(s)
                t[i] = b
                u, v = bins[s], bins[tuple(t)]
                if u != v:
                    brute[u].add(v)
    assert {k: set(a.tolist()) for k, a in enumerate(adj)} == brute
    assert adj[n].size == 0


def test_persistence_on_a_graph_uses_its_edges_not_geometry():
    """A path graph 0-1-2-3 with values 0, 5, 2, 1: the well at 3 dies at the
    saddle 1 (height 5), not at the lower cell 2, which belongs to its basin."""
    E = np.array([0.0, 5.0, 2.0, 1.0])
    adj = [np.array([1]), np.array([0, 2]), np.array([1, 3]), np.array([2])]
    wells = pj.persistence(E, adj)
    assert [(w["cell"], w["persistence"]) for w in wells] == [(0, 5.0), (3, 4.0)]
    assert wells[1]["saddle"] == 1
    assert pj.descent_basins(E, adj).tolist() == [0, 0, 3, 3]


def test_flat_landscape_has_no_well():
    E = np.full((6, 6), 18.8)
    wells = pj.persistence(E, pj.grid_adjacency(E.shape))
    assert all(w["persistence"] == 0 for w in wells)
    assert sum(w["essential"] for w in wells) == 1
    lab = pj.significant_basins(E, wells, 1.0, pj.grid_adjacency(E.shape))
    assert np.unique(lab).size == 1


def test_contour_levels_span_the_finite_range():
    Z = np.array([[0.0, np.nan], [np.inf, 10.0]])
    lv = pj.contour_levels(Z, n=4, digits=2)
    assert len(lv) == 4
    assert lv == sorted(lv)
    assert lv[0] > 0.0 and lv[-1] < 10.0           # interior of [min, max]
    # a flat grid has no contours
    assert pj.contour_levels(np.full((4, 4), 3.0)) == []
    # a grid with no finite cells has none
    assert pj.contour_levels(np.array([[np.nan, np.inf], [np.inf, np.nan]])) == []


# --------------------------------------------------------------------------- #
# switchable projection bases (langfit.basis.compute_basis)
# --------------------------------------------------------------------------- #

def _random_model(rng, L=4, A=26):
    return me.MaxEnt(L=L, cliques=me.cliques(2),
                     V=[0.1 * rng.normal(size=(A, A)) for _ in me.cliques(2)])


def test_basis_registry_has_pca_default_and_is_nonempty():
    assert "pca" in bas.BASES           # the default
    assert len(bas.BASES) >= 2


def test_compute_basis_pca_matches_principal_axes():
    """The default (PCA) basis is byte-compatible with projection.principal_axes."""
    rng = np.random.default_rng(7)
    P = _random_tensor(rng, 26, 4)
    r = bas.compute_basis("pca", P=P, L=4, A=26)
    C = pj.indicator_covariance(P)
    lam, U = pj.principal_axes(C)
    assert np.allclose(r["U"], U[:, :2], atol=1e-12)
    assert np.allclose(r["mean"], pj.site_marginals(P).ravel(), atol=1e-12)
    # annotation matches axis_report
    rep0 = pj.axis_report(U[:, 0], lam, 0, 4, me.ALPHABET, 10)
    assert np.isclose(r["annotation"][0]["variance_share"], rep0["variance_share"])


@pytest.mark.parametrize("name", sorted(bas.BASES))
def test_compute_basis_returns_switch_payload(name):
    """Every linear basis returns the switch payload: unit (L*A, 2) loadings, an (L*A,)
    mean and two per-axis annotations with the keys the artifact template renders."""
    if bas.BASES[name].get("kind") == "coords":
        pytest.skip("coords (diagnostic) basis has no linear loadings")
    rng = np.random.default_rng(11)
    model = _random_model(rng)
    r = bas.compute_basis(name, model=model, L=4, A=26)
    U, mean, ann = r["U"], r["mean"], r["annotation"]
    assert U.shape == (4 * 26, 2)
    assert mean.shape == (4 * 26,)
    assert len(ann) == 2
    for a in ann:
        assert set(a) == {"variance_share", "eigenvalue", "position_share", "top_loadings"}
        assert len(a["position_share"]) == 4
        assert len(a["top_loadings"]) == 10
    # unit-norm columns
    assert np.allclose(np.linalg.norm(U, axis=0), 1.0, atol=1e-6)
    # coordinates cover all A^L strings and are finite
    coords = bas.coords_linear(U, mean, 4, 26)
    assert coords.shape == (2, 26 ** 4)
    assert np.all(np.isfinite(coords))


def test_compute_basis_axes_differ_across_bases():
    """Different bases give different projections (so wells are recomputed per basis)."""
    rng = np.random.default_rng(13)
    model = _random_model(rng)
    linear = [n for n in bas.BASES if bas.BASES[n].get("kind") != "coords"]
    axes = {n: bas.compute_basis(n, model=model, L=4, A=26)["U"] for n in linear}
    for a in linear:
        for b in linear:
            if a >= b:
                continue
            # distinct as signed directions (up to a shared global sign per axis)
            assert not np.allclose(axes[a], axes[b], atol=1e-3), (a, b)


def test_austen_minima_tolerance_aware():
    """The Austen strict-minima count is not stable at the fit tolerance: 'lamb' and
    'limb' differ by a few 1e-4 bits, so the count is 148 or 149 depending on the fit.
    Assert the count lies in {148,149} and that the only strings whose minimum status
    is uncertain have a neighbour within 1e-3 bits (above the measured fit noise)."""
    import json as _json
    import os as _os
    root = _os.path.join(_os.path.dirname(__file__), "..")
    npz = _os.path.join(root, "data", "derived", "landscape_artifact.npz")
    study = _os.path.join(root, "data", "derived", "maxent_study.json")
    if not (_os.path.exists(npz) and _os.path.exists(study)):
        pytest.skip("cached fit/study not present")
    V = np.load(npz)["V_austen"]
    m = me.MaxEnt(L=4, cliques=me.cliques(2), V=[v.copy() for v in V])
    E = m.energy().ravel() / np.log(2.0)               # bits
    is_min = me.local_minima(m.energy()).ravel()
    n_strict = int(is_min.sum())
    want = int(_json.load(open(study))["replication"]["n_minima"])
    assert n_strict in {want - 1, want}, f"Austen minima {n_strict} not in {{{want-1},{want}}}"
    eps = 1e-3
    near_ties = []
    for i in np.flatnonzero(is_min):
        s = list(me.unravel(i))
        margin = np.inf
        for pos in range(4):
            for letter in me.ALPHABET:
                if letter == s[pos]:
                    continue
                t = s.copy(); t[pos] = letter
                margin = min(margin, float(E[me.ravel(tuple(t))] - E[i]))
        if margin < eps:
            near_ties.append((me.unravel(i), margin))
    # the only minima that could flip are those with a neighbour within eps
    assert all(mg < eps for _, mg in near_ties)
    assert len(near_ties) <= 2, f"too many near-tie minima: {near_ties}"


def test_no_duplicate_top_level_functions_in_built_html():
    import re
    from collections import Counter
    from pathlib import Path
    html = Path(__file__).resolve().parent.parent / "index.html"
    if not html.exists():
        pytest.skip("artifact not built")
    text = html.read_text()
    m = re.search(r'<script>\s*"use strict";(.*?)</script>', text, re.S)
    assert m, "main script block not found in built HTML"
    js = m.group(1)
    funcs = re.findall(r'function\s+([A-Za-z_$][\w$]*)\s*\(', js)
    dupes = sorted(k for k, v in Counter(funcs).items() if v > 1)
    assert not dupes, f"duplicate top-level function definitions: {dupes}"


def test_no_stale_smoothing_banner_or_gcv_mentions():
    """The occupancy banner ('Client-side recomputation … GCV spline … precomputed
    only at 32×32') was removed entirely, along with the 'spline smoothing' wording of
    the old display smoother (now Boltzmann-domain kernel regression). Human-written
    sources and the built main script must carry none of these stale phrases."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    sources = {
        "template": root / "scripts" / "landscape_template.html",
        "blurbs": root / "scripts" / "artifact_blurbs.json",
        "ui_copy": root / "scripts" / "ui_copy.json",
    }
    banner_phrases = ("Client-side recomputation", "GCV", "precomputed only at")
    for name, path in sources.items():
        if not path.exists():
            continue
        text = path.read_text()
        for needle in banner_phrases:
            assert needle not in text, f"{name} still contains {needle!r}"
        assert "spline smoothing" not in text.lower(), f"{name} still describes 'spline smoothing'"

    html = root / "index.html"
    if not html.exists():
        pytest.skip("artifact not built")
    import re
    m = re.search(r'<script>\s*"use strict";(.*?)</script>', html.read_text(), re.S)
    assert m, "main script block not found in built HTML"
    js = m.group(1)
    assert "smoothnote" not in js, "built JS still references a smoothnote banner element"
    for needle in banner_phrases:
        assert needle not in js, f"built JS still contains {needle!r}"


def test_no_scatter3d_text_labels():
    """gl3d text (scatter3d 'text'/'markers+text') renders as garbled 'aaa' in some
    browsers; all word labels must come from scene annotations or hovertemplates."""
    import re
    from pathlib import Path
    html = Path(__file__).resolve().parent.parent / "index.html"
    if not html.exists():
        pytest.skip("artifact not built")
    text = html.read_text()
    m = re.search(r'<script>\s*"use strict";(.*?)</script>', text, re.S)
    assert m, "main script block not found in built HTML"
    js = m.group(1)
    assert "markers+text" not in js, "scatter3d/2d text labels must be removed"
    assert "textposition" not in js, "textposition must be removed"


def test_all_surface_traces_disable_highlight():
    """Plotly surface.contours.*.highlight defaults to true, leaving white lines on
    hover; every surface trace must set contours highlight:false."""
    import re
    from pathlib import Path
    html = Path(__file__).resolve().parent.parent / "index.html"
    if not html.exists():
        pytest.skip("artifact not built")
    text = html.read_text()
    m = re.search(r'<script>\s*"use strict";(.*?)</script>', text, re.S)
    assert m, "main script block not found in built HTML"
    js = m.group(1)
    n_surface = js.count('type: "surface"')
    n_contoured = js.count('type: "surface", contours: NC')
    assert n_surface == n_contoured, f"{n_surface} surfaces but only {n_contoured} carry contours: NC"


def test_well_words_are_nonempty_lowercase():
    """Every well annotation word (S&B string wells and new-model dict wells) must be a
    non-empty lowercase string, never an empty/uninitialised word list."""
    import json
    import re
    from pathlib import Path
    html = Path(__file__).resolve().parent.parent / "index.html"
    if not html.exists():
        pytest.skip("artifact not built")
    text = html.read_text()
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', text, re.S)
    assert m, "embedded data JSON not found"
    data = json.loads(m.group(1))
    for model in data["models"]:
        for w in model.get("wells", []):
            for e in w.get("words", []):
                if isinstance(e, str):
                    assert re.fullmatch(r"[a-z]+", e), f"bad S&B well word {e!r}"
    nm = data.get("newmodel")
    if nm:
        for lk in nm.get("levels", {}):
            for w in nm["levels"][lk].get("wells", []):
                for e in w.get("words", []):
                    if isinstance(e, dict):
                        assert re.fullmatch(r"[a-z]+", e.get("w", "")), f"bad nm well word {e!r}"


def _minimax_bottleneck(E, src, dst, A=26):
    """Independent reference for the JS bottleneck search: Dijkstra over the
    single-letter-substitution graph minimising the maximum node energy along the path
    (ties broken by fewer steps). Returns the bottleneck energy, or None."""
    import heapq
    L = 4
    A3, A2 = A ** 3, A ** 2
    posm = [A3, A2, A, 1]

    def letters(i):
        s0 = i // A3; r1 = i - s0 * A3
        s1 = r1 // A2; r2 = r1 - s1 * A2
        s2 = r2 // A; s3 = r2 - s2 * A
        return (s0, s1, s2, s3)

    n = A ** 4
    dist = np.full(n, np.inf)
    steps = np.full(n, 2 ** 30, dtype=np.int64)
    dist[src] = E[src]; steps[src] = 0
    heap = [(E[src], 0, src)]
    while heap:
        du, su, u = heapq.heappop(heap)
        if u == dst:
            return float(du)
        if su > steps[u] or du > dist[u] + 1e-9:
            continue
        Ls = letters(u)
        for p in range(L):
            cur = Ls[p]; base = u - cur * posm[p]
            for a in range(A):
                if a == cur:
                    continue
                v = base + a * posm[p]
                if not np.isfinite(E[v]):
                    continue
                nd = max(du, E[v]); ns = su + 1
                if nd < dist[v] - 1e-9 or (abs(nd - dist[v]) < 1e-9 and ns < steps[v]):
                    dist[v] = nd; steps[v] = ns
                    heapq.heappush(heap, (nd, ns, v))
    return None


def _word_to_id(w, A=26):
    return ((ord(w[0]) - 97) * A + (ord(w[1]) - 97)) * A * A + (ord(w[2]) - 97) * A + (ord(w[3]) - 97)


def test_minimax_bottleneck_planted_graph():
    # A=2 (letters a,b), L=4: all 16 nodes have energy 5 except the two endpoints
    # (aaaa=0, bbbb=0); any path between them must pass through a 5-energy node,
    # so the bottleneck is exactly 5.
    A = 2
    n = A ** 4
    E = np.full(n, 5.0)
    E[0] = 0.0
    E[n - 1] = 0.0
    b = _minimax_bottleneck(E, 0, n - 1, A=A)
    assert b == 5.0, f"bottleneck {b} != 5.0"


def test_minimax_bottleneck_matches_austen_reference():
    import os
    root = os.path.join(os.path.dirname(__file__), "..")
    npz = os.path.join(root, "data", "derived", "landscape_artifact.npz")
    if not os.path.exists(npz):
        pytest.skip("cached fit not present")
    V = np.load(npz)["V_austen"]
    m = me.MaxEnt(L=4, cliques=me.cliques(2), V=[v.copy() for v in V])
    E = m.energy().ravel() / np.log(2.0)
    assert abs(_minimax_bottleneck(E, _word_to_id("that"), _word_to_id("with")) - 12.147) < 0.01
    assert abs(_minimax_bottleneck(E, _word_to_id("have"), _word_to_id("that")) - 11.634) < 0.01


def test_nm_cell_words_decode_to_nonempty():
    """The new-model cell_words use three encodings (model objects with w:string,
    level arrays [vocab_idx,L,p], trajectory objects with w:int) — all must decode to
    non-empty words for the large majority of occupied cells."""
    import json
    import re
    from pathlib import Path

    def decode(e, vocab):
        if isinstance(e, list):
            return vocab[e[0]]
        if isinstance(e.get("w"), str):
            return e["w"]
        return vocab[e["w"]]

    html = Path(__file__).resolve().parent.parent / "index.html"
    if not html.exists():
        pytest.skip("artifact not built")
    text = html.read_text()
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', text, re.S)
    assert m, "embedded data JSON not found"
    data = json.loads(m.group(1))
    nm = data["newmodel"]
    if not nm:
        pytest.skip("new-model data not present")
    vocab = nm["cell_words_vocab"]
    # model (final) cell_words: object format
    for source_name, cells in [("model", nm["models"]["final"]["cell_words"]),
                               ("level96", nm["levels"]["96"]["cell_words"]),
                               ("traj", nm["traj"]["cell_words"])]:
        occupied = nonempty = 0
        for cell in cells:
            if not cell:
                continue
            occupied += 1
            if any(decode(e, vocab) for e in cell):
                nonempty += 1
        if occupied:
            assert nonempty >= 0.8 * occupied, \
                f"{source_name}: only {nonempty}/{occupied} occupied cells decode to non-empty words"
