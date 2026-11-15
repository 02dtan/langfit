"""Generate ``tests/data/engine_golden.json``: golden reference values for the
browser-side energy-landscape recomputation (scripts/landscape_engine.js).

For two fitted models from ``data/derived/landscape_artifact.npz`` (Austen and
Twitter 2023), the PCA basis and one non-PCA basis (``coupling_spectral``), and
G in {8, 32, 64, 128}, we store the model couplings, the 2-D basis, the fixed
grid extent, and the quantities the JS engine must reproduce: logZ, S2 (bits),
per-cell E_bin, occupancy counts, P-mass, top-3 words, persistent wells and the
main barrier (both on the 8-neighbour grid and, for G in {8, 32}, on the
substitution adjacency actually used by the artifact). One ``coords`` case
(model = Austen, G = 32) exercises the non-parametric basis interface with
synthetic per-string coordinates.

The golden is **self-contained**: every case embeds the exact inputs its outputs
were derived from (the couplings ``V``, and the basis ``U1``/``U2``/``mean`` or
the ``coords``), so the output derivation can be re-checked without the cache
(``tests/test_engine_golden.py`` recomputes each case from its embedded inputs).
The cache itself is only needed to *produce* the golden; ``meta.input_hash`` is a
SHA-256 of the fitted couplings, and a separate cheap test compares it to the
current cache so that input drift is reported explicitly instead of as a
mysterious output diff.

Everything is deterministic (eigendecompositions plus a seeded RNG for the
synthetic coordinates), so regenerating in the same environment reproduces this
file byte-for-byte.

Encoding conventions (so the JS parity runner can round-trip floats exactly):
  - V, U1, U2 are base64 of float64 bytes (little-endian, C order).
  - coords is base64 of float32 bytes (x of string id ``i`` at ``coords[i]``,
    y at ``coords[26**4 + i]``).
  - Ebin entries: a rounded float for finite cells, ``null`` for empty cells
    (NaN), and the string ``"inf"`` for all-forbidden cells (+Infinity).
  - E_saddle of an essential well is ``"inf"``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langfit import basis as bas, maxent as me, projection as pj  # noqa: E402

NPZ = ROOT / "data" / "derived" / "landscape_artifact.npz"
OUT = ROOT / "tests" / "data" / "engine_golden.json"

G_GRID = (8, 32, 64, 128)
SUB_G_GRID = (8, 32)            # G at which the substitution adjacency is also checked
TOP_K = 3
PERSIST_BITS = 1.0              # artifact's well/barrier labelling threshold
LN2 = np.log(2.0)
A = 26
L = 4
PROBE_WORDS = ("have", "that", "lamb", "zxqj")


def _b64_f64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float64).tobytes()).decode()


def _b64_f32(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def b64_decode(b: str, dtype) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b), dtype=dtype)


def _enc(x, nd: int = 9):
    """Encode a float for JSON: None (NaN), "inf" (+Inf), else rounded."""
    f = float(x)
    if np.isnan(f):
        return None
    if np.isinf(f):
        return "inf"
    return round(f, nd)


def _unpack(V: np.ndarray) -> me.MaxEnt:
    return me.MaxEnt(L=L, cliques=me.cliques(2), V=[v.copy() for v in V], lam=None)


def _pca_basis(m: me.MaxEnt):
    C = pj.indicator_covariance(m.prob())
    _lam, U = pj.principal_axes(C)
    mean = pj.site_marginals(m.prob()).ravel()
    return U[:, :2], mean


def _named_basis(m: me.MaxEnt, name: str):
    r = bas.compute_basis(name, model=m, L=L, A=A)
    return r["U"], r["mean"]


def _well(w: dict) -> dict:
    return {"cell": int(w["cell"]),
            "E_min": round(float(w["E_min"]), 9),
            "E_saddle": _enc(w["E_saddle"]),
            "persistence": round(float(w["persistence"]), 9),
            "saddle": w["saddle"],
            "merges_into": w["merges_into"],
            "essential": bool(w["essential"])}


def _wells_payload(E: np.ndarray, adj):
    allw = pj.persistence(E, adj)
    by_cell = {w["cell"]: w for w in allw}
    wells = [_well(w) for w in allw]
    barrier = None
    merging = [w for w in allw if w["persistence"] >= PERSIST_BITS and not w["essential"]]
    if merging:
        w = merging[0]
        other = by_cell[w["merges_into"]]
        barrier = {"cell": int(w["saddle"]), "E": round(float(w["E_saddle"]), 9),
                   "wells": [int(w["cell"]), int(other["cell"])],
                   "above": [round(float(w["E_saddle"] - w["E_min"]), 9),
                             round(float(w["E_saddle"] - other["E_min"]), 9)]}
    return wells, barrier


def _top_words(log2p: np.ndarray, bins: np.ndarray, n_cells: int, k: int):
    """Per cell, the top-k string ids by log2p desc (stable -> ascending id)."""
    ids = np.full((n_cells, k), -1, dtype=np.int64)
    order = np.lexsort((-log2p, bins))
    b = bins[order]
    first = np.r_[0, np.flatnonzero(np.diff(b)) + 1]
    starts = {cell: i for cell, i in zip(b[first], first)}
    for cell in range(n_cells):
        i = starts.get(cell)
        if i is None:
            continue
        j, kk = i, 0
        while j < b.size and b[j] == cell and kk < k:
            if np.isfinite(log2p[order[j]]):
                ids[cell, kk] = int(order[j])
                kk += 1
            j += 1
    return ids


def _landscape(m: me.MaxEnt, coords: np.ndarray, G: int):
    """E_bin, counts, mass, top words, wells for one (model, basis, G) at fixed coords.

    The JS engine bins in float64 (float32 coordinates are widened), so the reference
    is computed on float64 coordinates too; otherwise the float32 grid-edge/binning
    arithmetic of projection.bin_index would flip a handful of boundary strings.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n_cells = G * G
    edges = pj.grid_edges(coords, G)
    bins = pj.bin_index(coords, edges)
    lw = m.log_weights().ravel()
    logZ = float(np.log(np.exp(lw - lw.max()).sum()) + lw.max())
    log2p = (lw - logZ) / LN2
    Ebin, count = pj.bin_energy(log2p, bins, n_cells)
    p = np.exp(lw - logZ)
    mass = np.bincount(bins, weights=p, minlength=n_cells)
    S2bits = float(-(p[p > 0] * (lw[p > 0] - logZ)).sum() / LN2)
    top3 = _top_words(log2p, bins, n_cells, TOP_K)
    out = {
        "logZ": logZ, "S2bits": S2bits,
        "Ebin": [_enc(v) for v in Ebin],
        "count": count.astype(int).tolist(),
        "mass": [round(float(v), 9) for v in mass],
        "top_words": top3.ravel().tolist(),
    }
    wells8, barrier8 = _wells_payload(Ebin, pj.grid_adjacency((G, G)))
    out["wells_grid8"] = wells8
    out["barrier_grid8"] = barrier8
    if G in SUB_G_GRID:
        adj = pj.substitution_adjacency(bins.reshape((A,) * L), n_cells)
        wells_s, barrier_s = _wells_payload(Ebin, adj)
        out["wells_sub"] = wells_s
        out["barrier_sub"] = barrier_s
    else:
        out["wells_sub"] = None
        out["barrier_sub"] = None
    return out


def _synthetic_coords() -> np.ndarray:
    """Deterministic synthetic per-string coordinates (2, 26^4) in float32."""
    rng = np.random.default_rng(20260123)
    Ur = rng.normal(size=(L * A, 2))
    return pj.project(Ur, np.zeros(L * A), L, A).astype(np.float32)


def _case_from_V(model_key: str, V: np.ndarray, basis_name: str, G: int,
                 coords: np.ndarray | None = None) -> dict:
    """One golden case derived from pinned inputs (V and optionally coords)."""
    V = np.asarray(V, dtype=np.float64).reshape(6, A, A)
    m = _unpack(V)
    if coords is not None:
        coords = np.asarray(coords, dtype=np.float32).reshape(2, -1)
        extent = {"xmin": float(coords[0].min()), "xmax": float(coords[0].max()),
                  "ymin": float(coords[1].min()), "ymax": float(coords[1].max())}
        ls = _landscape(m, coords, G)
        return {"model": model_key, "basis": basis_name, "G": G, "coords": True,
                "V": _b64_f64(V), "coords": _b64_f32(coords), "extent": extent, **ls}

    U, mean = _pca_basis(m) if basis_name == "pca" else _named_basis(m, basis_name)
    c = pj.project(U, mean, L, A)
    extent = {"xmin": float(c[0].min()), "xmax": float(c[0].max()),
              "ymin": float(c[1].min()), "ymax": float(c[1].max())}
    ls = _landscape(m, c, G)
    return {"model": model_key, "basis": basis_name, "G": G, "coords": False,
            "V": _b64_f64(V), "U1": _b64_f64(U[:, 0]), "U2": _b64_f64(U[:, 1]),
            "mean": [float((U.T @ mean)[0]), float((U.T @ mean)[1])],
            "extent": extent, **ls}


def _probe(m: me.MaxEnt) -> list:
    """Reference energy (bits) and probability of a few known strings, straight from
    maxent.MaxEnt, for an independent cross-check of the engine's energyAt/probAt."""
    out = []
    for w in PROBE_WORDS:
        idx = tuple(ord(ch) - 97 for ch in w)
        e = m.energy()[idx]
        p = m.prob()[idx]
        out.append({"word": w, "id": int(me.ravel(w)),
                    "E_bits": _enc(e / LN2, nd=6),
                    "prob": round(float(p), 9)})
    return out


def input_hash(npz) -> str:
    """SHA-256 of the fitted couplings the golden is produced from."""
    h = hashlib.sha256()
    for k in ("V_austen", "V_twitter2023"):
        h.update(np.ascontiguousarray(npz[k], dtype=np.float64).tobytes())
    return h.hexdigest()


def generate(out_path: Path = OUT) -> dict:
    npz = np.load(NPZ)
    models = {"austen": npz["V_austen"], "twitter2023": npz["V_twitter2023"]}
    cases = []
    for model_key, V in models.items():
        for basis_name in ("pca", "coupling_spectral"):
            for G in G_GRID:
                cases.append(_case_from_V(model_key, V, basis_name, G))
    cases.append(_case_from_V("austen", models["austen"], "energy_metric", 32,
                              coords=_synthetic_coords()))

    doc = {
        "meta": {
            "note": "golden reference for scripts/landscape_engine.js; see "
                    "scripts/engine_golden.py for the encoding conventions.",
            "TOP_K": TOP_K, "PERSIST_BITS": PERSIST_BITS,
            "G_GRID": list(G_GRID), "SUB_G_GRID": list(SUB_G_GRID),
            "input_hash": input_hash(npz),
            "probe": _probe(_unpack(models["austen"])),
        },
        "cases": cases,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, separators=(",", ":")))
    return doc


def main(argv) -> None:
    out = Path(argv[0]) if argv else OUT
    doc = generate(out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(doc['cases'])} cases)")


if __name__ == "__main__":
    main(sys.argv[1:])
