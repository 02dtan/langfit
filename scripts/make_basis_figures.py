"""Generate figures/basis_compare_<model>.png: the raw (uncapped) E_bin surface of the
five selectable bases at 32x32, one panel per basis, shared viridis range, wells marked.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import langfit.maxent as me
import langfit.projection as pj
from langfit import basis as bs

L4, A = 4, me.A
G = 32
BASIS_NAMES = ["pca", "coupling_spectral", "energy_aligned", "tempered_pca", "lda"]
MODELS = [("austen", "(a) Austen"), ("twitter2023", "(b) Twitter 2023"),
          ("bluesky2024", "(c) Bluesky 2024"), ("pooled", "(d) Twitter pooled")]


def load_model(key):
    npz = np.load("data/derived/landscape_artifact.npz")
    V = npz["V_traj"][-1] if key == "pooled" else npz[f"V_{key}"]
    m = me.MaxEnt(L=L4, cliques=me.cliques(2), V=[v.copy() for v in V])
    npz.close()
    return m


def surfaces(model):
    lp = model.log_prob().ravel() / np.log(2)
    out = {}
    for name in BASIS_NAMES:
        b = bs.compute_basis(name, model=model)
        coords = bs.coords_linear(b["U"], b["mean"], L4, A)
        bins = pj.bin_index(coords, pj.grid_edges(coords, G))
        E, _ = pj.bin_energy(lp, bins, G * G)
        wells = [w for w in pj.persistence(E, pj.grid_adjacency((G, G)))
                 if w["persistence"] >= 1.0]
        out[name] = (E.reshape(G, G), wells)
    return out


def make_figure(key, label):
    model = load_model(key)
    surfs = surfaces(model)
    fin = np.concatenate([s[0][np.isfinite(s[0])] for s in surfs.values()])
    # robust shared colour range (2nd-98th percentile of the pooled finite values),
    # so a handful of >100-bit outlier cells do not compress the figure; the raw
    # (uncapped) numbers are in basis_study.json
    vmin = float(np.percentile(fin, 2))
    vmax = float(np.percentile(fin, 98))

    fig, axes = plt.subplots(1, 5, figsize=(24, 5.4))
    for ax, (name, (E, wells)) in zip(axes, surfs.items()):
        em = np.where(np.isfinite(E), E, np.nan)
        im = ax.imshow(em, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        for w in wells:
            r, c = divmod(int(w["cell"]), G)
            ax.plot(c, r, marker="o", ms=9, mfc="none", mec="white", mew=1.2)
        ax.set_title(bs.BASES[name]["label"], fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.92, 0.14, 0.012, 0.72])       # dedicated colour-bar axis
    fig.colorbar(im, cax=cax, label="E_bin (bits, uncapped)")
    fig.suptitle(f"{label} — raw E_bin per basis (shared 2-98% colour range; wells >= 1 bit circled)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path("figures") / f"basis_compare_{key}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}  (2-98% range {vmin:.1f}..{vmax:.1f} bits)")


if __name__ == "__main__":
    for key, label in MODELS:
        make_figure(key, label)
