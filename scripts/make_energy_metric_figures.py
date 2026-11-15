"""Figures for the energy-metric basis: extend figures/basis_compare_<model>.png to six
panels (add the energy-metric layout) and write figures/energy_metric_controls_<model>.png
(rank, Hamming, and the pooled lambda sweep).  Reads the cached layouts."""
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


def load_model(key):
    npz = np.load("data/derived/landscape_artifact.npz")
    V = npz["V_traj"][-1] if key == "pooled" else npz[f"V_{key}"]
    m = me.MaxEnt(L=L4, cliques=me.cliques(2), V=[v.copy() for v in V])
    npz.close()
    return m


def surface(lp, coords):
    bins = pj.bin_index(coords, pj.grid_edges(coords, G))
    E, _ = pj.bin_energy(lp, bins, G * G)
    wells = [w for w in pj.persistence(E, pj.grid_adjacency((G, G))) if w["persistence"] >= 1.0]
    return E.reshape(G, G), wells


def plot_panel(ax, lp, coords, title, vmin, vmax, wells=True):
    E, ws = surface(lp, coords)
    em = np.where(np.isfinite(E), E, np.nan)
    im = ax.imshow(em, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    if wells:
        for w in ws:
            r, c = divmod(int(w["cell"]), G)
            ax.plot(c, r, marker="o", ms=8, mfc="none", mec="white", mew=1.1)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return im, np.nanmin(em), np.nanmax(em)


def six_panel(key, label):
    m = load_model(key)
    lp = m.log_prob().ravel() / np.log(2)
    z = np.load(f"data/derived/energy_metric_layout_{key}.npz")
    coords_em = z["coords"]
    z.close()
    surfs = {}
    for name in BASIS_NAMES:
        b = bs.compute_basis(name, model=m)
        surfs[name] = bs.coords_linear(b["U"], b["mean"], L4, A)
    surfs["energy_metric"] = coords_em
    fin = []
    for c in surfs.values():
        E, _ = surface(lp, c)
        fin.append(E[np.isfinite(E)])
    fin = np.concatenate(fin)
    vmin, vmax = float(np.percentile(fin, 2)), float(np.percentile(fin, 98))
    labels = [bs.BASES[n]["label"] for n in BASIS_NAMES] + ["Energy-metric layout"]
    fig, axes = plt.subplots(1, 6, figsize=(28, 5.2))
    for ax, (n, c), lbl in zip(axes, surfs.items(), labels):
        plot_panel(ax, lp, c, lbl, vmin, vmax)
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.92, 0.14, 0.012, 0.72])
    fig.colorbar(axes[0].images[0], cax=cax, label="E_bin (bits, uncapped)")
    fig.suptitle(f"{label} — raw E_bin (2-98% colour range; wells >= 1 bit circled)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.90, 0.95])
    fig.savefig(Path("figures") / f"basis_compare_{key}.png", dpi=100)
    plt.close(fig)
    print(f"wrote figures/basis_compare_{key}.png (6 panels)")


def controls_figure(key, label):
    m = load_model(key)
    lp = m.log_prob().ravel() / np.log(2)
    z = np.load(f"data/derived/energy_metric_layout_{key}.npz")
    panels = [("rank (lambda->0)", z["rank"]), ("Hamming (lambda->inf)", z["hamming"]),
              (f"energy-metric (lambda={z['lam']:.2f})", z["coords"])]
    z.close()
    if key == "pooled":
        s = np.load("data/derived/energy_metric_sweep_pooled.npz")
        panels += [(f"lambda={s['lam']*0.25:.2f}", s["lam025"]),
                   (f"lambda={s['lam']*4:.2f}", s["lam400"])]
        s.close()
    fin = []
    for _, c in panels:
        E, _ = surface(lp, c)
        fin.append(E[np.isfinite(E)])
    fin = np.concatenate(fin)
    vmin, vmax = float(np.percentile(fin, 2)), float(np.percentile(fin, 98))
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.2))
    for ax, (t, c) in zip(axes, panels):
        plot_panel(ax, lp, c, t, vmin, vmax)
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.92, 0.14, 0.012, 0.72])
    fig.colorbar(axes[0].images[0], cax=cax, label="E_bin (bits, uncapped)")
    fig.suptitle(f"{label} — energy-metric controls and lambda sweep (2-98% range)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.90, 0.95])
    fig.savefig(Path("figures") / f"energy_metric_controls_{key}.png", dpi=100)
    plt.close(fig)
    print(f"wrote figures/energy_metric_controls_{key}.png")


if __name__ == "__main__":
    for key, label in [("austen", "(a) Austen"), ("twitter2023", "(b) Twitter 2023"),
                       ("bluesky2024", "(c) Bluesky 2024"), ("pooled", "(d) Twitter pooled")]:
        six_panel(key, label)
        controls_figure(key, label)
