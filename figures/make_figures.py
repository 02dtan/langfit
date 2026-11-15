"""Figures for the pairwise maximum-entropy study of four-letter words.

Every number is read from data/derived/maxent_study.json (written by
scripts/maxent_study.py); this script only draws. Each figure answers one question:

  fig1  How much of the structure of four-letter words is pairwise, and does that
        differ between edited prose and short-form social media?
  fig2  Is the pairwise structure real, i.e. does it predict held-out text?
  fig3  Does the pairwise energy predict how often each word is used, and which
        words does it fail on?
  fig4  How does the landscape move over time: which new forms become wells?

Run:  python figures/make_figures.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plotsvg as svg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "data" / "derived" / "maxent_study.json"
OUT = Path(__file__).resolve().parent
N_STATES = 26 ** 4


def save(fig, name: str, key: str) -> None:
    p = fig.save(OUT / name)
    print(f"  {name:32s} {p.stat().st_size / 1024:5.1f} kB  {key}")


def mean_sd(rows, name, key):
    v = np.array([r[key] for r in rows if r["name"] == name], dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def captured(rows, name):
    v = np.array([(r["S1"] - r["S2"]) / (r["S1"] - r["S_full"])
                  for r in rows if r["name"] == name])
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


# --------------------------------------------------------------------------- #
# fig 1
# --------------------------------------------------------------------------- #

def fig1(res) -> None:
    rows = res["corpora"]
    N = res["meta"]["N"]
    sb = res["meta"]["sb_published"]
    tw_years = sorted(int(r["name"].split()[1]) for r in rows
                      if r["name"].startswith("Twitter ") and r["rep"] == 0)
    show = ["Austen", "Gutenberg"] + [f"Twitter {y}" for y in
                                      (tw_years[0], tw_years[len(tw_years) // 2], tw_years[-1])] \
        + ["Bluesky"]
    labels = ["S&B 2007, Austen (published)"] + [
        "Bluesky 2024" if s == "Bluesky" else s for s in show]

    fig = svg.Figure(920, 345)
    n = len(labels)
    a = fig.add(svg.Panel(178, 52, 372, 240, xlim=(0, 20), ylim=(n - 0.4, -0.6),
                          title="Entropy of four-letter words, bits",
                          xlabel="entropy (bits)", yticks=list(range(n)),
                          yticklabels=labels, grid=False))
    a.vline(4 * math.log2(26), color=svg.MUTED)
    a.label(4 * math.log2(26) - 0.2, -0.75, "S_rand = 4 log2 26", color=svg.MUTED,
            size=10, anchor="end")
    vals = [(sb["S_full"], sb["S_2"], sb["S_ind"], 0, 0, 0)]
    for s in show:
        f, fs = mean_sd(rows, s, "S_full")
        two, ts = mean_sd(rows, s, "S2")
        one, os_ = mean_sd(rows, s, "S1")
        vals.append((f, two, one, fs, ts, os_))
    for i, (f, two, one, *_) in enumerate(vals):
        a.line([f, one], [i, i], color=svg.BORDER, width=5, opacity=0.6)
        a.points([one], [i], color=svg.SERIES[0], r=4.2)
        a.points([two], [i], color=svg.SERIES[1], r=4.2)
        a.points([f], [i], color=svg.SERIES[2], r=4.2)
        frac = (one - two) / (one - f)
        a.label(19.9, i + 0.12, f"{100 * frac:.0f}%", color=svg.FG, size=11, anchor="end")
    a.note(0.0, -0.13, "independent S_1", color=svg.SERIES[0], size=10.5)
    a.note(0.27, -0.13, "pairwise S_2", color=svg.SERIES[1], size=10.5)
    a.note(0.50, -0.13, "observed S_full", color=svg.SERIES[2], size=10.5)
    a.note(0.80, -0.13, "% captured", color=svg.FG, size=10.5)

    years = tw_years
    caps = [captured(rows, f"Twitter {y}") for y in years]
    refs = {k: captured(rows, k) for k in ("Austen", "Gutenberg", "Bluesky")}
    lo = min([c - s for c, s in caps] + [v - s for v, s in refs.values()] + [sb["captured"]])
    hi = max([c + s for c, s in caps] + [v + s for v, s in refs.values()] + [sb["captured"]])
    pad = 0.08 * (hi - lo)
    b = fig.add(svg.Panel(616, 52, 290, 240, xlim=(years[0] - 0.7, years[-1] + 0.7),
                          ylim=(lo - pad, hi + pad),
                          title="Share of multi-information that is pairwise",
                          xlabel="Twitter year", xticks=years[::3],
                          xticklabels=[str(y) for y in years[::3]],
                          ylabel="(S_1 - S_2) / (S_1 - S_full)"))
    for (k, (v, s)), col in zip(refs.items(), (svg.SERIES[3], svg.SERIES[4], svg.SERIES[5])):
        b.area([years[0] - 0.7, years[-1] + 0.7], [v - s] * 2, [v + s] * 2, color=col,
               opacity=0.14)
        b.hline(v, color=col, dash="5 3")
        b.label(years[0] - 0.5, v + 0.15 * pad, "Bluesky 2024" if k == "Bluesky" else k,
                color=col, size=10)
    for y, (c, s) in zip(years, caps):
        b.errorbar(y, c, s, color=svg.ACCENT)
    b.line(years, [c for c, _ in caps], color=svg.ACCENT, width=1.6)
    b.points(years, [c for c, _ in caps], color=svg.ACCENT, r=2.8)
    save(fig, "fig1_entropy_hierarchy.svg",
         f"N = {N:,} tokens per corpus x {res['meta']['reps']} subsamples; "
         f"Austen {100 * refs['Austen'][0]:.1f}%, Bluesky {100 * refs['Bluesky'][0]:.1f}%, "
         f"Twitter {100 * min(c for c, _ in caps):.1f}-{100 * max(c for c, _ in caps):.1f}%")


# --------------------------------------------------------------------------- #
# fig 2
# --------------------------------------------------------------------------- #

def fig2(res) -> None:
    """In-sample pairwise information S_1 - S_2 vs the held-out gain H_1 - H_2, both at
    the same N. The ratio is the share of the pairwise structure that is real rather
    than fitted sampling noise."""
    cv = res["cv"]
    rows = res["corpora"]
    names = list(cv)
    disp = ["Bluesky 2024" if k == "Bluesky" else k for k in names]

    h1 = np.array([cv[k]["1"]["bits"] for k in names])
    h2 = np.array([cv[k]["2"]["bits"] for k in names])
    se1 = np.array([cv[k]["1"]["se"] for k in names])
    se2 = np.array([cv[k]["2"]["se"] for k in names])
    diff = [np.array(cv[k]["1"]["per_fold"]) - np.array(cv[k]["2"]["per_fold"]) for k in names]
    gain = np.array([d.mean() for d in diff])
    gain_se = np.array([d.std(ddof=1) / np.sqrt(d.size) for d in diff])
    ins = np.array([mean_sd(rows, k, "S1")[0] - mean_sd(rows, k, "S2")[0] for k in names])

    fig = svg.Figure(920, 318)
    top = float(h1.max() + se1.max()) * 1.12
    a = fig.add(svg.Panel(64, 34, 400, 220, xlim=(-0.6, len(names) - 0.4), ylim=(0, top),
                          title="Held-out code length per word token (nested CV)",
                          xticks=list(range(len(names))), xticklabels=disp,
                          ylabel="bits per token", grid=False))
    for i in range(len(names)):
        a.bar(i - 0.38, i - 0.02, 0, h1[i], color=svg.SERIES[0], opacity=0.75)
        a.bar(i + 0.02, i + 0.38, 0, h2[i], color=svg.SERIES[1], opacity=0.9)
        a.errorbar(i - 0.2, h1[i], se1[i], color=svg.FG)
        a.errorbar(i + 0.2, h2[i], se2[i], color=svg.FG)
    a.note(0.02, 0.06, "independent model", color=svg.SERIES[0], size=10.5)
    a.note(0.36, 0.06, "pairwise model", color=svg.SERIES[1], size=10.5)

    hi = float(max(ins.max(), (gain + gain_se).max())) * 1.1
    b = fig.add(svg.Panel(540, 34, 366, 220, xlim=(0, hi), ylim=(0, hi),
                          title="Pairwise information: in-sample vs held-out",
                          xlabel="in-sample S_1 - S_2 (bits)",
                          ylabel="held-out H_1 - H_2 (bits/token)"))
    b.line([0, hi], [0, hi], color=svg.MUTED, dash="5 3")
    for j, k in enumerate(names):
        b.errorbar(ins[j], gain[j], gain_se[j], color=svg.SERIES[j % 6])
        b.points([ins[j]], [gain[j]], color=svg.SERIES[j % 6], r=3.6)
        b.note(0.04, 0.08 + 0.075 * j, f"{disp[j]}: {100 * gain[j] / ins[j]:.0f}% generalises",
               color=svg.SERIES[j % 6], size=10.5)
    b.note(0.55, 0.95, "dashed: all of it generalises", color=svg.MUTED, size=10)
    save(fig, "fig2_generalisation.svg",
         "; ".join(f"{k}: in-sample {i_:.2f}, held-out {g:.2f} +- {s:.2f}"
                   for k, i_, g, s in zip(names, ins, gain, gain_se)))


# --------------------------------------------------------------------------- #
# fig 3
# --------------------------------------------------------------------------- #

def _binned(x, y, n_bins):
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    cx, cy = [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            cx.append(x[m].mean())
            cy.append(y[m].mean())
    return np.array(cx), np.array(cy)


def fig3(res) -> None:
    first = {r["name"]: r for r in res["corpora"] if r["rep"] == 0}
    last_year = max(int(k.split()[1]) for k in first if k.startswith("Twitter "))
    names = ["Austen", f"Twitter {last_year}", "Bluesky"]
    fig = svg.Figure(920, 318)
    rng = np.random.default_rng(0)
    keys = []
    pw = 250
    for j, k in enumerate(names):
        sc = first[k]["scatter"]
        x = np.log10(np.array(sc["p_emp"]))
        y2 = np.log10(np.array(sc["p2"]))
        y1 = np.log10(np.array(sc["p1"]))
        # x spans the data: from one token in N to 1. y is clipped at the 1st percentile
        # of the pairwise probabilities; points below it are drawn hollow on the floor.
        xlo = math.floor(x.min() * 2) / 2
        ylo = math.floor(np.percentile(y2, 1))
        p = fig.add(svg.Panel(64 + j * (pw + 50), 34, pw, 230, xlim=(xlo, 0), ylim=(ylo, 0),
                              title=f"{'Bluesky 2024' if k == 'Bluesky' else k}  "
                                    f"({len(x):,} word types)",
                              xlabel="log10 observed frequency",
                              ylabel="log10 pairwise-model probability" if j == 0 else ""))
        p.line([max(xlo, ylo), 0], [max(xlo, ylo), 0], color=svg.MUTED, dash="5 3")
        pick = rng.choice(x.size, min(x.size, 600), replace=False)
        inside = y2[pick] >= ylo
        p.points(x[pick][inside], y2[pick][inside], color=svg.BORDER, r=1.4)
        p.points(x[pick][~inside], np.full((~inside).sum(), ylo), color=svg.BORDER, r=1.6,
                 hollow=True)
        # 20 equally spaced bins in log observed frequency, as in S&B Fig. 2
        bx, b2 = _binned(x, y2, 20)
        _, b1 = _binned(x, y1, 20)
        p.line(bx, np.clip(b1, ylo, 0), color=svg.SERIES[0], width=1.6)
        p.line(bx, np.clip(b2, ylo, 0), color=svg.SERIES[1], width=2.0)
        p.points(bx, np.clip(b2, ylo, 0), color=svg.SERIES[1], r=2.6)
        r2 = float(np.corrcoef(x, y2)[0, 1] ** 2)
        r1 = float(np.corrcoef(x, y1)[0, 1] ** 2)
        p.note(0.30, 0.78, f"pairwise: binned mean, r^2 = {r2:.2f}", color=svg.SERIES[1],
               size=10.5)
        p.note(0.30, 0.86, f"independent: binned mean, r^2 = {r1:.2f}", color=svg.SERIES[0],
               size=10.5)
        p.note(0.30, 0.94, "hollow: below the axis floor", color=svg.MUTED, size=10)
        keys.append(f"{k} r2 {r1:.2f}->{r2:.2f}")
    save(fig, "fig3_energy_vs_frequency.svg", "; ".join(keys))


# --------------------------------------------------------------------------- #
# fig 4
# --------------------------------------------------------------------------- #

def fig4(res) -> None:
    rows_all = res["corpora"]
    years = sorted({r["name"].split()[1] for r in rows_all if r["name"].startswith("Twitter ")},
                   key=int)
    groups = res["meta"]["probes"]
    rows = [(g, w) for g in ("slang", "control", "typo") for w in groups[g]]
    cols = years + ["Bsky 24"]
    col_names = [f"Twitter {y}" for y in years] + ["Bluesky"]
    N = res["meta"]["N"]

    nr, nc = len(rows), len(cols)
    cell_w, cell_h = 18.0, 11.5
    gap = 6.0
    yrow = []
    y = 0.0
    prev = None
    for g, _ in rows:
        if prev is not None and g != prev:
            y += gap / cell_h
        yrow.append(y)
        y += 1
        prev = g
    height = y
    fig = svg.Figure(920, 70 + height * cell_h + 50)

    def panel(x0, title):
        return fig.add(svg.Panel(x0, 40, nc * cell_w, height * cell_h, xlim=(0, nc),
                                 ylim=(height, 0), title=title, grid=False,
                                 xticks=[i + 0.5 for i in range(0, nc, 2)],
                                 xticklabels=[cols[i][-2:] if i < nc - 1 else "Bs"
                                              for i in range(0, nc, 2)],
                                 yticks=[r + 0.5 for r in yrow],
                                 yticklabels=[w for _, w in rows]))

    left = panel(110, "Usage: log10 tokens per million")
    right = panel(110 + nc * cell_w + 120, "Pairwise landscape: energy rank of the form")

    # each cell summarises the independent subsamples of that column: mean usage,
    # median energy rank, and the fraction of subsamples in which the form is a well
    freq = np.full((nr, nc), np.nan)
    rank = np.zeros((nr, nc))
    is_min = np.zeros((nr, nc))
    for j, cn in enumerate(col_names):
        reps = [r for r in rows_all if r["name"] == cn]
        for i, (_, w) in enumerate(rows):
            cnt = np.mean([r["probes"][w]["count"] for r in reps])
            if cnt > 0:
                freq[i, j] = math.log10(cnt / N * 1e6)
            rank[i, j] = np.median([r["probes"][w]["rank"] for r in reps])
            is_min[i, j] = np.mean([r["probes"][w]["is_min"] for r in reps])
    fmin, fmax = np.nanmin(freq), np.nanmax(freq)
    lmax = math.log10(N_STATES)
    for i in range(nr):
        for j in range(nc):
            if np.isfinite(freq[i, j]):
                op = 0.08 + 0.92 * (freq[i, j] - fmin) / (fmax - fmin)
                left.bar(j + 0.04, j + 0.96, yrow[i] + 0.06, yrow[i] + 0.94,
                         color=svg.SERIES[1], opacity=op)
            depth = 1.0 - math.log10(rank[i, j]) / lmax       # 1 = global minimum
            right.bar(j + 0.04, j + 0.96, yrow[i] + 0.06, yrow[i] + 0.94,
                      color=svg.ACCENT, opacity=max(0.04, depth))
            if is_min[i, j] > 0:
                right.points([j + 0.5], [yrow[i] + 0.5], color=svg.FG, r=2.6 * is_min[i, j])
    left.note(0.0, 1.0 + 26 / (height * cell_h),
              f"faint = {10 ** fmin:.2g}/M, solid = {10 ** fmax:.3g}/M; blank = unused",
              color=svg.MUTED, size=10)
    right.note(0.0, 1.0 + 26 / (height * cell_h),
               "solid = rank 1 of 456,976 strings; dot = a well (size: share of subsamples)",
               color=svg.MUTED, size=10)
    gi = {g: [i for i, (gg, _) in enumerate(rows) if gg == g] for g in groups}
    for g, idx in gi.items():
        right.label(nc + 0.4, yrow[idx[0]] + 0.8, g, color=svg.MUTED, size=10.5)
    became = [w for i, (g, w) in enumerate(rows) if g == "slang" and (is_min[i] >= 0.5).any()]
    save(fig, "fig4_landscape_by_year.svg",
         f"{len(years)} Twitter years + Bluesky; slang forms that are a well in some year: "
         f"{became or 'none'}")


def main() -> None:
    res = json.loads(STUDY.read_text())
    fig1(res)
    fig2(res)
    fig3(res)
    fig4(res)


if __name__ == "__main__":
    main()
