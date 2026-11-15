"""Dependency-free SVG plotting for the validation figures.

Self-contained output: no scripts, no external references, no background. Colours
come from the surrounding document's theme variables, so the same file renders
correctly in a light or dark theme. Nothing here computes anything; it only draws.
"""

from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import escape

FG = "var(--fg)"
MUTED = "var(--muted)"
BORDER = "var(--border)"
ACCENT = "var(--accent)"
OK = "var(--success)"
WARN = "var(--warning)"
ERR = "var(--error)"
SERIES = ("var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)")
FONT = "ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif"


def fmt(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e4 or a < 1e-2:
        return f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e")
    if a >= 1e3:
        return f"{v / 1e3:g}k"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.3f}".rstrip("0").rstrip(".")


def nice_ticks(lo: float, hi: float, n: int = 5):
    if not hi > lo:
        return [lo]
    raw = (hi - lo) / max(n, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    step = mag
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= mult * mag + 1e-30:
            step = mult * mag
            break
    out, v = [], math.ceil(lo / step - 1e-9) * step
    while v <= hi + step * 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def _ok(v, log: bool) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(f):
        return False
    return f > 0 if log else True


class Panel:
    """One axes box. Plot area is (x, y, w, h) in the parent figure's coordinates."""

    def __init__(self, x, y, w, h, xlim, ylim, xlog=False, ylog=False, title="",
                 xlabel="", ylabel="", xticks=None, yticks=None, grid=True,
                 y_ticks_side="left", xticklabels=None, yticklabels=None):
        self.xticklabels = list(xticklabels) if xticklabels is not None else None
        self.yticklabels = list(yticklabels) if yticklabels is not None else None
        self.px, self.py, self.w, self.h = float(x), float(y), float(w), float(h)
        self.xlim, self.ylim = tuple(xlim), tuple(ylim)
        self.xlog, self.ylog = xlog, ylog
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.grid = grid
        self.y_ticks_side = y_ticks_side
        self.xticks = list(xticks) if xticks is not None else self._auto(self.xlim, xlog)
        self.yticks = list(yticks) if yticks is not None else self._auto(self.ylim, ylog)
        self.body: list[str] = []
        self._extent = [math.inf, math.inf, -math.inf, -math.inf]

    def _track(self, px: float, py: float) -> None:
        e = self._extent
        e[0] = min(e[0], px)
        e[1] = min(e[1], py)
        e[2] = max(e[2], px)
        e[3] = max(e[3], py)

    def escapes(self, tol: float = 1.5) -> bool:
        """True if any primitive was drawn outside this panel's box.

        A wrong axis limit shows up here rather than as a silently misplaced curve:
        the data would be drawn over a neighbouring panel.
        """
        if self._extent[0] == math.inf:
            return False
        x0, y0 = self.px, self.py + self.h
        return (self._extent[0] < x0 - tol or self._extent[1] < self.py - tol
                or self._extent[2] > x0 + self.w + tol or self._extent[3] > y0 + tol)

    @staticmethod
    def _label(t, ticks, labels) -> str:
        if labels is None:
            return fmt(t)
        return str(labels[ticks.index(t)])

    @staticmethod
    def _auto(lim, log):
        if log:
            return [10.0 ** e for e in range(math.ceil(math.log10(lim[0])),
                                             math.floor(math.log10(lim[1])) + 1)]
        return nice_ticks(lim[0], lim[1])

    # -- coordinate transforms ------------------------------------------------
    def tx(self, v: float) -> float:
        lo, hi = self.xlim
        f = ((math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
             if self.xlog else (v - lo) / (hi - lo))
        return self.px + f * self.w

    def ty(self, v: float) -> float:
        lo, hi = self.ylim
        f = ((math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
             if self.ylog else (v - lo) / (hi - lo))
        return self.py + self.h - f * self.h

    # -- drawing -------------------------------------------------------------
    def line(self, xs, ys, color=ACCENT, width=1.7, dash=None, opacity=None):
        kept = [(self.tx(x), self.ty(y)) for x, y in zip(xs, ys)
                if _ok(x, self.xlog) and _ok(y, self.ylog)]
        if not kept:
            return
        for px, py in kept:
            self._track(px, py)
        pts = " ".join(f"{px:.1f},{py:.1f}" for px, py in kept)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        if opacity is not None:
            extra += f' opacity="{opacity}"'
        self.body.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                         f'stroke-width="{width}"{extra}/>')

    def points(self, xs, ys, color=ACCENT, r=2.5, hollow=False):
        for x, y in zip(xs, ys):
            if not (_ok(x, self.xlog) and _ok(y, self.ylog)):
                continue
            X, Y = self.tx(x), self.ty(y)
            self._track(X, Y)
            fill = "none" if hollow else color
            self.body.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" '
                             f'r="{r}" fill="{fill}" stroke="{color}"/>')

    def errorbar(self, x, y, err, color=ACCENT, width=1.2):
        if not (_ok(x, self.xlog) and _ok(y, self.ylog)):
            return
        X = self.tx(x)
        y0, y1 = self.ty(max(y - err, self.ylim[0])), self.ty(min(y + err, self.ylim[1]))
        self._track(X, min(y0, y1))
        self._track(X, max(y0, y1))
        self.body.append(f'<line x1="{X:.1f}" y1="{y0:.1f}" x2="{X:.1f}" y2="{y1:.1f}" '
                         f'stroke="{color}" stroke-width="{width}" opacity="0.7"/>')

    def hline(self, y, color=MUTED, dash="5 3", width=1.1, opacity=None):
        if not _ok(y, self.ylog):
            return
        Y = self.ty(y)
        self._track(self.px, Y)
        self._track(self.px + self.w, Y)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        if opacity is not None:
            extra += f' opacity="{opacity}"'
        self.body.append(f'<line x1="{self.px:.1f}" y1="{Y:.1f}" x2="{self.px + self.w:.1f}" '
                         f'y2="{Y:.1f}" stroke="{color}" stroke-width="{width}"{extra}/>')

    def vline(self, x, color=MUTED, dash="5 3", width=1.1):
        if not _ok(x, self.xlog):
            return
        X = self.tx(x)
        self._track(X, self.py)
        self._track(X, self.py + self.h)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.body.append(f'<line x1="{X:.1f}" y1="{self.py:.1f}" x2="{X:.1f}" '
                         f'y2="{self.py + self.h:.1f}" stroke="{color}" '
                         f'stroke-width="{width}"{extra}/>')

    def band(self, x0, x1, color=MUTED, opacity=0.10):
        a, b = self.tx(x0), self.tx(x1)
        self._track(min(a, b), self.py)
        self._track(max(a, b), self.py + self.h)
        self.body.append(f'<rect x="{min(a, b):.1f}" y="{self.py:.1f}" '
                         f'width="{abs(b - a):.1f}" height="{self.h:.1f}" '
                         f'fill="{color}" opacity="{opacity}"/>')

    def area(self, xs, ys_lo, ys_hi, color=MUTED, opacity=0.12):
        top = [(self.tx(x), self.ty(y)) for x, y in zip(xs, ys_hi)
               if _ok(x, self.xlog) and _ok(y, self.ylog)]
        bot = [(self.tx(x), self.ty(y)) for x, y in zip(xs, ys_lo)
               if _ok(x, self.xlog) and _ok(y, self.ylog)]
        if len(top) < 2 or len(bot) < 2:
            return
        for px, py in top + bot:
            self._track(px, py)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in top + bot[::-1])
        self.body.append(f'<polygon points="{pts}" fill="{color}" opacity="{opacity}" '
                         f'stroke="none"/>')

    def bar(self, x0, x1, y0, y1, color=ACCENT, opacity=0.85):
        a, b = self.tx(x0), self.tx(x1)
        ya, yb = self.ty(y0), self.ty(y1)
        self._track(min(a, b), min(ya, yb))
        self._track(max(a, b), max(ya, yb))
        self.body.append(f'<rect x="{min(a, b):.1f}" y="{min(ya, yb):.1f}" '
                         f'width="{abs(b - a):.1f}" height="{abs(yb - ya):.1f}" '
                         f'fill="{color}" opacity="{opacity}"/>')

    def label(self, x, y, s, color=FG, size=11.5, anchor="start"):
        if not (_ok(x, self.xlog) and _ok(y, self.ylog)):
            return
        self.body.append(f'<text x="{self.tx(x):.1f}" y="{self.ty(y):.1f}" fill="{color}" '
                         f'font-size="{size}" text-anchor="{anchor}">{escape(s)}</text>')

    def note(self, fx, fy, s, color=FG, size=11.5, anchor="start"):
        """Text at a fraction of the panel box: (0,0) top-left, (1,1) bottom-right."""
        x = self.px + fx * self.w
        y = self.py + fy * self.h
        self.body.append(f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-size="{size}" '
                         f'text-anchor="{anchor}">{escape(s)}</text>')

    # -- assembly ------------------------------------------------------------
    def render(self) -> str:
        o: list[str] = []
        x0, y0 = self.px, self.py + self.h
        x1, y1 = self.px + self.w, self.py
        if self.grid:
            for t in self.xticks:
                if _ok(t, self.xlog) and x0 <= self.tx(t) <= x1:
                    o.append(f'<line x1="{self.tx(t):.1f}" y1="{y1:.1f}" x2="{self.tx(t):.1f}" '
                             f'y2="{y0:.1f}" stroke="{BORDER}" opacity="0.45"/>')
            for t in self.yticks:
                if _ok(t, self.ylog) and y1 <= self.ty(t) <= y0:
                    o.append(f'<line x1="{x0:.1f}" y1="{self.ty(t):.1f}" x2="{x1:.1f}" '
                             f'y2="{self.ty(t):.1f}" stroke="{BORDER}" opacity="0.45"/>')
        o.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y0:.1f}" stroke="{BORDER}"/>')
        o.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y1:.1f}" stroke="{BORDER}"/>')
        for t in self.xticks:
            if not _ok(t, self.xlog):
                continue
            X = self.tx(t)
            if not (x0 - 0.6 <= X <= x1 + 0.6):
                continue
            o.append(f'<line x1="{X:.1f}" y1="{y0:.1f}" x2="{X:.1f}" y2="{y0 + 3:.1f}" stroke="{BORDER}"/>')
            o.append(f'<text x="{X:.1f}" y="{y0 + 15:.1f}" fill="{MUTED}" font-size="10.5" '
                     f'text-anchor="middle">{escape(self._label(t, self.xticks, self.xticklabels))}</text>')
        for t in self.yticks:
            if not _ok(t, self.ylog):
                continue
            Y = self.ty(t)
            if not (y1 - 0.6 <= Y <= y0 + 0.6):
                continue
            o.append(f'<line x1="{x0 - 3:.1f}" y1="{Y:.1f}" x2="{x0:.1f}" y2="{Y:.1f}" stroke="{BORDER}"/>')
            anchor = "end" if self.y_ticks_side == "left" else "start"
            tx = x0 - 6 if self.y_ticks_side == "left" else x1 + 6
            o.append(f'<text x="{tx:.1f}" y="{Y + 3.5:.1f}" fill="{MUTED}" font-size="10.5" '
                     f'text-anchor="{anchor}">{escape(self._label(t, self.yticks, self.yticklabels))}</text>')
        if self.title:
            o.append(f'<text x="{x0:.1f}" y="{y1 - 9:.1f}" fill="{FG}" font-size="12.5" '
                     f'font-weight="600">{escape(self.title)}</text>')
        if self.xlabel:
            o.append(f'<text x="{x0 + self.w / 2:.1f}" y="{y0 + 32:.1f}" fill="{MUTED}" '
                     f'font-size="11.5" text-anchor="middle">{escape(self.xlabel)}</text>')
        if self.ylabel:
            cx, cy = x0 - 40, y0 - self.h / 2
            o.append(f'<text x="{cx:.1f}" y="{cy:.1f}" fill="{MUTED}" font-size="11.5" '
                     f'text-anchor="middle" transform="rotate(-90 {cx:.1f} {cy:.1f})">'
                     f'{escape(self.ylabel)}</text>')
        return "\n".join(o + self.body)


class Figure:
    def __init__(self, w: float, h: float):
        self.w, self.h = float(w), float(h)
        self.panels: list[Panel] = []

    def add(self, panel: Panel) -> Panel:
        self.panels.append(panel)
        return panel

    def svg(self) -> str:
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w:g} {self.h:g}" '
                f'width="{self.w:g}" height="{self.h:g}" font-family="{FONT}" font-size="12">')
        return "\n".join([head] + [p.render() for p in self.panels] + ["</svg>"])

    def escapes(self) -> list[str]:
        return [p.title or "<untitled>" for p in self.panels if p.escapes()]

    def save(self, path) -> Path:
        bad = self.escapes()
        if bad:
            print(f"  WARNING {Path(path).name}: primitives outside panel box: {bad}")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.svg() + "\n")
        return p


def layout(ncols: int, nrows: int, w=920, panel_h=200, ml=52, mr=14, mt=16, mb=44,
           gapx=46, gapy=64, y_ticks_side="left"):
    """Even grid of panels inside a figure of width ``w``. Returns (figure, rects)."""
    pw = (w - ml - mr - gapx * (ncols - 1)) / ncols
    h = mt + nrows * panel_h + gapy * (nrows - 1) + mb
    fig = Figure(w, h)
    rects = []
    for r in range(nrows):
        for c in range(ncols):
            rects.append((ml + c * (pw + gapx), mt + r * (panel_h + gapy), pw, panel_h))
    return fig, rects
