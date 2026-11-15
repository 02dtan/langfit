"""Completion concentration of the triplet cells highlighted in ternary.md §4.4.

For each letter-triple cell c (e.g. 'm_et') on pooled Twitter (all 41 shards): the
word with the most tokens among the cell's 26 completions, that word's share of the
cell's empirical mass, and the word-level ratio P_emp(word) / P_2(word) under the
exact pairwise model of the full corpus. Separates the marginal ratio mu_c / P_2|c
(stored in ternary_study.json) from a statement about a single word.
Output: data/derived/ternary_cells.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langfit import corpus, potts, social  # noqa: E402

L = 4
OUT = corpus.DATA / "derived" / "ternary_cells.json"
TWITTER = "twitter_year_L1-20_all.pkl.gz"
# cells quoted in ternary.md §4.4 (top contributors to D_c in detail.Twitter.full)
CELLS = ["las_", "tim_", "gam_", "sur_", "wa_t", "la_t", "ci_y", "we_e", "m_et", "d_es",
         "w_nt", "p_rt", "_oes", "_eam", "_eet"]


def main() -> None:
    periods = social.load_periods(TWITTER)["periods"]
    pooled = Counter()
    for rec in periods.values():
        pooled.update(rec["counts"].get(L, {}))
    del periods
    st, ct = potts.counts_to_arrays(pooled, L)
    N = ct.sum()
    m2, diag = potts.fit(st, ct, L, 2, "exact")
    P2 = m2.prob().ravel()
    words = ["".join(chr(97 + x) for x in s) for s in st]
    count = dict(zip(words, ct))
    out = {"N": float(N), "converged": diag["converged"], "cells": {}}
    for cell in CELLS:
        match = [w for w in words if all(a == "_" or a == b for a, b in zip(cell, w))]
        mass = sum(count[w] for w in match)
        top = max(match, key=lambda w: count[w])
        out["cells"][cell] = {
            "top_word": top, "top_share": float(count[top] / mass),
            "word_ratio_emp_over_p2": float(count[top] / N / P2[potts.ravel(top)])}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
