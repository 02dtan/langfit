"""Command-line wrapper for langfit.social.ingest (multiprocessing needs a real module).

Examples
  # month-binned four-letter counts from the 16 shards behind the existing study
  python scripts/ingest_social.py twitter --lengths 4 --period month \
      --shards 0-15 --out twitter_month_L4_existing16.pkl.gz
  # yearly counts of every word length 1..20 from all downloaded shards
  python scripts/ingest_social.py twitter --lengths 1-20 --period year \
      --out twitter_year_L1-20_all.pkl.gz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langfit import social  # noqa: E402


def span(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=sorted(social.SOURCES))
    ap.add_argument("--lengths", default="4")
    ap.add_argument("--period", choices=("year", "month"), default="year")
    ap.add_argument("--shards", default=None, help="twitter shard indices, e.g. 0-15")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    files = None
    if a.shards is not None:
        files = [social.DATA / f"tw100m/shard{i:02d}.parquet" for i in span(a.shards)]
        missing = [f for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"missing shards: {[f.name for f in missing]}")
    t = time.time()
    p = social.ingest(a.source, workers=a.workers, lengths=span(a.lengths), period=a.period,
                      files=files, out_name=a.out)
    print(f"wrote {p} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
