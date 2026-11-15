"""Short-form social-media corpora -> time-binned fixed-length word counts.

Sources (both openly downloadable as parquet):

  twitter  enryu43/twitter100m_tweets  (doi:10.5281/zenodo.15086029)
           user timelines, each truncated at the account's ~1,001 most recent
           tweets as of the scrape (early 2023). Years are therefore NOT equal
           samples of the platform: later years are dense, and earlier years are
           reachable only through low-volume accounts. ``per_year`` records the
           number of distinct users and the effective number of users per year so
           this composition can be checked rather than assumed away.
  bluesky  Roronotalt/bluesky-ten-million (posts from late November 2024)

The archive.org "Twitter Stream Grab" collection would be the unbiased 1% sample,
but every item in it is flagged access-restricted and returns 401/403.

Cleaning is identical for every source: HTML entities decoded; URLs, @mentions and
#hashtags removed; a leading retweet marker removed. Language is assigned by the
argmax of fastText lid.176 (Joulin et al. 2016) -- the Bayes decision rule, with no
tuned confidence threshold. Exact duplicate texts within a year are counted once,
so bot reposts and copy-paste spam do not count as independent utterances.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import corpus

DATA = corpus.DATA
DERIVED = DATA / "derived"
LID_PATH = DATA / "lid" / "lid.176.ftz"

_URL = re.compile(r"https?://\S+|www\.\S+")
# Bluesky handles and bare domains (name.bsky.social, example.com). The final label
# must be a listed suffix, so a missing space after a full stop ("sentence.Next")
# is not mistaken for a domain.
_TLDS = ("social|com|org|net|edu|gov|io|co|uk|us|ca|au|de|fr|jp|ly|me|tv|be|gl|gg|"
         "app|dev|xyz|info|biz|blog|news|online|site|art|ai")
_DOMAIN = re.compile(rf"\b[\w-]+(?:\.[\w-]+)*\.(?:{_TLDS})\b(?:/\S*)?", re.I)
_MENTION = re.compile(r"@[\w.-]+")
_HASHTAG = re.compile(r"#\w+")
_RT = re.compile(r"^\s*rt\b[:\s]*", re.I)
_WS = re.compile(r"\s+")

SOURCES = {
    "twitter": {"glob": "tw100m/shard*.parquet", "text": "tweet", "time": "date",
                "user": "user"},
    "bluesky": {"glob": "bsky/train-*.parquet", "text": "text", "time": "created_at",
                "user": "author_did"},
}


def clean(text: str) -> str:
    t = html.unescape(text)
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _DOMAIN.sub(" ", t)
    t = _HASHTAG.sub(" ", t)
    t = _RT.sub("", t)
    return _WS.sub(" ", t).strip()


_lid = None


def _model():
    global _lid
    if _lid is None:
        import fasttext
        fasttext.FastText.eprint = lambda *a, **k: None     # silence load warning
        _lid = fasttext.load_model(str(LID_PATH))
    return _lid


def is_english(texts: list[str]) -> np.ndarray:
    labels, _ = _model().predict([t.replace("\n", " ") for t in texts], k=1)
    return np.array([lab[0] == "__label__en" for lab in labels])


def _process_file(args) -> dict:
    """Per-file pass: per-period counts of words of the requested lengths.

    ``period`` is "year" or "month" (key "YYYY" or "YYYY-MM"). Exact duplicate texts
    are removed within each calendar year regardless of ``period``.
    """
    import pyarrow.parquet as pq

    path, spec, lengths, period = args
    cut = 4 if period == "year" else 7
    lengths = set(lengths)
    pf = pq.ParquetFile(path)
    out: dict[str, dict] = {}
    seen: dict[str, set] = defaultdict(set)
    for batch in pf.iter_batches(batch_size=50_000,
                                 columns=[spec["text"], spec["time"], spec["user"]]):
        texts = batch.column(0).to_pylist()
        times = batch.column(1).to_pylist()
        users = batch.column(2).to_pylist()
        cleaned = [clean(t or "") for t in texts]
        keep = [i for i, t in enumerate(cleaned) if t]
        if not keep:
            continue
        eng = is_english([cleaned[i] for i in keep])
        for flag, i in zip(eng, keep):
            stamp = str(times[i])
            key, year = stamp[:cut], stamp[:4]
            rec = out.setdefault(key, {"posts": 0, "english": 0, "dupes": 0,
                                       "counts": defaultdict(Counter),
                                       "user_tokens": Counter()})
            rec["posts"] += 1
            if not flag:
                continue
            rec["english"] += 1
            h = hashlib.blake2b(cleaned[i].lower().encode(), digest_size=8).digest()
            if h in seen[year]:
                rec["dupes"] += 1
                continue
            seen[year].add(h)
            n = 0
            for tok in corpus._TOKEN.findall(cleaned[i].lower().replace("_", " ")):
                if len(tok) in lengths and tok.isascii() and tok.isalpha():
                    rec["counts"][len(tok)][tok] += 1
                    n += 1
            if n:
                rec["user_tokens"][users[i]] += n
    return {"file": str(path), "periods": out}


def _summary(rec: dict) -> dict:
    ut = np.array(list(rec["user_tokens"].values()), dtype=float)
    share = ut / ut.sum() if ut.size else ut
    return {
        "posts": rec["posts"],
        "english_posts": rec["english"],
        "duplicate_english_posts": rec["dupes"],
        "users": int(ut.size),
        # inverse Simpson index of user token shares: the number of equally active
        # users that would give the same concentration
        "effective_users": float(1.0 / (share ** 2).sum()) if ut.size else 0.0,
        "max_user_share": float(share.max()) if ut.size else 0.0,
    }


def ingest(source: str, L: int = 4, workers: int = 8, *, lengths=None, period: str = "year",
           files=None, out_name: str | None = None) -> Path:
    """Process downloaded files of ``source`` and write per-period word counts.

    Default call (one length, yearly, all files) writes the JSON layout read by
    ``load_counts``. With ``lengths``, ``period`` or ``out_name`` it writes a gzipped
    pickle {"source", "lengths", "period", "files", "periods": {key: {stats...,
    "counts": {L: {word: n}}}}} read by ``load_periods``.

    Duplicate removal is per file (files hold disjoint users for the twitter source,
    and the bluesky source is a single file), which is exact for both layouts.
    """
    spec = SOURCES[source]
    files = sorted(DATA.glob(spec["glob"])) if files is None else [Path(f) for f in files]
    if not files:
        raise FileNotFoundError(f"no files for {source} under {DATA / spec['glob']}")
    lens = (L,) if lengths is None else tuple(lengths)
    legacy = lengths is None and period == "year" and out_name is None
    legacy_path = DERIVED / f"{source}_L{L}_by_year.json"
    if legacy and legacy_path.exists():
        # this file backs the frozen four-letter study; never replace it with counts
        # from a different set of input files
        recorded = json.loads(legacy_path.read_text())["files"]
        if recorded != [f.name for f in files]:
            raise FileExistsError(
                f"{legacy_path.name} was built from {len(recorded)} files and backs the "
                f"existing study; pass files= matching it, or out_name= for a new output")
    periods: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(files))) as ex:
        for res in ex.map(_process_file, [(f, spec, lens, period) for f in files]):
            for key, rec in res["periods"].items():
                agg = periods.setdefault(key, {"posts": 0, "english": 0, "dupes": 0,
                                               "counts": defaultdict(Counter),
                                               "user_tokens": Counter()})
                for k in ("posts", "english", "dupes"):
                    agg[k] += rec[k]
                for ln, c in rec["counts"].items():
                    agg["counts"][ln].update(c)
                agg["user_tokens"].update(rec["user_tokens"])
            print(f"  {Path(res['file']).name} done", flush=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    if legacy:
        summary = {}
        for y, rec in sorted(periods.items()):
            c = rec["counts"].get(L, Counter())
            summary[y] = {**_summary(rec), "tokens": int(sum(c.values())), "types": len(c),
                          "counts": dict(c)}
        path = legacy_path
        path.write_text(json.dumps({"source": source, "L": L,
                                    "files": [f.name for f in files], "years": summary}))
        return path
    import gzip
    import pickle
    data = {"source": source, "lengths": list(lens), "period": period,
            "files": [f.name for f in files],
            "periods": {k: {**_summary(rec),
                            "counts": {ln: dict(c) for ln, c in rec["counts"].items()}}
                        for k, rec in sorted(periods.items())}}
    path = DERIVED / (out_name or f"{source}_{period}_L{'-'.join(map(str, lens))}.pkl.gz")
    with gzip.open(path, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_counts(source: str, L: int = 4) -> dict:
    return json.loads((DERIVED / f"{source}_L{L}_by_year.json").read_text())


def load_periods(name: str) -> dict:
    import gzip
    import pickle
    with gzip.open(DERIVED / name, "rb") as fh:
        return pickle.load(fh)
