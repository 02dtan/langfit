"""Corpus loading and tokenisation.

Real text only; no synthetic filler. Gutenberg texts are cached under data/.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

GUTENBERG = {
    "pride": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "moby": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    "sherlock": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    "frankenstein": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    "dracula": "https://www.gutenberg.org/cache/epub/345/pg345.txt",
    "twocities": "https://www.gutenberg.org/cache/epub/98/pg98.txt",
    "expectations": "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
    "warandpeace": "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
    "grimm": "https://www.gutenberg.org/cache/epub/2591/pg2591.txt",
    "tomsawyer": "https://www.gutenberg.org/cache/epub/74/pg74.txt",
    "huckfinn": "https://www.gutenberg.org/cache/epub/76/pg76.txt",
    "dorian": "https://www.gutenberg.org/cache/epub/174/pg174.txt",
    "janeeyre": "https://www.gutenberg.org/cache/epub/1260/pg1260.txt",
    "emma": "https://www.gutenberg.org/cache/epub/158/pg158.txt",
    "wuthering": "https://www.gutenberg.org/cache/epub/768/pg768.txt",
}

DEFAULT_CORPUS = tuple(GUTENBERG)

# The eight Austen texts Stephens & Bialek used (arXiv:0801.0253, ref. 16).
AUSTEN = {
    "emma": "https://www.gutenberg.org/cache/epub/158/pg158.txt",
    "ladysusan": "https://www.gutenberg.org/cache/epub/946/pg946.txt",
    "loveandfreindship": "https://www.gutenberg.org/cache/epub/1212/pg1212.txt",
    "mansfield": "https://www.gutenberg.org/cache/epub/141/pg141.txt",
    "northanger": "https://www.gutenberg.org/cache/epub/121/pg121.txt",
    "persuasion": "https://www.gutenberg.org/cache/epub/105/pg105.txt",
    "pride": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "sense": "https://www.gutenberg.org/cache/epub/161/pg161.txt",
}
GUTENBERG.update({k: v for k, v in AUSTEN.items() if k not in GUTENBERG})

# One tokenizer for every corpus, so word counts are comparable across sources.
# A token is a maximal run of word characters and apostrophes; only runs that are
# exactly L ASCII letters count. "don't" and "café" are therefore excluded, while
# "dont" -- a real orthographic variant on social media -- is kept.
_TOKEN = re.compile(r"[\w']+")

_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.S)
_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.S)


def fetch(name: str, path: Path | None = None) -> Path:
    """Download a Gutenberg text once and cache it."""
    url = GUTENBERG[name]
    path = path or (DATA / f"{name}.txt")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    return path


def strip_boilerplate(text: str) -> str:
    """Remove the Project Gutenberg header/footer."""
    m = _START.search(text)
    if m:
        text = text[m.end():]
    m = _END.search(text)
    if m:
        text = text[: m.start()]
    return text


def load_text(name: str) -> str:
    return strip_boilerplate(fetch(name).read_text(encoding="utf-8", errors="replace"))


def load_corpus(names=None) -> str:
    names = DEFAULT_CORPUS if names is None else names
    out = []
    for n in names:
        try:
            out.append(load_text(n))
        except Exception as e:                      # keep going: one bad mirror is not fatal
            print(f"  [skip {n}: {e}]")
    return "\n".join(out)


def fixed_length_words(text: str, L: int = 4, counter=None):
    """Count tokens that are exactly L ASCII letters (lowercased).

    Underscores are treated as separators (Gutenberg marks italics as _word_).
    """
    from collections import Counter

    c = Counter() if counter is None else counter
    for tok in _TOKEN.findall(text.lower().replace("_", " ")):
        if len(tok) == L and tok.isascii() and tok.isalpha():
            c[tok] += 1
    return c


def word_list(path: Path | None = None) -> list[str]:
    """The system dictionary: a real lexicon used as the support of the
    word-level landscape."""
    p = path or Path("/usr/share/dict/words")
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        w = line.strip().lower()
        if w.isalpha():
            out.append(w)
    return out
