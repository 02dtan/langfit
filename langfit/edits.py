"""Edit-move kinetics on the word energy landscape across lengths.

Stephens & Bialek (S&B; arXiv:0801.0253, Phys. Rev. E 81, 066119, 2010) define the
landscape of four-letter words by single-letter substitution: a word is a well if every
substitution raises its energy, and a misspelling is corrected by relaxing downhill.
This module extends the neighbourhood to the four elementary edits of Damerau (1964)
and Levenshtein (1966), each read as an elementary move of a Markov chain on
configuration space:

  sub    single-letter substitution         single-spin flip (Glauber 1963)
  trans  adjacent transposition             nearest-neighbour exchange that conserves
                                            composition (Kawasaki 1966)
  del    delete one letter, L -> L-1        particle removal  } grand-canonical moves
  ins    insert one letter, L -> L+1        particle insertion} (Norman & Filinov 1969,
                                                                Adams 1974)

Cross-length energy. Words of different lengths live in one grand-canonical
configuration space. With one length-L block per modelled length, the maximum-entropy
distribution constrained by the pairwise letter marginals of each block and by the
length distribution P(L) (Jaynes 1957) is

    P(w) = P(L) exp(-H_L(w) ln 2) / Z_L,  H_L(w) = -sum_{i<j} V^L_ij(w_i, w_j) / ln 2,

so in bits

    E(w) = -log2 P(w) = H_L(w) + log2 Z_L - log2 P(L) = H_L(w) - mu_L,
    mu_L = log2 P(L) - log2 Z_L,

the grand-canonical weight exp(beta mu_L - beta H_L) with one chemical potential per
length class (beta = ln 2 per bit). The usual mu N is the special case of a linear
mu_L, i.e. of constraining only the mean length. Within a block E differs from S&B's
energy by a constant, so substitution-only landscapes are unchanged.

Moves change kinetics, never the equilibrium: any move set that connects the space,
used as a Metropolis-Hastings proposal (Metropolis et al. 1953; Hastings 1970), has P
as its stationary law. What a move set changes is the graph on which local minima,
descent paths and basins are defined (landscape = energy on a graph: Stadler 2002;
Reidys & Stadler 2002; Wales 2003).

Graph conventions (all functions here):
  * the space is the union of the A^L grids for a run of consecutive lengths; global
    index = offset(L) + row-major index within the grid (as ``maxent.ravel``);
  * neighbours outside the modelled lengths do not exist (boundary: insertions from
    the longest and deletions from the shortest modelled length are absent; callers
    must qualify statements about boundary lengths accordingly);
  * a strict local minimum has finite energy below every modelled neighbour;
  * steepest descent moves to the lowest-energy neighbour if strictly lower; exact
    ties go to the first in the order sub < trans < del < ins, then ascending
    position, then ascending letter. With ``moves=("sub",)`` this reproduces
    ``maxent.descend`` exactly.

Error naming: a corruption type names what the channel did to the intended word; a
letter *inserted* by the typist is undone by a *deletion* move, and vice versa.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterator

import numpy as np

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
A = len(ALPHABET)
MOVES = ("sub", "trans", "del", "ins")        # order = tie-break order of descent
ERRORS = ("sub", "trans", "ins", "del")       # corruption types (what the typist did)
# A corruption of type t is undone by the move of type INVERSE[t].
INVERSE = {"sub": "sub", "trans": "trans", "ins": "del", "del": "ins"}


# --------------------------------------------------------------------------- #
# configuration space
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Space:
    """Union of the A^L word grids for consecutive lengths ``lengths``."""

    lengths: tuple[int, ...]

    def __post_init__(self):
        ls = tuple(self.lengths)
        if not ls or ls[0] < 1 or list(ls) != list(range(ls[0], ls[0] + len(ls))):
            raise ValueError(f"lengths must be consecutive positive integers: {ls}")
        object.__setattr__(self, "lengths", ls)

    def offset(self, L: int) -> int:
        return sum(A ** k for k in self.lengths if k < L)

    @property
    def size(self) -> int:
        return sum(A ** k for k in self.lengths)

    def block(self, x: np.ndarray, L: int) -> np.ndarray:
        """The length-L slice of a global array, as a (A,)*L view."""
        o = self.offset(L)
        return x[o:o + A ** L].reshape((A,) * L)

    def index(self, word: str) -> int:
        L = len(word)
        if L not in self.lengths:
            raise KeyError(word)
        f = 0
        for ch in word:
            f = f * A + (ord(ch) - 97)
        return self.offset(L) + f

    def length_of(self, idx) -> np.ndarray:
        idx = np.asarray(idx)
        out = np.zeros(idx.shape, dtype=np.int64)
        for L in self.lengths:
            o = self.offset(L)
            out[(idx >= o) & (idx < o + A ** L)] = L
        return out

    def word(self, i: int) -> str:
        i = int(i)
        for L in self.lengths:
            o = self.offset(L)
            if o <= i < o + A ** L:
                f, s = i - o, []
                for _ in range(L):
                    f, r = divmod(f, A)
                    s.append(ALPHABET[r])
                return "".join(reversed(s))
        raise IndexError(i)

    def flat(self, states: np.ndarray, L: int) -> np.ndarray:
        """Global indices of an (n, L) array of letter indices."""
        return self.offset(L) + np.ravel_multi_index(states.T, (A,) * L)


# --------------------------------------------------------------------------- #
# grand-canonical energy
# --------------------------------------------------------------------------- #

def chemical_potential(lw: np.ndarray, p_L: float) -> dict:
    """mu_L = log2 P(L) - log2 Z_L (bits) for unnormalised block log-weights (nats).

    mu_L carries the gauge of H_L (a constant added to the couplings of block L shifts
    log2 Z_L and mu_L together); only E = H_L - mu_L is physical. In the normalised
    gauge (Z_L = 1, H_L = -log2 P(w | L)) mu_L = log2 P(L), reported as well."""
    mx = float(lw.max())
    log2Z = (mx + float(np.log(np.exp(lw - mx).sum()))) / np.log(2.0)
    return {"log2_Z": log2Z, "log2_P_L": float(np.log2(p_L)),
            "mu_bits": float(np.log2(p_L) - log2Z)}


def grand_canonical_energy(space: Space, log_weights, p_len: dict[int, float]
                           ) -> tuple[np.ndarray, dict[int, dict]]:
    """E(w) = H_L(w) - mu_L = -log2 P(w) on the whole space (bits; +inf forbidden),
    and the chemical potentials.

    ``log_weights[L]`` is sum_c V_c on the A^L grid (nats, any additive constant), or a
    zero-argument callable returning it (blocks are then built and freed one at a
    time). ``p_len[L]`` is the probability of length L among all tokens (any
    normalisation over lengths only shifts E by a constant and leaves the landscape
    unchanged).
    """
    E = np.empty(space.size)
    mu = {}
    for L in space.lengths:
        lw = log_weights[L]
        lw = lw() if callable(lw) else lw
        mu[L] = chemical_potential(lw, p_len[L])
        blk = space.block(E, L)
        np.multiply(lw, -1.0 / np.log(2.0), out=blk)
        blk -= mu[L]["mu_bits"]
        del lw
    return E, mu


def widom_insertion(space: Space, E: np.ndarray, L: int) -> dict:
    """Discrete Widom (1963) test-particle identity for one letter inserted into length-L
    words. With W(w) = sum over the A(L+1) insertion events w -> w' of 2^-(E(w') - E(w)),

        <W>_L = sum_{w: P(w)>0} P(w | L) W(w) = (L+1) P(L+1) / P(L)

    whenever every deletion parent of an allowed (L+1)-word is allowed (each w' arises
    from exactly L+1 (parent, event) pairs). The left side is computed directly; the
    deficit measures probability at L+1 whose parents are forbidden at L."""
    if L + 1 not in space.lengths:
        raise ValueError("L+1 not modelled")
    p1 = np.exp2(-space.block(E, L + 1))
    pL = np.exp2(-space.block(E, L))
    allowed = pL > 0
    # sum of P(w') over (L+1)-words whose parent at deletion site p is allowed: the
    # parent does not depend on letter p, so sum that axis out first
    lhs = sum(float(p1.sum(axis=p)[allowed].sum()) for p in range(L + 1))
    PL, PL1 = float(pL.sum()), float(p1.sum())
    return {"L": L, "mean_insertion_factor": lhs / PL, "identity_rhs": (L + 1) * PL1 / PL,
            "P_L": PL, "P_L1": PL1}


# --------------------------------------------------------------------------- #
# move graph
# --------------------------------------------------------------------------- #

def _letters(f: np.ndarray, L: int, p: int) -> np.ndarray:
    return (f // A ** (L - 1 - p)) % A


# Numerical convention, not a model parameter: index arithmetic for large selections
# is done in chunks of this many states to bound temporary memory (~CHUNK*A floats).
CHUNK = 1 << 18


def _chunked(sel: np.ndarray, fn) -> np.ndarray:
    if sel.size <= CHUNK:
        return fn(sel)
    return np.concatenate([fn(sel[k:k + CHUNK]) for k in range(0, sel.size, CHUNK)])


def _moves_from(space: Space, E: np.ndarray, L: int, move: str) -> Iterator[tuple]:
    """For every word of length L, the best neighbour of one move family at one
    position, in tie-break order (ascending position; within a position the
    lowest-energy letter, ties to the lowest letter). Yields (e, index_of): e is the
    neighbour energy for every word of the block (+inf for self-moves), index_of(sel)
    returns the neighbour's global index for the flat block positions ``sel``.
    Only e is materialised at full size; indices are computed for updated words only."""
    o = space.offset(L)
    EL = space.block(E, L)
    flat = EL.ravel()
    shape = (A,) * L
    ax = lambda p: tuple(A if q == p else 1 for q in range(L))   # noqa: E731
    if move == "sub":
        for p in range(L):
            s = A ** (L - 1 - p)
            two = np.partition(EL, 1, axis=p)             # two lowest of each fibre (copy)
            e0, e1 = np.take(two, [0], axis=p), np.take(two, [1], axis=p)
            del two
            own = np.arange(A).reshape(ax(p)) == np.expand_dims(np.argmin(EL, axis=p), p)
            e = np.where(own, e1, e0).ravel()
            del own, e0, e1

            def index_of(sel, s=s, p=p):
                def fn(q):
                    lp = (q // s) % A
                    base = q - lp * s
                    fib = flat[base[:, None] + np.arange(A) * s]
                    fib[np.arange(q.size), lp] = np.inf
                    return o + base + np.argmin(fib, axis=1) * s
                return _chunked(sel, fn)
            yield e, index_of
    elif move == "trans":
        for p in range(L - 1):
            s0, s1 = A ** (L - 1 - p), A ** (L - 2 - p)
            e = np.swapaxes(EL, p, p + 1).ravel()          # E of the swapped word (copy)
            same = (np.arange(A).reshape(ax(p)) == np.arange(A).reshape(ax(p + 1)))
            e[np.broadcast_to(same, shape).ravel()] = np.inf

            def index_of(sel, s0=s0, s1=s1):
                l0, l1 = (sel // s0) % A, (sel // s1) % A
                return o + sel + (l1 - l0) * s0 + (l0 - l1) * s1
            yield e, index_of
    elif move == "del":
        if L - 1 not in space.lengths:
            return
        o1 = space.offset(L - 1)
        EL1 = space.block(E, L - 1)
        for p in range(L):
            s = A ** (L - 1 - p)
            e = np.broadcast_to(np.expand_dims(EL1, p), shape).ravel()

            def index_of(sel, s=s):
                return o1 + (sel // (s * A)) * s + sel % s
            yield e, index_of
    elif move == "ins":
        if L + 1 not in space.lengths:
            return
        o1 = space.offset(L + 1)
        EL1 = space.block(E, L + 1)
        for p in range(L + 1):
            s = A ** (L - p)
            e = np.min(EL1, axis=p).ravel()
            a = np.argmin(EL1, axis=p).ravel()

            def index_of(sel, s=s, a=a):
                return o1 + ((sel // s) * A + a[sel]) * s + sel % s
            yield e, index_of
    else:
        raise ValueError(move)


def _index_dtype(space: Space):
    return np.int32 if space.size < np.iinfo(np.int32).max else np.int64


def best_neighbour(space: Space, E: np.ndarray, moves=MOVES
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Lowest neighbour energy (self excluded) and its global index, per state.

    Index -1 (energy +inf) where no neighbour has finite energy.
    Ties resolved as in the module docstring.
    """
    bad = set(moves) - set(MOVES)
    if bad:
        raise ValueError(bad)
    best_e = np.full(space.size, np.inf)
    best_i = np.full(space.size, -1, dtype=_index_dtype(space))
    for L in space.lengths:
        be = space.block(best_e, L).ravel()
        bi = space.block(best_i, L).ravel()
        for mv in MOVES:
            if mv not in moves:
                continue
            for e, index_of in _moves_from(space, E, L, mv):
                upd = np.flatnonzero(e < be)
                be[upd] = e[upd]
                bi[upd] = index_of(upd)
                del e
    return best_e, best_i


@dataclass
class Landscape:
    """Steepest-descent structure of E on the move graph ``moves``.

    Stored per state: the best neighbour (``best_i``), whether it is strictly lower
    (``lower``), the strict-minimum mask and the descent terminal; the neighbour
    energy is E[best_i]."""

    space: Space
    E: np.ndarray
    moves: tuple[str, ...]
    best_i: np.ndarray
    lower: np.ndarray
    minima: np.ndarray            # finite E strictly below every modelled neighbour
    terminal: np.ndarray

    def best_e(self, i: int) -> float:
        j = int(self.best_i[i])
        return float(self.E[j]) if j >= 0 else float("inf")

    def path(self, word: str) -> list[dict]:
        """Explicit steepest-descent path from ``word`` (same rule as ``terminal``)."""
        cur = self.space.index(word)
        out = [{"word": word, "E_bits": float(self.E[cur]), "move": None}]
        while self.lower[cur]:
            nxt = int(self.best_i[cur])
            w = self.space.word(nxt)
            out.append({"word": w, "E_bits": float(self.E[nxt]),
                        "move": classify_edit(out[-1]["word"], w)})
            cur = nxt
        return out


def landscape(space: Space, E: np.ndarray, moves=MOVES) -> Landscape:
    """Best neighbours, then the terminal state of steepest descent for every state
    (pointer jumping; energy strictly decreases along a path, so there are no cycles)."""
    moves = tuple(m for m in MOVES if m in moves)
    be, bi = best_neighbour(space, E, moves)
    lower = be < E
    minima = np.isfinite(E) & (be > E)
    del be
    target = np.arange(E.size, dtype=bi.dtype)
    target[lower] = bi[lower]
    while True:
        nxt = target[target]
        if np.array_equal(nxt, target):
            break
        target = nxt
    return Landscape(space, E, moves, bi, lower, minima, target)


def basins(ls: Landscape, tokens: tuple[np.ndarray, np.ndarray] | None = None
           ) -> dict[str, np.ndarray]:
    """Per strict minimum: number of states draining to it, model probability mass
    sum 2^-E over its basin, and (if ``tokens = (global indices, counts)`` of the
    observed types is given) observed token mass. Terminal states that are not strict
    minima (plateaus) are omitted. Accumulated in chunks of the state space."""
    m = np.flatnonzero(ls.minima)
    size = np.zeros(m.size)
    mass = np.zeros(m.size)
    if m.size == 0:
        out = {"index": m, "states": size.astype(np.int64), "model_mass": mass}
        if tokens is not None:
            out["token_mass"] = np.zeros(0)
        return out

    def which(t):
        k = np.minimum(np.searchsorted(m, t), m.size - 1)
        return k, m[k] == t

    step = CHUNK * A
    for a in range(0, ls.E.size, step):
        k, ok = which(ls.terminal[a:a + step])
        size += np.bincount(k[ok], minlength=m.size)
        mass += np.bincount(k[ok], weights=np.exp2(-ls.E[a:a + step][ok]), minlength=m.size)
    out = {"index": m, "states": size.astype(np.int64), "model_mass": mass}
    if tokens is not None:
        idx, cnt = tokens
        k, ok = which(ls.terminal[idx])
        out["token_mass"] = np.bincount(k[ok], weights=cnt[ok], minlength=m.size)
    return out


def classify_edit(a: str, b: str) -> str | None:
    """The single edit turning ``a`` into ``b`` (as a move: sub/trans/del/ins)."""
    if len(b) == len(a):
        d = [i for i in range(len(a)) if a[i] != b[i]]
        if len(d) == 1:
            return "sub"
        if len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]]:
            return "trans"
        return None
    if len(b) == len(a) - 1:
        return "del" if any(a[:i] + a[i + 1:] == b for i in range(len(a))) else None
    if len(b) == len(a) + 1:
        return "ins" if any(b[:i] + b[i + 1:] == a for i in range(len(b))) else None
    return None


# --------------------------------------------------------------------------- #
# corruptions and recovery
# --------------------------------------------------------------------------- #

def corruptions(space: Space, states: np.ndarray, L: int, kind: str
                ) -> Iterator[tuple[np.ndarray, np.ndarray]] | None:
    """All single-edit corruption events of type ``kind`` of each word in ``states``
    (n, L): yields (global target index, valid mask) per event slot. One event per
    (position, letter) choice of the channel; slots that reproduce the word itself
    are invalid. Returns None if the corrupted length is outside the space."""
    if kind not in ERRORS:
        raise ValueError(kind)
    if kind == "ins" and L + 1 not in space.lengths:
        return None
    if kind == "del" and (L - 1 not in space.lengths):
        return None
    f = np.ravel_multi_index(states.T, (A,) * L).astype(np.int64)
    return _corruptions(space, f, states, L, kind)


def _corruptions(space, f, states, L, kind):
    o = space.offset(L)
    if kind == "sub":
        for p in range(L):
            s = A ** (L - 1 - p)
            for a in range(A):
                yield o + f + (a - states[:, p]) * s, states[:, p] != a
    elif kind == "trans":
        for p in range(L - 1):
            l0, l1 = states[:, p], states[:, p + 1]
            s0, s1 = A ** (L - 1 - p), A ** (L - 2 - p)
            yield o + f + (l1 - l0) * s0 + (l0 - l1) * s1, l0 != l1
    elif kind == "ins":                       # typist inserted letter a at p
        o1 = space.offset(L + 1)
        ok = np.ones(f.size, dtype=bool)
        for p in range(L + 1):
            s = A ** (L - p)
            for a in range(A):
                yield o1 + ((f // s) * A + a) * s + f % s, ok
    else:                                     # typist omitted position p
        o1 = space.offset(L - 1)
        ok = np.ones(f.size, dtype=bool)
        for p in range(L):
            s = A ** (L - 1 - p)
            yield o1 + (f // (s * A)) * s + f % s, ok


def recovery(ls: Landscape, states: np.ndarray, L: int, kind: str) -> np.ndarray | None:
    """Per word type: fraction of its valid corruption events of type ``kind`` whose
    steepest descent ends exactly on the word (NaN if it has no valid event; None if
    the corruption leaves the modelled space). With ``kind="sub"`` on a
    substitution-only landscape this is ``maxent.substitution_recovery``."""
    it = corruptions(ls.space, states, L, kind)
    if it is None:
        return None
    me = ls.space.flat(states, L)
    hits = np.zeros(states.shape[0])
    n_ev = np.zeros(states.shape[0])
    for tgt, ok in it:
        hits += ok & (ls.terminal[tgt] == me)
        n_ev += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_ev > 0, hits / n_ev, np.nan)


def token_weighted(frac: np.ndarray, counts: np.ndarray) -> tuple[float | None, float]:
    """Token-weighted mean of per-type recovery over types that have at least one
    valid corruption event, and the token weight it rests on (None, 0 if none)."""
    m = np.isfinite(frac)
    w = float(counts[m].sum())
    return (float((frac[m] * counts[m]).sum() / w) if w > 0 else None), w


def move_delta(space: Space, E: np.ndarray, states: np.ndarray, L: int, move: str,
               weights: np.ndarray) -> dict | None:
    """Energy change dE = E(w') - E(w) over the single moves of type ``move`` from the
    words ``states`` (n, L), averaged with ``weights`` over words and uniformly over each
    word's events (the move set of ``corruptions``: insertion = 26(L+1) events, etc.).

    Returns, over words with at least one event inside the space: mean dE of a uniform
    event with a finite-energy target, fraction of events downhill (dE < 0), fraction
    to forbidden targets, and the mean of the lowest dE of each word (its best move of
    this type). None if the move leaves the modelled space."""
    it = corruptions(space, states, L, move)
    if it is None:
        return None
    e0 = E[space.flat(states, L)]
    n = np.zeros(states.shape[0])
    nfin = np.zeros_like(n)
    sfin = np.zeros_like(n)
    down = np.zeros_like(n)
    forb = np.zeros_like(n)
    best = np.full_like(n, np.inf)
    for tgt, ok in it:
        with np.errstate(invalid="ignore"):
            d = E[tgt] - e0
        fin = ok & np.isfinite(d)
        n += ok
        nfin += fin
        sfin += np.where(fin, d, 0.0)
        down += ok & (d < 0)
        forb += ok & np.isinf(E[tgt])
        best = np.where(ok, np.minimum(best, d), best)
    m = (n > 0) & np.isfinite(e0)
    w = weights[m] / weights[m].sum()
    mf = m & (nfin > 0)
    wf = weights[mf] / weights[mf].sum()
    bf = np.isfinite(best[m])
    return {"mean_dE_bits": float((wf * sfin[mf] / nfin[mf]).sum()),
            "frac_downhill": float((w * down[m] / n[m]).sum()),
            "frac_to_forbidden": float((w * forb[m] / n[m]).sum()),
            "mean_best_dE_bits": float((w[bf] * best[m][bf]).sum() / w[bf].sum()),
            "frac_words_with_downhill_move": float((w * (down[m] > 0)).sum())}


# --------------------------------------------------------------------------- #
# noisy channel (Kernighan, Church & Gale 1990)
# --------------------------------------------------------------------------- #

def n_events(c: str, kind: str) -> int:
    """Number of channel events of type ``kind`` available to intended word c."""
    L = len(c)
    if kind == "sub":
        return (A - 1) * L
    if kind == "trans":
        return sum(c[i] != c[i + 1] for i in range(L - 1))
    if kind == "ins":
        return A * (L + 1)
    if kind == "del":
        return L
    raise ValueError(kind)


def preimages(o: str) -> dict[str, tuple[str, int]]:
    """All intended words c from which one channel event produces o, with the
    error type and its multiplicity m(c -> o) (the number of distinct events of c
    yielding o). Multiplicities follow from the bijection between events of c and
    inverse edits of o (e.g. inserting 'l' into 'wil' at position 2 or 3 gives
    'will'; deleting either 'l' of 'will' gives 'wil')."""
    out: dict[str, tuple[str, int]] = {}
    acc: dict[str, Counter] = {k: Counter() for k in ERRORS}
    L = len(o)
    for p in range(L):
        for ch in ALPHABET:
            if ch != o[p]:
                acc["sub"][o[:p] + ch + o[p + 1:]] += 1
        acc["ins"][o[:p] + o[p + 1:]] += 1                  # o has an extra letter
    for p in range(L - 1):
        if o[p] != o[p + 1]:
            acc["trans"][o[:p] + o[p + 1] + o[p] + o[p + 2:]] += 1
    for p in range(L + 1):
        for ch in ALPHABET:
            acc["del"][o[:p] + ch + o[p:]] += 1             # o lost a letter
    for k, cnt in acc.items():
        for c, m in cnt.items():
            if c:
                out[c] = (k, m)
    return out


def channel_log2(o: str, c: str, kind: str | None, m: int, eps: float,
                 rho: dict[str, float]) -> float:
    """log2 K(o | c) of the generative channel: with probability 1 - eps no error; with
    probability eps * rho_t an attempt of type t, realised as one of the M_t(c) events of
    that type chosen uniformly. A source with no event of the attempted type (M_t = 0:
    only transpositions, for words without two unequal adjacent letters) is left
    unchanged, so K(c | c) = 1 - eps + eps * sum_{t: M_t(c) = 0} rho_t and every row of
    K sums to 1 (tested). Otherwise K(o | c) = eps rho_t m(c -> o) / M_t(c)."""
    if kind is None:
        stay = 1.0 - eps + eps * sum(rho[t] for t in ERRORS if n_events(c, t) == 0)
        return float(np.log2(stay))
    return float(np.log2(eps * rho[kind] * m / n_events(c, kind)))


def _decode_terms(o: str, space: Space, E: np.ndarray, rho: dict[str, float]):
    """Candidates {o} + one-edit preimages of o in the space, with log2 P(c) and the
    eps-free part of log2 K(o | c)."""
    cands = [(o, None, 1)] if len(o) in space.lengths else []
    cands += [(c, k, m) for c, (k, m) in preimages(o).items() if len(c) in space.lengths]
    lp = np.array([-E[space.index(c)] for c, _, _ in cands])
    with np.errstate(divide="ignore"):
        part = np.array([0.0 if k is None else np.log2(rho[k] * m / n_events(c, k))
                         for c, k, m in cands])
    return cands, lp, part


def decode_many(o: str, space: Space, E: np.ndarray, eps_grid, rho: dict[str, float]
                ) -> list[tuple[str, float]]:
    """``decode`` for each eps in ``eps_grid`` (candidates computed once)."""
    cands, lp, part = _decode_terms(o, space, E, rho)
    out = []
    for eps in eps_grid:
        ch = part + np.log2(eps)
        if cands and cands[0][1] is None:
            ch[0] = channel_log2(o, o, None, 1, eps, rho)
        sc = lp + ch
        j = int(np.argmax(sc))
        if not np.isfinite(sc[j]):
            out.append((cands[j][0], float("nan")))
        else:
            out.append((cands[j][0], float(1.0 / np.exp2(sc - sc[j]).sum())))
    return out


def decode(o: str, space: Space, E: np.ndarray, eps: float, rho: dict[str, float]
           ) -> tuple[str, float]:
    """MAP intended word: argmax_c log2 P(c) + log2 K(o | c) over c in {o} and the
    one-edit preimages of o inside the modelled space; ties to the observation, then
    to the first candidate in ``preimages`` order. Returns (c, posterior of c)."""
    return decode_many(o, space, E, [eps], rho)[0]


def n_transposable(space: Space, idx: np.ndarray) -> np.ndarray:
    """M_trans(c) = number of adjacent unequal letter pairs, for global indices."""
    idx = np.asarray(idx, dtype=np.int64)
    Ls = space.length_of(idx)
    out = np.zeros(idx.shape, dtype=np.int64)
    for L in space.lengths:
        m = Ls == L
        f = idx[m] - space.offset(L)
        for p in range(L - 1):
            out[m] += _letters(f, L, p) != _letters(f, L, p + 1)
    return out


def n_events_array(space: Space, idx: np.ndarray, kind: str) -> np.ndarray:
    """``n_events`` for an array of global indices."""
    Ls = space.length_of(idx)
    if kind == "sub":
        return (A - 1) * Ls
    if kind == "ins":
        return A * (Ls + 1)
    if kind == "del":
        return Ls
    return n_transposable(space, idx)


def selection_probabilities(space: Space, E: np.ndarray, dictionary: np.ndarray,
                            retained_lengths) -> dict[str, float]:
    """s_t = sum_c P(c | lexicon) f_t(c): probability that an error attempt of type t
    on a lexicon source c ~ P (restricted to finite-energy lexicon words of the space)
    yields an observation that enters the retained sample, i.e. a non-lexicon string
    whose length is in ``retained_lengths``. f_t(c) is the fraction of c's type-t
    events with such an output (0 if M_t(c) = 0: the attempt leaves c unchanged).
    Every retained length must have both neighbouring lengths modelled, so that no
    source of a retained observation lies outside the space."""
    keep = set(retained_lengths)
    if any(L - 1 not in space.lengths or L + 1 not in space.lengths for L in keep):
        raise ValueError(f"retained lengths {sorted(keep)} need both neighbours modelled")
    num = dict.fromkeys(ERRORS, 0.0)
    tot = 0.0
    for L in space.lengths:
        o = space.offset(L)
        f = np.flatnonzero(dictionary[o:o + A ** L] & np.isfinite(space.block(E, L).ravel()))
        if f.size == 0:
            continue
        st = np.stack([_letters(f, L, p) for p in range(L)], axis=1)
        w = np.exp2(-E[o + f])
        tot += float(w.sum())
        for t in ERRORS:
            Lout = L + {"ins": 1, "del": -1}.get(t, 0)
            if Lout not in keep:
                continue
            it = corruptions(space, st, L, t)
            if it is None:
                raise ValueError(f"retained length {Lout} needs length {L} neighbours")
            hit = np.zeros(f.size)
            for tgt, ok in it:
                hit += ok & ~dictionary[tgt]
            M = n_events_array(space, o + f, t).astype(float)
            frac = np.divide(hit, M, out=np.zeros_like(hit), where=M > 0)
            num[t] += float((w * frac).sum())
    return {t: num[t] / tot for t in ERRORS}


def em_channel(space: Space, E: np.ndarray, typos: dict[int, tuple[np.ndarray, np.ndarray]],
               dictionary: np.ndarray, selection: dict[str, float] | None = None,
               tol: float = 1e-10, maxiter: int = 10_000) -> dict:
    """Per-type channel rates rho_t (summing to 1 given an error attempt) by the
    bootstrap of Kernighan, Church & Gale (1990): every retained non-dictionary string
    o is one error away from a dictionary word c; the posterior over (c, t) is
    proportional to P(c) rho_t m(c->o) / M_t(c) with P = 2^-E.

    Without ``selection`` the update is the posterior share of each type among the
    retained observations (KCG's assignment shares): it estimates the forward rates
    only if every type enters the retained sample with equal probability. With
    ``selection = s`` (``selection_probabilities``) the likelihood is conditioned on
    retention, P(o | retained) = sum_{c,t} P(c) rho_t m/M_t(c) / sum_t rho_t s_t, whose
    EM fixed point is rho_t proportional to n_t / s_t (n_t the expected retained count
    of type t): stationarity gives n_t/rho_t - W s_t / sum rho s = 0 with W = sum n_t.
    Only token weights make this a likelihood; type weights give assignment shares.

    ``typos[L] = (states (n, L), weights (n,))``; ``dictionary`` is a boolean mask over
    the space. Each channel event is one entry, so a candidate reached by m events
    carries m entries. Typos with no dictionary preimage in the space are dropped and
    their weight is reported. ``tol`` is a numerical tolerance on max |delta rho|.
    """
    rows, kinds, base, w = [], [], [], []
    start = 0
    for L, (st, wt) in sorted(typos.items()):
        for t_id, t in enumerate(ERRORS):
            # preimage c with o = t(c): reached from o by the inverse corruption
            it = corruptions(space, st, L, INVERSE[t])
            if it is None:
                continue
            for c, ok in it:
                keep = ok & dictionary[c] & np.isfinite(E[c])
                if not keep.any():
                    continue
                c = c[keep]
                rows.append(start + np.flatnonzero(keep))
                kinds.append(np.full(c.size, t_id))
                base.append(-E[c] - np.log2(n_events_array(space, c, t)))
        w.append(np.asarray(wt, dtype=float))
        start += st.shape[0]
    w = np.concatenate(w)
    rows, kinds, base = (np.concatenate(x) for x in (rows, kinds, base))
    used = np.bincount(rows, minlength=w.size) > 0
    s_vec = None if selection is None else np.array([selection[t] for t in ERRORS])
    rho = np.full(len(ERRORS), 1.0 / len(ERRORS))
    done, it, shares = False, 0, rho
    for it in range(1, maxiter + 1):
        with np.errstate(divide="ignore"):
            s = base + np.log2(rho)[kinds]
        mx = np.full(w.size, -np.inf)
        np.maximum.at(mx, rows, s)
        q = np.exp2(s - mx[rows])
        q /= np.bincount(rows, weights=q, minlength=w.size)[rows]
        n_t = np.bincount(kinds, weights=q * w[rows], minlength=len(ERRORS))
        shares = n_t / n_t.sum()
        new = shares if s_vec is None else np.divide(
            n_t, s_vec, out=np.zeros_like(n_t), where=s_vec > 0)
        new = new / new.sum()
        done = bool(np.abs(new - rho).max() < tol)
        rho = new
        if done:
            break
    return {"rho": dict(zip(ERRORS, rho.tolist())),
            "retained_assignment_shares": dict(zip(ERRORS, shares.tolist())),
            "selection": None if selection is None else dict(selection),
            "iterations": it, "converged": done,
            "typos_used": int(used.sum()), "weight_used": float(w[used].sum()),
            "weight_dropped": float(w[~used].sum())}
