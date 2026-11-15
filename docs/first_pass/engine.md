# Landscape engine (browser recomputation)

`scripts/landscape_engine.js` recomputes the Stephens & Bialek energy landscape of
four-letter words directly in the browser (or in Bun), from the fitted pairwise
max-entropy couplings — no Python, no precomputed grids. A resolution slider can
therefore re-bin the landscape at any `G` without re-running the fit.

## What it computes

For a model with pair couplings `V` (one 26×26 matrix per position pair, in
`langfit/maxent.py` clique order `[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]`) and a
2-D projection, every one of the 26⁴ = 456 976 strings is enumerated exactly once:

- log-weight `lw(s) = Σ_pairs V_p(s_i, s_j)` (nats); `P(s) = exp(lw − logZ)`;
  `logZ = logsumexp(lw)` (max-shifted, float64); `S2 = −Σ P log₂ P` (bits).
- Coordinates `(x,y)` are either a linear basis (`U1`, `U2`, `mean`) — the
  projection `Uᵀ(x(s) − mean)` — or precomputed per-string coordinates
  (`coords`, a `Float32Array` of length 2·26⁴, or `Int16Array` quantised against
  the extent) for non-parametric bases such as the energy-metric layout.
- Binning is identical to `langfit/projection.py`:
  `k = clip(⌊(v − lo)/(hi − lo)·G⌋, 0, G−1)`, `cell = row·G + col`. The extent
  `{xmin,xmax,ymin,ymax}` is passed by the caller (so raising `G` does not move
  the axes); if omitted it is derived from the loadings exactly as
  `projection.grid_edges` (the min/max of a separable per-position sum).
- Per cell: `count` (all strings, forbidden included), `mass` (ΣP), and
  `E_bin = −log₂(P(cell)/|cell|)` bits, computed with a per-cell max shift so tiny
  probabilities do not underflow (`projection.bin_energy`). Empty cells are
  `NaN`; cells whose strings are all forbidden are `+Infinity`. Forbidden strings
  are those with any pair cell `V = −Infinity` (their log-weight sums to −∞
  naturally, matching `maxent.py`).
- `words`/`wordP`: the top-`K` most probable strings per cell (sorted by P
  descending, ascending-id tie-break, matching `np.lexsort((-lp, bins))`).

The result also carries helpers `cellOf(id)`, `decodeId(id)`, `nearestInCell(cell,
x, y)` (a counting-sort index of string ids per cell), `energyAt(id)` (energy in
bits, `+Infinity` for forbidden strings), `probAt(id)` (`P(s)`), the raw
log-weights `lws`, and occupancy statistics (`res.occupancy`).

`LandscapeEngine.persistence(Ebin, G, {adjacency})` is a
byte-for-byte port of `projection.persistence` (elder rule, the essential-minimum
convention pairing each component's minimum with its highest cell, `(value, index)`
tie-break, `NaN`/`+Inf` cells as holes); `adjacency` is `'grid4'`, `'grid8'`
(= `projection.grid_adjacency`), or `'substitution'` (= `projection.
substitution_adjacency`, the artifact's native cell graph, built from the
string→cell map). `LandscapeEngine.barrier(wells, thresholdBits)` reproduces the
artifact's main-barrier selection.

## Conventions and parity

The engine is validated against Python (`scripts/engine_golden.py`,
`tests/js/engine_parity.mjs`) on 17 cases: two models (Austen, Twitter 2023), the
PCA and `coupling_spectral` bases, `G ∈ {8, 32, 64, 128}`, plus one non-parametric
`coords` case. Measured maximum parity error across all cases:

| quantity | tolerance | measured |
|---|---|---|
| max \|ΔE_bin\| | 1e-6 bits | 5.0e-10 bits |
| counts | exact | 0 |
| top-3 words | exact (ties allowed) | 0 |
| grid8 wells / barrier | exact | 0 |
| substitution wells / barrier (G ∈ {8,32}) | exact | 0 |
| P-mass per cell | 1e-8 | 5.0e-10 |

## Performance (Bun, M1 laptop, warm)

`compute()` is a single 26⁴ nested-loop pass with accumulated partial sums and
typed arrays (no per-string six-lookup rebuild); the log-sum-exp uses a max shift.
Warm timings (the enumeration dominates and is `G`-independent):

- `G = 32`: ~8–12 ms
- `G = 256`: ~15–25 ms

Both are far inside the 60 ms / 200 ms targets. `substitutionAdjacency(G=32)`
costs ~190 ms but is model-independent (one per basis/`G`, cacheable).

## Occupancy: the cloud does not fill the grid

`compute()` also returns `res.occupancy`, with fields `occupied` (occupied cells),
`fractionOccupied` (occupied/G²), `mean`, `median`, `p90`, `max` (strings per
*occupied* cell), `singleFraction` (fraction of occupied cells holding exactly one
string), and `massWeightedMedian` (the strings-per-cell of the median-probability
string; median of the cell count weighted by cell P-mass). Real figures (PCA, so
the numbers are model-independent up to the fitted couplings):

**Austen, PCA basis**

| G | occupied | fraction | mean | median | p90 | max | single | mass-med |
|---|---|---|---|---|---|---|---|---|
| 8 | 46 | 0.719 | 9934 | 234 | 18168 | 262823 | 0.065 | 4437 |
| 16 | 147 | 0.574 | 3109 | 65 | 5682 | 110183 | 0.082 | 766 |
| 32 | 480 | 0.469 | 952 | 33 | 984 | 82254 | 0.185 | 265 |
| 64 | 1404 | 0.343 | 325 | 18 | 421 | 44082 | 0.176 | 33 |
| 128 | 3958 | 0.242 | 115 | 9 | 172 | 24419 | 0.227 | 7 |
| 256 | 9988 | 0.152 | 46 | 4 | 73 | 15012 | 0.279 | 2 |
| 512 | 23242 | 0.089 | 20 | 2 | 32 | 5094 | 0.361 | 1 |
| 1024 | 49216 | 0.047 | 9 | 2 | 16 | 2196 | 0.469 | 1 |

**Pooled Twitter (MAP), PCA basis**

| G | occupied | fraction | mean | median | p90 | max | single | mass-med |
|---|---|---|---|---|---|---|---|---|
| 8 | 48 | 0.750 | 9520 | 326 | 19157 | 212210 | 0.083 | 5196 |
| 16 | 162 | 0.633 | 2821 | 67 | 4971 | 177507 | 0.111 | 1992 |
| 32 | 503 | 0.491 | 909 | 34 | 1101 | 116643 | 0.171 | 366 |
| 64 | 1476 | 0.360 | 310 | 18 | 437 | 49196 | 0.184 | 74 |
| 128 | 4119 | 0.251 | 111 | 8 | 166 | 32131 | 0.238 | 14 |
| 256 | 10811 | 0.165 | 42 | 4 | 60 | 12078 | 0.315 | 4 |
| 512 | 25177 | 0.096 | 18 | 2 | 30 | 4645 | 0.384 | 1 |
| 1024 | 51949 | 0.050 | 9 | 2 | 16 | 2280 | 0.476 | 1 |

Notes: `mean = 456 976 / occupied` exactly (every string is binned). The cloud
fills only about half the grid at `G = 32` and thins out at higher `G`; occupancy
is extremely skewed (median ≪ mean ≪ max), so the naive `26⁴/G²` average is
misleading. Because the cloud is not space-filling, the "every string in its own
cell" regime is reached around `G ≈ 700` (not the naive `√26⁴ = 676`): at `G =
1024` only ~47 % of occupied cells are singletons, and only ~5 % of the grid is
occupied at all. Once a cell holds a single string `s`, `E_bin = −log₂P(s)`
exactly (no averaging), so beyond that point finer resolution adds nothing.

## Limits

- Couplings are float64; log-sum-exp is max-shifted, so `V` magnitudes up to
  ~1e300 are safe (the per-string sum of six terms cannot overflow until each
  term is ~3e307). Forbidden cells are `−Infinity`.
- `G = 1` is a single cell (`E_bin = log₂26⁴ ≈ 18.8` bits for the uniform model);
  `G` larger than the cloud is fine (empty cells become `NaN`).
- Resolution beyond ~680–700 makes most occupied cells singletons, so `E_bin`
  stops averaging and converges to the per-string energy.

## Running the tests

- `bun tests/js/engine_parity.mjs` — engine vs golden parity (also runnable via
  the `eval` tool: `await import(...)` and call `run()`).
- `bun tests/js/engine_edge.mjs` — edge cases (forbidden/empty cells, `G = 1`,
  `G` > cloud size, extreme couplings, helpers, persistence on hand-built grids).
- `.venv/bin/python -m pytest tests/test_engine_golden.py` — regenerates the
  golden into a temp dir and checks it equals the stored file (~23 s).
