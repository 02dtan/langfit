/* Edge-case self-tests for scripts/landscape_engine.js (run under Bun):
 *     bun tests/js/engine_edge.mjs
 * Covers forbidden (-Infinity) couplings, all-forbidden and empty cells, G = 1,
 * G larger than the number of distinct coordinates, extreme couplings (no
 * overflow), id/nearest helpers, and the persistence semantics against the same
 * hand-built grids used by tests/test_landscape_artifact.py.
 */

import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const eng = require('../../scripts/landscape_engine.js');

const A = 26, N = 456976; // 26^4

let failures = 0;
function assert(cond, msg) {
  if (!cond) { failures++; console.error('FAIL:', msg); }
}
function close(a, b, tol) { return Math.abs(a - b) <= tol; }

export function run() {
  // 1. Uniform model at G = 1: one cell, flat landscape.
  {
    const V = new Float64Array(6 * A * A);
    const U1 = new Float64Array(104), U2 = new Float64Array(104);
    const res = eng.compute({ V, U1, U2, mean: [0, 0], extent: { xmin: 0, xmax: 1, ymin: 0, ymax: 1 }, G: 1, topK: 3 });
    assert(close(res.logZ, 4 * Math.log(A), 1e-9), `uniform logZ ${res.logZ}`);
    assert(close(res.S2bits, 4 * (Math.log(A) / Math.LN2), 1e-9), `uniform S2 ${res.S2bits}`);
    assert(res.count[0] === N, `uniform count ${res.count[0]}`);
    assert(close(res.mass[0], 1.0, 1e-9), `uniform mass ${res.mass[0]}`);
    assert(close(res.Ebin[0], 4 * (Math.log(A) / Math.LN2), 1e-9), `uniform Ebin ${res.Ebin[0]}`);
    assert(res.occupancy.occupied === 1 && res.occupancy.max === N, 'uniform occupancy');
  }

  // 2. Forbidden strings: V01[0][0] = -Infinity forbids every string with s0=0,s1=0.
  //    Basis x = s0, y = s1 at G = 26 -> cell (s0, s1); cell 0 is all-forbidden.
  {
    const V = new Float64Array(6 * A * A);
    V[0 * A + 0] = -Infinity; // pair (0,1), letters (0,0)
    const U1 = new Float64Array(104), U2 = new Float64Array(104);
    for (let i = 0; i < 26; i++) { U1[i] = i; U2[26 + i] = i; }
    const res = eng.compute({ V, U1, U2, mean: [0, 0], extent: { xmin: 0, xmax: 25, ymin: 0, ymax: 25 }, G: 26, topK: 3 });
    assert(res.count[0] === A * A, `forbidden cell count ${res.count[0]}`);       // 676 strings, all forbidden
    assert(res.Ebin[0] === Infinity, `forbidden cell Ebin ${res.Ebin[0]}`);
    assert(res.mass[0] === 0, `forbidden cell mass ${res.mass[0]}`);
    let countSum = 0, massSum = 0;
    for (let c = 0; c < 26 * 26; c++) { countSum += res.count[c]; massSum += res.mass[c]; }
    assert(countSum === N, 'count includes forbidden strings');
    assert(close(massSum, 1.0, 1e-9), 'mass excludes forbidden strings');
    assert(close(res.logZ, Math.log(N - A * A), 1e-9), `forbidden logZ ${res.logZ}`);
  }

  // 3. Empty cells: the same basis at G = 64 leaves many cells unoccupied.
  {
    const V = new Float64Array(6 * A * A);
    const U1 = new Float64Array(104), U2 = new Float64Array(104);
    for (let i = 0; i < 26; i++) { U1[i] = i; U2[26 + i] = i; }
    const res = eng.compute({ V, U1, U2, mean: [0, 0], extent: { xmin: 0, xmax: 25, ymin: 0, ymax: 25 }, G: 64, topK: 1 });
    let empty = 0;
    for (let c = 0; c < 64 * 64; c++) {
      if (res.count[c] === 0) { empty++; assert(Number.isNaN(res.Ebin[c]), 'empty cell Ebin not NaN'); }
    }
    assert(empty > 0, 'expected empty cells at G=64');
  }

  // 4. G larger than the number of strings per axis: every string nearly its own cell.
  {
    const V = new Float64Array(6 * A * A);
    const U1 = new Float64Array(104), U2 = new Float64Array(104);
    for (let i = 0; i < 26; i++) { U1[i] = i; U2[26 + i] = i; }
    const res = eng.compute({ V, U1, U2, mean: [0, 0], extent: { xmin: 0, xmax: 25, ymin: 0, ymax: 25 }, G: 1000, topK: 1 });
    assert(res.occupancy.occupied <= N, 'occupied <= N at G=1000');
    assert(res.occupancy.max <= 676, 'max strings per cell bounded by 676 at G=1000');
    // G = 676 makes the single-letter basis exactly one (s0,s1) pair per cell: 676 strings each.
    const res676 = eng.compute({ V, U1, U2, mean: [0, 0], extent: { xmin: 0, xmax: 25, ymin: 0, ymax: 25 }, G: 676, topK: 1 });
    assert(res676.occupancy.occupied === A * A, `G=676 occupied ${res676.occupancy.occupied}`);
    assert(res676.occupancy.max === A * A, `G=676 max ${res676.occupancy.max}`);
    assert(res676.occupancy.singleFraction === 0, 'G=676: no single-string cells for this basis');
  }

  // 5. Extreme couplings: no overflow in logZ / S2 / Ebin.
  {
    let seed = 12345;
    const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
    for (const scl of [1e6, -1e6]) {
      const V = new Float64Array(6 * A * A);
      for (let i = 0; i < V.length; i++) V[i] = (rnd() - 0.5) * scl;
      const U1 = new Float64Array(104), U2 = new Float64Array(104);
      for (let i = 0; i < 26; i++) { U1[i] = i; U2[26 + i] = i; }
      const res = eng.compute({ V, U1, U2, mean: [0, 0], extent: { xmin: 0, xmax: 25, ymin: 0, ymax: 25 }, G: 16, topK: 1 });
      assert(Number.isFinite(res.logZ), `extreme logZ (${scl}) ${res.logZ}`);
      assert(Number.isFinite(res.S2bits), `extreme S2 (${scl}) ${res.S2bits}`);
      let bad = 0;
      for (let c = 0; c < 16 * 16; c++) if (!(Number.isFinite(res.Ebin[c]) || Number.isNaN(res.Ebin[c]))) bad++;
      assert(bad === 0, `extreme couplings produced ${bad} non-finite Ebin values`);
    }
  }

  // 6. decodeId / cellOf / nearestInCell round-trips.
  {
    assert(eng.decodeId(0) === 'aaaa', 'decodeId(0)');
    assert(eng.decodeId(N - 1) === 'zzzz', 'decodeId(N-1)');
    assert(eng.decodeId(1) === 'aaab', 'decodeId(1)');
    assert(eng.decodeId(26) === 'aaba', 'decodeId(26)');
    const V = new Float64Array(6 * A * A);
    const U1 = new Float64Array(104), U2 = new Float64Array(104);
    for (let i = 0; i < 26; i++) { U1[i] = i; U2[26 + i] = i; }
    const res = eng.compute({ V, U1, U2, mean: [0, 0], extent: { xmin: 0, xmax: 25, ymin: 0, ymax: 25 }, G: 26, topK: 1 });
    for (const id of [0, 1, 26, 675, 12345, N - 1]) {
      assert(res.cellOf(id) === res.cells[id], 'cellOf mismatch');
      const cell = res.cells[id];
      const near = res.nearestInCell(cell, 12.5, 12.5);
      assert(near >= 0 && res.cellOf(near) === cell, 'nearestInCell wrong cell');
    }
    const emptyCell = 0; // with V01[0][0]=-inf it is all-forbidden but occupied; use a basis with an empty cell instead
    // nearestInCell on an empty cell: use G=64 basis and find a cell with count 0.
    const res64 = eng.compute({ V, U1, U2, mean: [0, 0], extent: { xmin: 0, xmax: 25, ymin: 0, ymax: 25 }, G: 64, topK: 1 });
    let e0 = -1;
    for (let c = 0; c < 64 * 64; c++) if (res64.count[c] === 0) { e0 = c; break; }
    assert(e0 >= 0 && res64.nearestInCell(e0, 12.5, 12.5) === -1, 'nearestInCell on empty cell');
  }

  // 7. Persistence semantics vs. the hand-built (square) grids verified in Python.
  {
    // Square 7x7 two-wells grid: left basin (2.0, min 0.0 at cell 23), right
    // basin (4.0, min 3.0 at cell 26), pass 5.0 at cell 24, surround 8.0.
    function twoWells7() {
      const E = new Float64Array(49).fill(8.0);
      for (let r = 2; r <= 4; r++) { for (let c = 1; c <= 3; c++) E[r * 7 + c] = 2.0; for (let c = 4; c <= 6; c++) E[r * 7 + c] = 4.0; }
      E[3 * 7 + 2] = 0.0; E[3 * 7 + 5] = 3.0; E[3 * 7 + 3] = 5.0;
      return E;
    }
    const w = eng.persistence(twoWells7(), 7, { adjacency: 'grid8' });
    assert(w.length === 3, `square two-wells count ${w.length}`);
    assert(w[0].cell === 23 && w[0].essential === true && close(w[0].persistence, 8.0, 1e-9), 'square two-wells global minimum');
    const w1 = w[1];
    assert(w1.cell === 26 && close(w1.E_min, 3.0, 1e-9) && close(w1.E_saddle, 4.0, 1e-9)
        && close(w1.persistence, 1.0, 1e-9) && w1.saddle === 18 && w1.merges_into === 23, 'square two-wells right basin');

    // holes / separate components: col 3 all-forbidden, cell (0,0) empty
    const Eh = twoWells7();
    for (let r = 0; r < 7; r++) Eh[r * 7 + 3] = Infinity;
    Eh[0] = NaN;
    const wh = eng.persistence(Eh, 7, { adjacency: 'grid8' });
    const ess = wh.filter((x) => x.essential).map((x) => x.E_min).sort((a, b) => a - b);
    assert(ess.length === 2 && close(ess[0], 0.0, 1e-9) && close(ess[1], 3.0, 1e-9), `holes essential minima ${ess}`);

    // plateau: flat 4x4 with two dips (tests/test_landscape_artifact.py)
    const Ep = new Float64Array(16).fill(0.0);
    Ep[0] = -0.5; Ep[15] = -0.2;
    const wp = eng.persistence(Ep, 4, { adjacency: 'grid8' });
    assert(wp.length === 2, `plateau wells ${wp.length}`);
    assert(wp[0].cell === 0 && wp[0].essential === true && close(wp[0].persistence, 0.5, 1e-9), 'plateau deep well');
    assert(wp[1].cell === 15 && close(wp[1].persistence, 0.2, 1e-9), 'plateau shallow well');

    // flat landscape has no (positive-persistence) well
    const Ef = new Float64Array(36).fill(18.8);
    const wf = eng.persistence(Ef, 6, { adjacency: 'grid8' });
    assert(wf.length === 1 && wf[0].essential === true && close(wf[0].persistence, 0.0, 1e-9), 'flat landscape');

    // single dip: one well at the global minimum
    const E1 = new Float64Array(25).fill(10.0);
    E1[1 * 5 + 1] = 2.0;
    const w1p = eng.persistence(E1, 5, { adjacency: 'grid8' });
    assert(w1p.length === 1 && w1p[0].cell === 6 && w1p[0].essential === true
        && close(w1p[0].E_min, 2.0, 1e-9) && close(w1p[0].persistence, 8.0, 1e-9), 'single dip');
  }

  console.log(failures === 0 ? 'ALL EDGE CASES PASS' : `${failures} EDGE FAILURE(S)`);
  return { failures };
}

if (import.meta.main) {
  const r = run();
  process.exit(r.failures === 0 ? 0 : 1);
}
