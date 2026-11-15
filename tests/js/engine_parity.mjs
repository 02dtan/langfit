/* Parity runner: checks scripts/landscape_engine.js against the golden reference
 * (tests/data/engine_golden.json, produced by scripts/engine_golden.py).
 *
 * Run with Bun from the repo root:
 *     bun tests/js/engine_parity.mjs
 * (the `eval` tool also runs Bun directly; `await import(...)` it to get the
 * result object without exiting the kernel).
 *
 * Asserts, per case: max |dEbin| < 1e-6 bits, counts exactly equal, top-3 words
 * equal except for exact log2p ties, and the persistent wells (and the main
 * barrier) equal on both the 8-neighbour grid and (where stored) the
 * substitution adjacency. Also prints the per-case timings.
 */

import { createRequire } from 'module';
import { readFileSync } from 'fs';

const require = createRequire(import.meta.url);
const eng = require('../../scripts/landscape_engine.js');

const GOLDEN = new URL('../../tests/data/engine_golden.json', import.meta.url);

function b64f64(b) {
  const r = Buffer.from(b, 'base64');
  return new Float64Array(r.buffer, r.byteOffset, r.byteLength / 8);
}
function b64f32(b) {
  const r = Buffer.from(b, 'base64');
  return new Float32Array(r.buffer, r.byteOffset, r.byteLength / 4);
}

// log-weight of string id (same pair order as the engine / maxent.py).
function lwOf(id, V) {
  let s0 = Math.floor(id / 17576); let rem = id - s0 * 17576;
  let s1 = Math.floor(rem / 676); rem -= s1 * 676;
  let s2 = Math.floor(rem / 26); const s3 = rem - s2 * 26;
  return V[s0 * 26 + s1] + V[676 + s0 * 26 + s2] + V[1352 + s0 * 26 + s3]
       + V[2028 + s1 * 26 + s2] + V[2704 + s1 * 26 + s3] + V[3380 + s2 * 26 + s3];
}

function numEq(a, b, tol) { return (a === Infinity && b === 'inf') || Math.abs(a - b) <= tol; }

function wellsEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i];
    if (x.cell !== y.cell || x.essential !== y.essential) return false;
    if (x.saddle !== y.saddle || x.merges_into !== y.merges_into) return false;
    if (!numEq(x.E_min, y.E_min, 1e-6)) return false;
    if (!numEq(x.E_saddle, y.E_saddle, 1e-6)) return false;
    if (!numEq(x.persistence, y.persistence, 1e-6)) return false;
  }
  return true;
}
function barrierEqual(a, b) {
  if (a === null && b === null) return true;
  if (a === null || b === null) return false;
  if (a.cell !== b.cell || a.wells[0] !== b.wells[0] || a.wells[1] !== b.wells[1]) return false;
  if (!numEq(a.E, b.E, 1e-6)) return false;
  if (!numEq(a.above[0], b.above[0], 1e-6) || !numEq(a.above[1], b.above[1], 1e-6)) return false;
  return true;
}

export function run() {
  const golden = JSON.parse(readFileSync(GOLDEN, 'utf8'));
  const TOP_K = golden.meta.TOP_K;
  const PERSIST_BITS = golden.meta.PERSIST_BITS;
  let failures = 0;
  function assert(cond, msg) {
    if (!cond) { failures++; console.error('FAIL:', msg); }
  }

  const timings = [];
  let maxErr = 0;
  let probeChecked = false;

  for (const c of golden.cases) {
    const V = b64f64(c.V);
    const G = c.G;
    const label = `${c.model}/${c.basis} G${G}${c.coords ? ' [coords]' : ''}`;
    const t0 = performance.now();
    const res = c.coords
      ? eng.compute({ V, coords: b64f32(c.coords), extent: c.extent, G, topK: TOP_K })
      : eng.compute({ V, U1: b64f64(c.U1), U2: b64f64(c.U2), mean: c.mean, extent: c.extent, G, topK: TOP_K });
    const dt = performance.now() - t0;
    timings.push([label, dt]);

    // Cross-check the engine's per-string energyAt / probAt against MaxEnt's
    // energy()/prob() for a few known strings (independent Python reference).
    if (!probeChecked && c.model === 'austen' && golden.meta.probe) {
      for (const p of golden.meta.probe) {
        const e = res.energyAt(p.id), pr = res.probAt(p.id);
        if (p.E_bits === 'inf') assert(e === Infinity, `probe ${p.word}: expected +Inf energy, got ${e}`);
        else assert(Math.abs(e - p.E_bits) < 1e-5, `probe ${p.word}: E_bits ${e} != ${p.E_bits}`);
        assert(Math.abs(pr - p.prob) < 1e-8, `probe ${p.word}: prob ${pr} != ${p.prob}`);
      }
      probeChecked = true;
    }

    let maxD = 0;
    for (let i = 0; i < res.Ebin.length; i++) {
      const ge = c.Ebin[i], e = res.Ebin[i];
      if (ge === null) assert(Number.isNaN(e), `${label} cell ${i}: expected NaN, got ${e}`);
      else if (ge === 'inf') assert(e === Infinity, `${label} cell ${i}: expected +Inf, got ${e}`);
      else {
        assert(!Number.isNaN(e), `${label} cell ${i}: expected finite, got NaN`);
        const d = Math.abs(e - ge); if (d > maxD) maxD = d;
      }
    }
    if (maxD > maxErr) maxErr = maxD;
    assert(maxD < 1e-6, `${label}: max |dEbin| = ${maxD} >= 1e-6`);

    for (let i = 0; i < res.count.length; i++)
      assert(res.count[i] === c.count[i], `${label} cell ${i}: count ${res.count[i]} != ${c.count[i]}`);

    let maxM = 0;
    for (let i = 0; i < res.mass.length; i++) { const d = Math.abs(res.mass[i] - c.mass[i]); if (d > maxM) maxM = d; }
    assert(maxM < 1e-8, `${label}: max |dmass| = ${maxM} >= 1e-8`);

    const logZ = c.logZ;
    for (let cell = 0; cell < G * G; cell++) {
      for (let k = 0; k < TOP_K; k++) {
        const ew = res.words[cell * TOP_K + k], gw = c.top_words[cell * TOP_K + k];
        if (ew === gw) continue;
        if (ew < 0 || gw < 0) { assert(false, `${label} cell ${cell} k${k}: ${ew} vs ${gw} (not a tie)`); continue; }
        const dlp = Math.abs((lwOf(ew, V) - logZ) / Math.LN2 - (lwOf(gw, V) - logZ) / Math.LN2);
        assert(dlp < 1e-9, `${label} cell ${cell} k${k}: ${ew} vs ${gw}, dlp=${dlp} (non-tie)`);
      }
    }

    const wells8 = eng.persistence(res.Ebin, G, { adjacency: 'grid8' });
    assert(wellsEqual(wells8, c.wells_grid8), `${label}: grid8 wells differ`);
    assert(barrierEqual(eng.barrier(wells8, PERSIST_BITS), c.barrier_grid8), `${label}: grid8 barrier differs`);

    if (c.wells_sub !== null) {
      const wellsSub = eng.persistence(res.Ebin, G, { adjacency: 'substitution', cells: res.cells });
      assert(wellsEqual(wellsSub, c.wells_sub), `${label}: substitution wells differ`);
      assert(barrierEqual(eng.barrier(wellsSub, PERSIST_BITS), c.barrier_sub), `${label}: substitution barrier differs`);
    }

    console.log(`${label}: |dEbin|max=${maxD.toExponential(2)} |dmass|max=${maxM.toExponential(2)} t=${dt.toFixed(1)}ms wells8=${wells8.length}`);
  }

  console.log(failures === 0 ? 'ALL PASS' : `${failures} FAILURE(S)`);
  return { failures, timings, maxErr };
}

if (import.meta.main) {
  const r = run();
  process.exit(r.failures === 0 ? 0 : 1);
}
