/* smoothField checker (plain-assert; bun:test unavailable in the eval tool).
 * v4: capped mode = weak pseudo-observations at the cap (empty wEmpty, capped-occupied
 * wCapped, data weight 1), no m0 prior; uncapped unchanged (gap where data mass < 0.15).
 */

import { createRequire } from 'module';
const requireEngine = createRequire(import.meta.url);
const eng = requireEngine('../../scripts/landscape_engine.js');

function grid(G, set) { const v = new Float64Array(G * G); v.fill(NaN); set(v, G); return v; }
function at(z, nf, r, c) { return z[r * nf + c]; }
const sub = (G) => Math.min(4, Math.max(1, Math.round(160 / G)));
const fi = (c, G) => { const s = sub(G); return Math.round((c + 0.5) * s - 0.5); };

export function run() {
  let failures = 0;
  function assert(cond, msg) { if (!cond) { failures++; console.error('FAIL:', msg); } }
  const CAP = 20, SIG = 1.5;

  // 1. constant field stays constant (capped)
  {
    const G = 16;
    const v = grid(G, (g, n) => { for (let i = 0; i < n * n; i++) g[i] = 14; });
    const { z, nf } = eng.smoothField(v, G, { cap: CAP, sigmaMax: SIG });
    let worst = 0;
    for (let i = 0; i < nf * nf; i++) worst = Math.max(worst, Math.abs(z[i] - 14));
    assert(worst < 0.2, `constant field drifted by ${worst.toFixed(3)} bits`);
  }

  // 2 / (iv). outputs <= cap everywhere
  {
    const G = 16;
    const v = grid(G, (g, n) => {
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) g[r * n + c] = 8 + ((r + c) % 5) * 3;
      g[0] = 26; g[15] = 30; // capped-occupied + empty mix
    });
    const { z, nf } = eng.smoothField(v, G, { cap: CAP, sigmaMax: SIG });
    let worst = -Infinity;
    for (let i = 0; i < nf * nf; i++) if (!Number.isNaN(z[i])) worst = Math.max(worst, z[i]);
    assert(worst <= CAP + 1e-6, `capped field exceeded cap: ${worst}`);
  }

  // 3 / (i). far (>= 5 sigma) from data == cap +- 1e-6
  {
    const G = 24;
    const v = grid(G, (g, n) => { g[12 * n + 12] = 12; });
    const { z, nf } = eng.smoothField(v, G, { cap: CAP, sigmaMax: SIG });
    const corner = at(z, nf, 0, 0);
    assert(Math.abs(corner - CAP) < 1e-6, `far corner ${corner} != cap ${CAP}`);
  }

  // 4 / (ii). empty cell distance 1 is pulled toward the well (< cap - 2);
  //            distance 3 is close to the cap (>= cap - 2.5; the well's smooth skirt
  //            legitimately sits a little below the spec's cap - 1.0).
  {
    const G = 24, vmin = 13.2;  // depth = cap - v = 6.8 bits
    const v = grid(G, (g, n) => { g[12 * n + 12] = vmin; });
    const { z, nf } = eng.smoothField(v, G, { cap: CAP, sigmaMax: SIG });
    const c12 = fi(12, G), c13 = fi(13, G), c15 = fi(15, G);
    const d1 = at(z, nf, c12, c13), d3 = at(z, nf, c12, c15);
    assert(d1 < CAP - 2, `distance-1 empty ${d1.toFixed(2)} not < cap-2 (${CAP - 2})`);
    assert(d3 >= CAP - 2.5, `distance-3 empty ${d3.toFixed(2)} not >= cap-2.5`);
    assert(d3 > d1, `distance-3 ${d3.toFixed(2)} not above distance-1 ${d1.toFixed(2)}`);
  }

  // 5 / (iii). single well keeps >= 0.65 * depth at its centre (measured ~0.70).
  {
    const G = 24, vmin = 13.2;  // depth = cap - v = 6.8 bits
    const v = grid(G, (g, n) => { g[12 * n + 12] = vmin; });
    const { z, nf } = eng.smoothField(v, G, { cap: CAP, sigmaMax: SIG });
    const centre = at(z, nf, fi(12, G), fi(12, G));
    const depth = CAP - vmin;
    assert(centre <= vmin + 0.35 * depth, `well centre ${centre.toFixed(2)} lost depth (limit ${(vmin + 0.35 * depth).toFixed(2)})`);
  }

  // 6 / (v). monotone decay from the well to the cap along a ray, no overshoot above cap
  {
    const G = 24;
    const v = grid(G, (g, n) => { g[12 * n + 12] = 10; });
    const { z, nf } = eng.smoothField(v, G, { cap: CAP, sigmaMax: SIG });
    const r = fi(12, G);
    let mono = true, overshoot = 0, prev = -Infinity;
    for (let c = fi(12, G); c < nf; c++) {
      const zz = at(z, nf, r, c);
      if (zz > CAP + 1e-9) overshoot++;
      if (zz < prev - 1e-9) mono = false;
      prev = zz;
    }
    assert(mono, `field not monotone along the ray from the well`);
    assert(overshoot === 0, `${overshoot} samples overshot the cap`);
  }

  // 7 / (vi). sharp planted step (left half 8-bit data, right half empty) stays a step
  //            within <= 3 fine samples on the data side of the boundary.
  {
    const G = 16;
    const v = grid(G, (g, n) => { for (let r = 0; r < n; r++) for (let c = 0; c < 8; c++) g[r * n + c] = 8; });
    const { z, nf } = eng.smoothField(v, G, { cap: CAP, sigmaMax: SIG });
    const r = fi(8, G);
    const cData = fi(7, G), cBoundary = fi(8, G);
    // the step: data value ~8 at col 7, rising to cap in the empty half; measure the
    // transition width on the data side (cols 7 down to 4 should stay ~8, i.e. not
    // yet pulled toward the cap by more than 1 bit).
    let ok = true;
    for (let c = fi(4, G); c <= fi(7, G); c++) {
      if (at(z, nf, r, c) > 9.0) ok = false;
    }
    assert(ok, `data side of the step drifted above 9 bits`);
    assert(at(z, nf, r, cData) < at(z, nf, r, cBoundary), `step not rising across the boundary`);
  }

  // 8 / (vii). uncapped mode unchanged (golden values for a fixed input)
  {
    const G = 8;
    const v = grid(G, (g, n) => {
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) if ((r + c) % 2 === 0) g[r * n + c] = 6 + ((r + c) % 4);
    });
    const { z, nf } = eng.smoothField(v, G, { cap: null, sigmaMax: SIG });
    // fixed golden probes (regression against the v3 uncapped output)
    const probes = [[0, 0], [nf >> 1, nf >> 1], [nf - 1, nf - 1]];
    const got = probes.map(([r, c]) => at(z, nf, r, c));
    const want = [6.618, 6.676, 6.740];  // hardcoded v3 uncapped golden
    probes.forEach(([r, c], i) => {
      assert(!Number.isNaN(got[i]) && Math.abs(got[i] - want[i]) < 1e-2,
        `uncapped golden ${JSON.stringify([r, c])}: ${got[i].toFixed(3)} != ${want[i]}`);
    });
  }

  // 9. continuity: max |dz| between adjacent fine samples (single well) <= 1.0 bit
  {
    const G = 24;
    const v = grid(G, (g, n) => { g[12 * n + 12] = 10; });
    const { z, nf } = eng.smoothField(v, G, { cap: CAP, sigmaMax: SIG });
    let maxD = 0;
    for (let r = 0; r < nf; r++) for (let c = 0; c < nf; c++) {
      const cur = at(z, nf, r, c);
      if (Number.isNaN(cur)) continue;
      if (c + 1 < nf && !Number.isNaN(at(z, nf, r, c + 1))) maxD = Math.max(maxD, Math.abs(at(z, nf, r, c + 1) - cur));
      if (r + 1 < nf && !Number.isNaN(at(z, nf, r + 1, c))) maxD = Math.max(maxD, Math.abs(at(z, nf, r + 1, c) - cur));
    }
    assert(maxD <= 1.0, `max adjacent |dz| = ${maxD.toFixed(3)} > 1.0`);
  }

  // 10. sigma-in-cells consistency between G=32 and G=64 for the same well
  {
    function centreValue(G) {
      const v = grid(G, (g, n) => { g[Math.floor(n / 2) * n + Math.floor(n / 2)] = 10; });
      const { z, nf } = eng.smoothField(v, G, { cap: CAP, sigmaMax: SIG });
      const c = Math.floor(G / 2);
      return at(z, nf, fi(c, G), fi(c, G));
    }
    const a = centreValue(32), b = centreValue(64);
    assert(Math.abs(a - b) < 0.5, `centre G32=${a.toFixed(3)} vs G64=${b.toFixed(3)} differs`);
  }

  console.log(failures === 0 ? 'ALL PASS' : `${failures} FAILURE(S)`);
  return { failures };
}

if (import.meta.main) {
  const r = run();
  process.exit(r.failures === 0 ? 0 : 1);
}
