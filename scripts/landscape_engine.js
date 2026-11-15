/* LandscapeEngine: browser/Bun recomputation of the Stephens & Bialek pairwise
 * max-entropy energy landscape over four-letter strings, from the stored model
 * couplings. Dependency-free ES2020.
 *
 * Conventions mirror langfit/maxent.py and langfit/projection.py exactly:
 *   - V is a flat Float64Array of 6*26*26 couplings, pair p at offset p*676 with
 *     V[p*676 + a*26 + b] = V_{(i,j)}(a,b) for clique (i,j) in
 *     cliques(2) = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)] (a at position i, b at j).
 *     Forbidden cells are -Infinity; a string is forbidden iff any of its six
 *     pair cells is -Infinity (so its log-weight sums to -Infinity naturally).
 *   - log-weight lw(s) = sum_pairs V_p(s_i, s_j) (nats); P(s) = exp(lw - logZ);
 *     logZ = logsumexp(lw); S2 = -(sum P log2 P) in bits.
 *   - Linear mode: U1/U2 are the two projection loadings (length 104 = 4*26, index
 *     i*26 + a for position i, letter a); mean = [m1, m2] = U^T * mean-indicator.
 *     Coordinate (x,y) = (U1 . one-hot(s) - m1, U2 . one-hot(s) - m2).
 *   - Coords mode (non-parametric bases): pass `coords` instead of {U1,U2,mean},
 *     either a Float32Array of length 2*26^4 (x of string id at coords[id], y at
 *     coords[26^4 + id], in the same id order as the enumeration) or an Int16Array
 *     quantised against the passed `extent` (q in [-32768,32767] maps linearly to
 *     [lo, hi]: v = lo + (q + 32768) * (hi - lo) / 65535).
 *   - Binning (projection.bin_index): k = clip(floor((v - lo)/(hi - lo) * G), 0, G-1);
 *     cell = row(y) * G + col(x). Edges span the full coordinate range of all 26^4
 *     strings (extent = {xmin,xmax,ymin,ymax} = the per-axis min/max of coords).
 *   - E_bin = -log2(P(cell)/|cell|) in bits = log2(count) - log2(sum P) per cell,
 *     computed with a per-cell max shift (projection.bin_energy). Empty cells are
 *     NaN; cells whose strings are all forbidden are +Infinity.
 *   - persistence() is a byte-for-byte port of projection.persistence: 0-dimensional
 *     sublevel-set persistence with the elder rule, the essential-minimum convention
 *     (an essential minimum is paired with its component's highest value), the
 *     (value, index) tie-break, and NaN/+Inf cells treated as holes.
 *
 * Use as a browser <script> (exports a global `LandscapeEngine`) or in Bun/Node
 * (module.exports guard).
 */
(function (global, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    global.LandscapeEngine = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  var A = 26, L = 4;
  var A2 = 676, A3 = 17576, N = 456976; // 26, 26^2, 26^3, 26^4
  var LN2 = Math.LN2;
  var NEIGHBOURS_8 = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];
  var NEIGHBOURS_4 = [[-1, 0], [1, 0], [0, -1], [0, 1]];

  function decodeId(id) {
    var s0 = (id / A3) | 0;
    var rem = id - s0 * A3;
    var s1 = (rem / A2) | 0;
    rem -= s1 * A2;
    var s2 = (rem / A) | 0;
    var s3 = rem - s2 * A;
    return String.fromCharCode(97 + s0, 97 + s1, 97 + s2, 97 + s3);
  }

  // ------------------------------------------------------------------------- //
  // grid / substitution adjacency (cell -> Int32Array of neighbour cell ids)
  // ------------------------------------------------------------------------- //

  function gridAdjacency(G, mode) {
    var dirs = mode === 'grid4' ? NEIGHBOURS_4 : NEIGHBOURS_8;
    var n = G * G;
    var out = new Array(n);
    var d;
    for (var k = 0; k < n; k++) {
      var r = (k / G) | 0, c = k % G, nb = [];
      for (d = 0; d < dirs.length; d++) {
        var rr = r + dirs[d][0], cc = c + dirs[d][1];
        if (rr >= 0 && rr < G && cc >= 0 && cc < G) nb.push(rr * G + cc);
      }
      out[k] = Int32Array.from(nb);
    }
    return out;
  }

  // substitution_adjacency (projection.substitution_adjacency): cells u != v are
  // adjacent iff some string of u and some string of v differ by a single letter.
  // bins is an Int32Array of length 26^4 (the cell of each string id). Model-free.
  function substitutionAdjacency(bins, G) {
    var n = G * G;
    var edges = new Set();
    var cells = new Int32Array(26);
    function scan(base, step) {
      var a, b;
      for (a = 0; a < 26; a++) cells[a] = bins[base + a * step];
      for (a = 0; a < 26; a++) {
        var u = cells[a];
        for (b = a + 1; b < 26; b++) {
          var v = cells[b];
          if (u !== v) edges.add((u < v ? u : v) * n + (u < v ? v : u));
        }
      }
    }
    var s0, s1, s2, s3;
    for (s1 = 0; s1 < 26; s1++) for (s2 = 0; s2 < 26; s2++) for (s3 = 0; s3 < 26; s3++)
      scan(s1 * A2 + s2 * A + s3, A3);                 // position 0 varies
    for (s0 = 0; s0 < 26; s0++) for (s2 = 0; s2 < 26; s2++) for (s3 = 0; s3 < 26; s3++)
      scan(s0 * A3 + s2 * A + s3, A2);                 // position 1 varies
    for (s0 = 0; s0 < 26; s0++) for (s1 = 0; s1 < 26; s1++) for (s3 = 0; s3 < 26; s3++)
      scan(s0 * A3 + s1 * A2 + s3, A);                 // position 2 varies
    for (s0 = 0; s0 < 26; s0++) for (s1 = 0; s1 < 26; s1++) for (s2 = 0; s2 < 26; s2++)
      scan(s0 * A3 + s1 * A2 + s2 * A, 1);             // position 3 varies

    var adj = new Array(n);
    var k;
    for (k = 0; k < n; k++) adj[k] = [];
    edges.forEach(function (e) {
      var u = (e / n) | 0, v = e % n;
      adj[u].push(v);
      adj[v].push(u);
    });
    for (k = 0; k < n; k++) adj[k] = Int32Array.from(adj[k]);
    return adj;
  }

  // ------------------------------------------------------------------------- //
  // main computation
  // ------------------------------------------------------------------------- //

  // extent: {xmin,xmax,ymin,ymax} spanning every string's coordinates. When omitted
  // it is derived identically to projection.grid_edges: for linear mode, the min/max
  // of a separable sum of per-position loadings (accumulated in position order); for
  // coords mode, the min/max of the supplied coordinates.
  function compute(cfg) {
    var V = cfg.V;
    var G = cfg.G;
    var topK = cfg.topK == null ? 3 : cfg.topK;
    var coords = cfg.coords;
    var isQ = coords && coords instanceof Int16Array;   // quantised against extent
    var linear = !coords;
    var U1, U2, m1, m2;
    if (linear) { U1 = cfg.U1; U2 = cfg.U2; m1 = cfg.mean[0]; m2 = cfg.mean[1]; }

    var xmin, xmax, ymin, ymax;
    if (cfg.extent) {
      xmin = cfg.extent.xmin; xmax = cfg.extent.xmax;
      ymin = cfg.extent.ymin; ymax = cfg.extent.ymax;
    } else if (linear) {
      var mn1 = [Infinity, Infinity, Infinity, Infinity], mx1 = [-Infinity, -Infinity, -Infinity, -Infinity];
      var mn2 = [Infinity, Infinity, Infinity, Infinity], mx2 = [-Infinity, -Infinity, -Infinity, -Infinity];
      var i, a;
      for (i = 0; i < 4; i++) {
        for (a = 0; a < 26; a++) {
          var v1 = U1[i * A + a], v2 = U2[i * A + a];
          if (v1 < mn1[i]) mn1[i] = v1;
          if (v1 > mx1[i]) mx1[i] = v1;
          if (v2 < mn2[i]) mn2[i] = v2;
          if (v2 > mx2[i]) mx2[i] = v2;
        }
      }
      xmin = (((mn1[0] + mn1[1]) + mn1[2]) + mn1[3]) - m1;
      xmax = (((mx1[0] + mx1[1]) + mx1[2]) + mx1[3]) - m1;
      ymin = (((mn2[0] + mn2[1]) + mn2[2]) + mn2[3]) - m2;
      ymax = (((mx2[0] + mx2[1]) + mx2[2]) + mx2[3]) - m2;
    } else if (isQ) {
      throw new Error('quantised (Int16) coords require an explicit extent');
    } else {
      xmin = Infinity; xmax = -Infinity; ymin = Infinity; ymax = -Infinity;
      for (var tt = 0; tt < N; tt++) {
        var xv = coords[tt], yv = coords[N + tt];
        if (xv < xmin) xmin = xv;
        if (xv > xmax) xmax = xv;
        if (yv < ymin) ymin = yv;
        if (yv > ymax) ymax = yv;
      }
    }

    var qsx = isQ ? (xmax - xmin) / 65535 : 0;
    var qsy = isQ ? (ymax - ymin) / 65535 : 0;
    var coordMode = linear ? 0 : (isQ ? 2 : 1);

    var nCells = G * G;
    var lws = new Float64Array(N);
    var cells = new Int32Array(N);
    var count = new Int32Array(nCells);
    var maxlw = new Float64Array(nCells);
    for (var c = 0; c < nCells; c++) maxlw[c] = -Infinity;
    var maxLw = -Infinity;

    var invwX = xmax - xmin, invwY = ymax - ymin;
    var Gm1 = G - 1;

    var s0, s1, s2, s3, id = 0;
    for (s0 = 0; s0 < 26; s0++) {
      var u10 = U1 ? U1[s0] : 0, u20 = U2 ? U2[s0] : 0;
      var o0 = s0 * A;                                  // V01 row
      for (s1 = 0; s1 < 26; s1++) {
        var u11 = U1 ? U1[26 + s1] : 0, u21 = U2 ? U2[26 + s1] : 0;
        var t01 = V[o0 + s1];                           // V01[s0][s1]
        var px1 = u10 + u11, py1 = u20 + u21;
        var o1 = s1 * A;
        for (s2 = 0; s2 < 26; s2++) {
          var u12 = U1 ? U1[52 + s2] : 0, u22 = U2 ? U2[52 + s2] : 0;
          var t02 = V[676 + o0 + s2] + V[2028 + o1 + s2]; // V02[s0][s2] + V12[s1][s2]
          var px2 = px1 + u12, py2 = py1 + u22;
          var t012 = t01 + t02;
          var off03 = 1352 + o0, off13 = 2704 + o1, off23 = 3380 + s2 * A;
          for (s3 = 0; s3 < 26; s3++) {
            var lw = t012 + V[off03 + s3] + V[off13 + s3] + V[off23 + s3];
            var x, y;
            if (coordMode === 0) {
              x = px2 + U1[78 + s3] - m1;
              y = py2 + U2[78 + s3] - m2;
            } else if (coordMode === 1) {
              x = coords[id];
              y = coords[N + id];
            } else {
              x = xmin + (coords[id] + 32768) * qsx;
              y = ymin + (coords[N + id] + 32768) * qsy;
            }
            var bx = Math.floor((x - xmin) / invwX * G);
            bx = bx < 0 ? 0 : (bx >= G ? Gm1 : bx);
            var by = Math.floor((y - ymin) / invwY * G);
            by = by < 0 ? 0 : (by >= G ? Gm1 : by);
            var cell = by * G + bx;
            lws[id] = lw;
            cells[id] = cell;
            count[cell]++;
            if (lw > maxlw[cell]) maxlw[cell] = lw;
            if (lw > maxLw) maxLw = lw;
            id++;
          }
        }
      }
    }

    // logZ (nats) by max-shifted log-sum-exp.
    var sumExp = 0.0;
    for (var t = 0; t < N; t++) {
      var lwv = lws[t];
      if (lwv !== -Infinity) sumExp += Math.exp(lwv - maxLw);
    }
    var logZ = maxLw === -Infinity ? -Infinity : maxLw + Math.log(sumExp);

    // Second pass: per-cell bin energy sums, top-K words, entropy.
    var sNats = new Float64Array(nCells);
    var topLw = new Float64Array(nCells * topK);
    var words = new Int32Array(nCells * topK);
    var topCnt = new Int32Array(nCells);
    words.fill(-1);
    topLw.fill(-Infinity);
    var meanE = 0.0;

    for (var q = 0; q < N; q++) {
      var lwq = lws[q];
      if (lwq === -Infinity) continue;
      var cq = cells[q];
      var log2p = (lwq - logZ) / LN2;
      sNats[cq] += Math.exp(lwq - maxlw[cq]);
      meanE += Math.exp(lwq - logZ) * lwq;
      // insertion into a descending (log2p, id ascending) top-K list, matching
      // numpy's lexsort((-lp, bins)) tie-break (stable -> ascending index).
      var base = cq * topK;
      var cnt = topCnt[cq];
      var p = 0;
      while (p < cnt && topLw[base + p] >= log2p) p++;
      if (p < topK) {
        var end = cnt < topK ? cnt : topK - 1;
        for (var j = end; j > p; j--) {
          topLw[base + j] = topLw[base + j - 1];
          words[base + j] = words[base + j - 1];
        }
        topLw[base + p] = log2p;
        words[base + p] = q;
        if (cnt < topK) topCnt[cq] = cnt + 1;
      }
    }

    var Ebin = new Float64Array(nCells);
    var mass = new Float64Array(nCells);
    var wordP = new Float32Array(nCells * topK);
    for (var cc = 0; cc < nCells; cc++) {
      var cntc = count[cc];
      var mlw = maxlw[cc];
      if (cntc === 0) {
        Ebin[cc] = NaN;
        mass[cc] = 0;
      } else if (mlw === -Infinity) {
        Ebin[cc] = Infinity;
        mass[cc] = 0;
      } else {
        var shift = (mlw - logZ) / LN2;
        Ebin[cc] = Math.log(cntc) / LN2 - shift - Math.log(sNats[cc]) / LN2;
        mass[cc] = sNats[cc] * Math.exp(mlw - logZ);
      }
      var b2 = cc * topK;
      for (var jj = 0; jj < topCnt[cc]; jj++) {
        wordP[b2 + jj] = Math.pow(2, topLw[b2 + jj]);
      }
    }
    var S2bits = (logZ - meanE) / LN2;

    // Counting-sort index of string ids per cell (ascending id within a cell).
    var cellStart = new Int32Array(nCells + 1);
    for (var w = 0; w < N; w++) cellStart[cells[w] + 1]++;
    for (var cc2 = 0; cc2 < nCells; cc2++) cellStart[cc2 + 1] += cellStart[cc2];
    var cellOrder = new Int32Array(N);
    var cursor = cellStart.slice(0, nCells);
    for (var w2 = 0; w2 < N; w2++) {
      var cw = cells[w2];
      cellOrder[cursor[cw]++] = w2;
    }

    var res = {
      G: G,
      x: _centres(xmin, xmax, G),
      y: _centres(ymin, ymax, G),
      count: count,
      mass: mass,
      Ebin: Ebin,
      words: words,
      wordP: wordP,
      logZ: logZ,
      S2bits: S2bits,
      // internals used by helpers / substitution adjacency
      cells: cells,
      cellStart: cellStart,
      cellOrder: cellOrder,
      _mode: coordMode, _coords: coords, _isQ: isQ,
      _xmin: xmin, _ymin: ymin, _qsx: qsx, _qsy: qsy,
      U1: U1, U2: U2, m1: m1, m2: m2,
      occupancy: _occupancy(count, mass, nCells)
    };
    res.cellOf = function (i) { return cells[i]; };
    res.decodeId = decodeId;
    res.nearestInCell = function (cell, x, y) { return _nearestInCell(res, cell, x, y); };
    // Per-string log-weights (nats) and derived E(s)/P(s) in bits, so the 2-D
    // "nearest word (cursor)" mode can report E(s) and P for the hovered string
    // without re-implementing the pairwise sum in the page.
    res.lws = lws;
    res.energyAt = function (id) { return (logZ - lws[id]) / LN2; };
    res.probAt = function (id) { return Math.pow(2, (lws[id] - logZ) / LN2); };
    res.xyAt = function (id) { return _xyAt(res, id); };
    return res;
  }

  function _centres(lo, hi, G) {
    var w = (hi - lo) / G;
    var out = new Float64Array(G);
    for (var i = 0; i < G; i++) out[i] = lo + (i + 0.5) * w;
    return out;
  }

  function _xyAt(res, id) {
    if (res._mode === 0) {
      var s0 = (id / A3) | 0, rem = id - s0 * A3;
      var s1 = (rem / A2) | 0; rem -= s1 * A2;
      var s2 = (rem / A) | 0; var s3 = rem - s2 * A;
      return [((res.U1[s0] + res.U1[26 + s1]) + res.U1[52 + s2]) + res.U1[78 + s3] - res.m1,
              ((res.U2[s0] + res.U2[26 + s1]) + res.U2[52 + s2]) + res.U2[78 + s3] - res.m2];
    } else if (res._mode === 1) {
      return [res._coords[id], res._coords[N + id]];
    }
    return [res._xmin + (res._coords[id] + 32768) * res._qsx,
            res._ymin + (res._coords[N + id] + 32768) * res._qsy];
  }

  function _nearestInCell(res, cell, x, y) {
    var start = res.cellStart[cell], end = res.cellStart[cell + 1];
    if (start >= end) return -1;
    var best = -1, bestD = Infinity;
    for (var k = start; k < end; k++) {
      var id = res.cellOrder[k];
      var xy = _xyAt(res, id);
      var dx = xy[0] - x, dy = xy[1] - y, d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = id; }
    }
    return best;
  }

  function _occupancy(count, mass, nCells) {
    var occupied = 0, single = 0, countSum = 0, maxCnt = 0, totalMass = 0;
    for (var c = 0; c < nCells; c++) {
      var cnt = count[c];
      if (cnt > 0) {
        occupied++;
        countSum += cnt;
        if (cnt > maxCnt) maxCnt = cnt;
        if (cnt === 1) single++;
        totalMass += mass[c];
      }
    }
    var occIdx = new Int32Array(occupied);
    var t = 0;
    for (var c2 = 0; c2 < nCells; c2++) {
      if (count[c2] > 0) { occIdx[t] = c2; t++; }
    }
    occIdx.subarray(0, occupied).sort(function (a, b) { return count[a] - count[b]; });
    var occSorted = new Float64Array(occupied);
    for (var k2 = 0; k2 < occupied; k2++) occSorted[k2] = count[occIdx[k2]];
    var median = _quantile(occSorted, 0.5);
    var p90 = _quantile(occSorted, 0.9);
    var wm = NaN, cum = 0;
    for (var k = 0; k < occupied; k++) {
      cum += mass[occIdx[k]];
      if (cum >= 0.5 * totalMass) { wm = count[occIdx[k]]; break; }
    }
    return {
      occupied: occupied,
      fractionOccupied: occupied / nCells,
      mean: occupied > 0 ? countSum / occupied : NaN,
      median: median,
      p90: p90,
      max: maxCnt,
      singleFraction: occupied > 0 ? single / occupied : NaN,
      massWeightedMedian: wm
    };
  }

  function _quantile(sorted, p) {
    var n = sorted.length;
    if (n === 0) return NaN;
    var pos = (n - 1) * p;
    var lo = Math.floor(pos), hi = Math.ceil(pos);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
  }

  // ------------------------------------------------------------------------- //
  // persistence (port of projection.persistence)
  // ------------------------------------------------------------------------- //

  function persistence(Ebin, G, opts) {
    opts = opts || {};
    var n = G * G;
    var adj;
    var mode = opts.adjacency || 'grid8';
    if (mode === 'grid4' || mode === 'grid8') {
      adj = gridAdjacency(G, mode);
    } else if (mode === 'substitution') {
      var bins = opts.bins || opts.cells;
      if (!bins) throw new Error('substitution adjacency needs bins (Int32Array of length 26^4)');
      adj = substitutionAdjacency(bins, G);
    } else {
      throw new Error('unknown adjacency: ' + mode);
    }
    return persistenceCore(Ebin, adj);
  }

  function persistenceCore(flat, adj) {
    var n = flat.length;
    var order = [];
    var k;
    for (k = 0; k < n; k++) if (Number.isFinite(flat[k])) order.push(k);
    order.sort(function (a, b) { return (flat[a] - flat[b]) || (a - b); });

    var parent = new Int32Array(n);
    parent.fill(-1);
    var rootMin = new Int32Array(n);
    var out = new Map();

    function find(kk) {
      while (parent[kk] !== kk) {
        parent[kk] = parent[parent[kk]];
        kk = parent[kk];
      }
      return kk;
    }

    for (var oi = 0; oi < order.length; oi++) {
      var kk = order[oi];
      var roots = new Set();
      var nbs = adj[kk];
      for (var ni = 0; ni < nbs.length; ni++) {
        var nb = nbs[ni];
        if (parent[nb] !== -1) roots.add(find(nb));
      }
      parent[kk] = kk;
      if (roots.size === 0) {
        rootMin[kk] = kk;
        out.set(kk, { cell: kk, E_min: flat[kk], E_saddle: Infinity, persistence: Infinity,
                      saddle: null, merges_into: null, essential: true });
        continue;
      }
      var rootsArr = Array.from(roots);
      var mins = rootsArr.map(function (r) { return rootMin[r]; })
        .sort(function (a, b) { return (flat[a] - flat[b]) || (a - b); });
      var elder = mins[0];
      rootsArr.forEach(function (r) { parent[r] = kk; });
      rootMin[kk] = elder;
      for (var mi = 1; mi < mins.length; mi++) {
        var m = mins[mi];
        var w = out.get(m);
        w.E_saddle = flat[kk];
        w.persistence = flat[kk] - flat[m];
        w.saddle = kk;
        w.merges_into = elder;
        w.essential = false;
      }
    }

    var top = new Map();
    for (var oj = 0; oj < order.length; oj++) {
      var kj = order[oj];
      top.set(rootMin[find(kj)], flat[kj]);
    }
    top.forEach(function (hi, m) {
      out.get(m).persistence = hi - out.get(m).E_min;
    });

    var wells = Array.from(out.values());
    wells.sort(function (a, b) {
      return (-a.persistence) - (-b.persistence) ||
             (-a.essential) - (-b.essential) ||
             a.E_min - b.E_min;
    });
    return wells;
  }

  // The artifact's `barrier`: the most persistent non-essential well (with
  // persistence >= thresholdBits) and the well it merges into, plus the saddle.
  function barrier(wells, thresholdBits) {
    var t = thresholdBits == null ? 1.0 : thresholdBits;
    var byCell = new Map();
    for (var i = 0; i < wells.length; i++) byCell.set(wells[i].cell, wells[i]);
    for (var j = 0; j < wells.length; j++) {
      var w = wells[j];
      if (!w.essential && w.persistence >= t) {
        var other = byCell.get(w.merges_into);
        return {
          cell: w.saddle, E: w.E_saddle,
          wells: [w.cell, other.cell],
          above: [w.E_saddle - w.E_min, w.E_saddle - other.E_min]
        };
      }
    }
    return null;
  }

  return {
    compute: compute,
    decodeId: decodeId,
    persistence: persistence,
    barrier: barrier,
    gridAdjacency: gridAdjacency,
    substitutionAdjacency: substitutionAdjacency,
    smoothField: smoothField,
    constants: { A: A, L: L, N: N }
  };
});

/* Boltzmann-domain kernel regression of a GxG value grid (row-major Float64Array,
   NaN = empty/unmeasured cell) onto a FINE grid of nf = G*sub points per axis.

   Each cell centre is at coordinate u = i (cell i centre at u = i); the fine grid
   samples u_k = -0.5 + (k+0.5)/sub (sub = min(4, max(1, round(160/G)))), i.e. half a
   cell of extrapolation at each end. sub is chosen so nf <= ~192 (fast Plotly surface).

   A cell is DATA iff it is occupied (finite) and, in capped mode, its value v < cap.
   In capped mode every cell is an observation: data (v < cap) weight 1, occupied-but-
   capped (v >= cap) weight wCapped = 0.5 (a real measured high-energy cell), and
   empty/forbidden cells weight wEmpty = 0.25 — the last two as WEAK pseudo-
   observations AT the cap (not pinned; each value is the regression's fitted value).
   Each observation enters in the Boltzmann domain with a = 2^(-(v - vmin)), with
   v = cap for the cap pseudo-observations (vmin = min over data, log-sum-exp).

   Kernel regression with unnormalised Gaussian K(d) = exp(-d^2/2s^2), separable
   (row pass then column pass), truncated at 5s; the grid border is handled by clamping
   the kernel window (border empty cells are cap observations, so the surface does not
   drift at the edge).

     CAPPED:   v~ = vmin - log2(N/D), N = sum K K w a, D = sum K K w. Far from data
               every neighbour is a cap observation, so v~ = cap exactly; an empty
               cell next to a well is pulled below the cap for continuity.
     UNCAPPED: v~ = vmin - log2(N/D) only where D >= 0.15 (data-kernel mass), else NaN
               (no pseudo-observations; unchanged).

   ADAPTIVE length (scale blend): two scales s_min = max(0.5, 0.4*s_max) and s_max
   (slider "length (cells)", default 1.5). The two fields S_min, S_max are computed;
   a local-frequency map rho (mean |v_a - v_b| over each data cell's occupied
   4-neighbours, 0 if none) is blurred with the same NW-kernel (s = 1.5 cells, over
   data cells), bilinearly sampled to the fine grid, and blended
   w = clamp(rho/rho0, 0, 1), rho0 = 3 bits/cell (display parameter, documented):
   field = w*S_min + (1-w)*S_max (both already <= cap, so the blend is too). Dense
   jagged regions keep the short scale (cell-to-cell shifts preserved); flat/isolated
   regions use the long scale (smooth skirts).

   Returns { z: Float64Array(nf*nf) (row-major, NaN = gap), nf, sub }. */
function smoothField(vals, G, opts) {
  opts = opts || {};
  var cap = (opts.cap != null) ? opts.cap : null;
  var sMax = (opts.sigmaMax != null) ? opts.sigmaMax : 1.5;
  var sMin = Math.max(0.5, 0.4 * sMax);
  var wEmpty = 0.25, wCapped = 0.5, rho0 = 3.0, blurSigma = 1.5;
  var sub = Math.min(4, Math.max(1, Math.round(160 / G)));
  var nf = G * sub;

  // Cell roles + kernel weights + Boltzmann weights. Capped: data (v<cap) weight 1,
  // occupied-but-capped (v>=cap) weight wCapped, empty/forbidden weight wEmpty — all
  // as pseudo-observations at the cap. Uncapped: occupied weight 1, empty 0 (gap).
  var data = new Uint8Array(G * G);
  var w = new Float64Array(G * G);
  var vmin = Infinity;
  var i;
  for (i = 0; i < G * G; i++) {
    var v = vals[i];
    var finite = (v === v && v !== Infinity);
    if (cap != null) {
      if (finite && v < cap) { data[i] = 1; w[i] = 1; if (v < vmin) vmin = v; }
      else if (finite) { w[i] = wCapped; }
      else { w[i] = wEmpty; }
    } else {
      if (finite) { data[i] = 1; w[i] = 1; if (v < vmin) vmin = v; }
      else { w[i] = 0; }
    }
  }
  if (vmin === Infinity) {
    var noz = new Float64Array(nf * nf);
    if (cap != null) for (i = 0; i < nf * nf; i++) noz[i] = cap;
    else for (i = 0; i < nf * nf; i++) noz[i] = NaN;
    return { z: noz, nf: nf, sub: sub };
  }
  var a = new Float64Array(G * G);
  for (i = 0; i < G * G; i++) {
    var vv2 = vals[i];
    if (cap != null) {
      a[i] = data[i] ? Math.pow(2, -(vv2 - vmin)) : Math.pow(2, -(cap - vmin));
    } else {
      if (data[i]) a[i] = Math.pow(2, -(vv2 - vmin));
    }
  }

  // separable kernel regression at scale s, on the fine grid
  function kernelField(s) {
    var r = Math.max(1, Math.ceil(5 * s));
    var Nh = new Float64Array(nf * G), Dh = new Float64Array(nf * G);
    for (var j = 0; j < G; j++) {
      var base = j * G;
      for (var u = 0; u < nf; u++) {
        var fu = -0.5 + (u + 0.5) / sub;
        var N = 0, D = 0;
        var i0 = Math.max(0, Math.ceil(fu - r)), i1 = Math.min(G - 1, Math.floor(fu + r));
        for (var ii = i0; ii <= i1; ii++) {
          var d = fu - ii;
          var kw = Math.exp(-(d * d) / (2 * s * s));
          N += kw * w[base + ii] * a[base + ii];
          D += kw * w[base + ii];
        }
        Nh[j * nf + u] = N; Dh[j * nf + u] = D;
      }
    }
    var Nout = new Float64Array(nf * nf), Dout = new Float64Array(nf * nf);
    for (var vv = 0; vv < nf; vv++) {
      var fv = -0.5 + (vv + 0.5) / sub;
      for (var u = 0; u < nf; u++) {
        var N = 0, D = 0;
        var j0 = Math.max(0, Math.ceil(fv - r)), j1 = Math.min(G - 1, Math.floor(fv + r));
        for (var jj = j0; jj <= j1; jj++) {
          var d = fv - jj;
          var kw = Math.exp(-(d * d) / (2 * s * s));
          N += kw * Nh[jj * nf + u];
          D += kw * Dh[jj * nf + u];
        }
        Nout[vv * nf + u] = N; Dout[vv * nf + u] = D;
      }
    }
    var f = new Float64Array(nf * nf);
    if (cap != null) {
      for (var idx = 0; idx < nf * nf; idx++) {
        f[idx] = vmin - Math.log2(Nout[idx] / Dout[idx]);
      }
    } else {
      for (var idx = 0; idx < nf * nf; idx++) {
        f[idx] = Dout[idx] >= 0.15 ? vmin - Math.log2(Nout[idx] / Dout[idx]) : NaN;
      }
    }
    return f;
  }

  var Smin = kernelField(sMin);
  var Smax = kernelField(sMax);

  // local-frequency map rho (mean |dv| over occupied 4-neighbours)
  var rho = new Float64Array(G * G);
  for (var ii = 0; ii < G; ii++) for (var jj = 0; jj < G; jj++) {
    var idx = ii * G + jj;
    if (!data[idx]) continue;
    var s = 0, n = 0;
    if (ii > 0 && data[idx - G]) { s += Math.abs(vals[idx] - vals[idx - G]); n++; }
    if (ii + 1 < G && data[idx + G]) { s += Math.abs(vals[idx] - vals[idx + G]); n++; }
    if (jj > 0 && data[idx - 1]) { s += Math.abs(vals[idx] - vals[idx - 1]); n++; }
    if (jj + 1 < G && data[idx + 1]) { s += Math.abs(vals[idx] - vals[idx + 1]); n++; }
    rho[idx] = n > 0 ? s / n : 0;
  }
  // blur rho with NW-kernel (s = blurSigma) over data cells
  var rhoBlur = new Float64Array(G * G);
  var rb = Math.ceil(3 * blurSigma);
  for (var ii = 0; ii < G; ii++) for (var jj = 0; jj < G; jj++) {
    var s = 0, w = 0;
    var i0 = Math.max(0, ii - rb), i1 = Math.min(G - 1, ii + rb);
    var j0 = Math.max(0, jj - rb), j1 = Math.min(G - 1, jj + rb);
    for (var iii = i0; iii <= i1; iii++) for (var jjj = j0; jjj <= j1; jjj++) {
      var idx2 = iii * G + jjj;
      if (!data[idx2]) continue;
      var dx = ii - iii, dy = jj - jjj;
      var kw = Math.exp(-(dx * dx + dy * dy) / (2 * blurSigma * blurSigma));
      s += kw * rho[idx2]; w += kw;
    }
    rhoBlur[ii * G + jj] = w > 0 ? s / w : 0;
  }
  // blend: field = w*Smin + (1-w)*Smax (energy domain)
  var out = new Float64Array(nf * nf);
  for (var vv = 0; vv < nf; vv++) {
    var fv = -0.5 + (vv + 0.5) / sub;
    var r0 = Math.floor(fv), ty = fv - r0;
    var r0c = Math.max(0, Math.min(G - 1, r0)), r1c = Math.max(0, Math.min(G - 1, r0 + 1));
    for (var u = 0; u < nf; u++) {
      var fu = -0.5 + (u + 0.5) / sub;
      var c0 = Math.floor(fu), tx = fu - c0;
      var c0c = Math.max(0, Math.min(G - 1, c0)), c1c = Math.max(0, Math.min(G - 1, c0 + 1));
      var v00 = rhoBlur[r0c * G + c0c], v10 = rhoBlur[r0c * G + c1c];
      var v01 = rhoBlur[r1c * G + c0c], v11 = rhoBlur[r1c * G + c1c];
      var rr = (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
      var wBlend = rr / rho0; if (wBlend > 1) wBlend = 1; if (wBlend < 0) wBlend = 0;
      var smn = Smin[vv * nf + u], smx = Smax[vv * nf + u];
      out[vv * nf + u] = wBlend * smn + (1 - wBlend) * smx;
    }
  }
  return { z: out, nf: nf, sub: sub };
}
