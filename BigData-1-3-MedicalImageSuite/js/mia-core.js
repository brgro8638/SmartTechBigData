/*
 * Medical image analysis core (JavaScript implementation).
 *
 * Implements SPEC.md step by step, mirroring python/mia/own.py.
 * Plain script (no ES modules) so index.html works when opened directly
 * from disk, and the same file can be require()'d by Node (cli.js).
 * Images are flat Float64Arrays in row-major order (index = y * W + x).
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.MIA = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

  // ---------------------------------------------------------------------
  // .npy reading / writing (shared file format with Python)
  // ---------------------------------------------------------------------

  const NPY_TYPES = {
    "<f4": Float32Array, "<f8": Float64Array,
    "|i1": Int8Array, "|u1": Uint8Array, "|b1": Uint8Array,
    "<i2": Int16Array, "<u2": Uint16Array,
    "<i4": Int32Array, "<u4": Uint32Array,
    "<i8": BigInt64Array, "<u8": BigUint64Array,
  };

  function parseNpy(buffer) {
    const bytes = new Uint8Array(buffer);
    const magic = String.fromCharCode(bytes[1], bytes[2], bytes[3], bytes[4], bytes[5]);
    if (bytes[0] !== 0x93 || magic !== "NUMPY") throw new Error("Not a .npy file");
    const view = new DataView(buffer);
    const major = bytes[6];
    const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
    const headerStart = major === 1 ? 10 : 12;
    let header = "";
    for (let i = 0; i < headerLen; i++) header += String.fromCharCode(bytes[headerStart + i]);

    let descr = /'descr':\s*'([^']+)'/.exec(header)[1];
    if (descr[0] === "=") descr = "<" + descr.slice(1);
    if (/'fortran_order':\s*True/.test(header)) throw new Error("Fortran-ordered .npy is not supported");
    const shape = /'shape':\s*\(([^)]*)\)/.exec(header)[1]
      .split(",").map((s) => s.trim()).filter((s) => s.length).map(Number);
    const Ctor = NPY_TYPES[descr];
    if (!Ctor) throw new Error("Unsupported .npy dtype " + descr + " (only little-endian numeric types)");

    const count = shape.reduce((a, b) => a * b, 1);
    const start = headerStart + headerLen;
    // slice() copies the data to a new, correctly aligned buffer
    let data = new Ctor(buffer.slice(start, start + count * Ctor.BYTES_PER_ELEMENT));
    if (data instanceof BigInt64Array || data instanceof BigUint64Array) data = Float64Array.from(data, Number);
    return { data, shape, dtype: descr };
  }

  function writeNpy(data, shape) {
    const descr = Object.keys(NPY_TYPES).find((k) => NPY_TYPES[k] === data.constructor && k !== "|b1");
    const shapeStr = shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(", ")})`;
    let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': ${shapeStr}, }`;
    // Header is padded with spaces so that the data starts at a multiple of 64 bytes
    const total = 10 + header.length + 1;
    header += " ".repeat((64 - (total % 64)) % 64) + "\n";
    const out = new Uint8Array(10 + header.length + data.byteLength);
    out.set([0x93, 78, 85, 77, 80, 89, 1, 0]);
    out[8] = header.length & 0xff;
    out[9] = header.length >> 8;
    for (let i = 0; i < header.length; i++) out[10 + i] = header.charCodeAt(i);
    out.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), 10 + header.length);
    return out;
  }

  // Volume from a parsed .npy: returns {data, shape: [slices, H, W]}.
  // A (H, W, 3|4) array is treated as an RGB(A) image and converted to grey.
  function toVolume(npy) {
    const s = npy.shape;
    if (s.length === 2) return { data: npy.data, shape: [1, s[0], s[1]] };
    if (s.length === 3 && (s[2] === 3 || s[2] === 4) && s[0] > 4) {
      const n = s[0] * s[1];
      const grey = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const p = i * s[2];
        grey[i] = 0.299 * npy.data[p] + 0.587 * npy.data[p + 1] + 0.114 * npy.data[p + 2];
      }
      return { data: grey, shape: [1, s[0], s[1]] };
    }
    if (s.length === 3) return { data: npy.data, shape: s };
    throw new Error("Expected a 2D image or 3D volume, got shape (" + s.join(", ") + ")");
  }

  function getSlice(volume, index) {
    const [, H, W] = volume.shape;
    const n = H * W;
    const out = new Float64Array(n);
    const offset = index * n;
    for (let i = 0; i < n; i++) out[i] = volume.data[offset + i];
    return out;
  }

  // ---------------------------------------------------------------------
  // Synthetic phantom (bit-identical to python/mia/phantom.py)
  // ---------------------------------------------------------------------

  const LESIONS = [ // centre x, centre y, radius, slice range t_min..t_max
    [0.35, 0.30, 0.09, -0.6, 0.6],
    [-0.30, 0.40, 0.06, -0.3, 1.0],
    [0.0, -0.50, 0.05, -1.0, 1.0],
  ];

  function makePhantom(size = 256, slices = 10) {
    const H = size, W = size, Z = slices;
    const data = new Float32Array(Z * H * W);
    const gt = new Uint8Array(Z * H * W);
    let state = 12345;
    const rand = () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 4294967296;
    };
    const half = (Z - 1) / 2;
    let i = 0;
    for (let z = 0; z < Z; z++) {
      const t = Z > 1 ? (z - half) / half : 0;
      const s = 1 - 0.35 * t * t;
      for (let y = 0; y < H; y++) {
        const v = (y - (H - 1) / 2) / (H / 2);
        for (let x = 0; x < W; x++, i++) {
          const u = (x - (W - 1) / 2) / (W / 2);
          let value = 0, label = 0;
          const ou = u / (0.85 * s), ov = v / (0.95 * s);
          if (ou * ou + ov * ov <= 1) {
            const iu = u / (0.78 * s), iv = v / (0.88 * s);
            if (iu * iu + iv * iv <= 1) {
              label = 2; value = 400;
              if (Math.abs(t) < 0.7) {
                for (const cx of [-0.12, 0.12]) {
                  const du = (u - cx) / (0.07 * s), dv = (v + 0.05) / (0.22 * s);
                  if (du * du + dv * dv <= 1) { label = 3; value = 150; }
                }
              }
              for (const [cx, cy, r, tMin, tMax] of LESIONS) {
                if (t < tMin || t > tMax) continue;
                const du = (u - cx) / r, dv = (v - cy) / r;
                if (du * du + dv * dv <= 1) { label = 4; value = 650; }
              }
            } else {
              label = 1; value = 900;
            }
            value = value * (1 + 0.15 * u); // smooth bias field, as in real MR
          }
          const noise = (rand() + rand() - 1) * 40;
          data[i] = value + noise;
          gt[i] = label;
        }
      }
    }
    return { volume: { data, shape: [Z, H, W] }, groundTruth: { data: gt, shape: [Z, H, W] } };
  }

  // ---------------------------------------------------------------------
  // Pre-processing
  // ---------------------------------------------------------------------

  // Same formula as numpy.percentile (method="linear"), including its
  // symmetric lerp, so both languages pick exactly the same value.
  function percentileSorted(sorted, p) {
    const n = sorted.length;
    const index = (p / 100) * (n - 1);
    const lo = Math.floor(index);
    const hi = Math.min(lo + 1, n - 1);
    const t = index - lo;
    const a = sorted[lo], b = sorted[hi];
    const diff = b - a;
    return t >= 0.5 ? b - diff * (1 - t) : a + diff * t;
  }

  function normalize(img, pLow, pHigh) {
    const sorted = Float64Array.from(img).sort();
    const lo = percentileSorted(sorted, pLow);
    const hi = percentileSorted(sorted, pHigh);
    const out = new Float64Array(img.length);
    if (hi <= lo) return { image: out, lo, hi };
    const range = hi - lo;
    for (let i = 0; i < img.length; i++) {
      const v = (img[i] - lo) / range;
      out[i] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
    return { image: out, lo, hi };
  }

  function gaussianKernel(sigma) {
    const r = Math.floor(3 * sigma + 0.5);
    const k = new Float64Array(2 * r + 1);
    let sum = 0;
    for (let i = -r; i <= r; i++) {
      k[i + r] = Math.exp((-0.5 / (sigma * sigma)) * (i * i));
      sum += k[i + r];
    }
    for (let i = 0; i < k.length; i++) k[i] /= sum;
    return k;
  }

  // 1D correlation along axis 0 (y) or 1 (x), edge mode "nearest"
  function correlateAxis(src, H, W, weights, axis) {
    const r = (weights.length - 1) >> 1;
    const out = new Float64Array(H * W);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let s = 0;
        for (let j = -r; j <= r; j++) {
          let yy = y, xx = x;
          if (axis === 0) { yy = y + j; if (yy < 0) yy = 0; else if (yy >= H) yy = H - 1; }
          else { xx = x + j; if (xx < 0) xx = 0; else if (xx >= W) xx = W - 1; }
          s += weights[j + r] * src[yy * W + xx];
        }
        out[y * W + x] = s;
      }
    }
    return out;
  }

  function gaussianBlur(img, H, W, sigma) {
    if (sigma <= 0) return Float64Array.from(img);
    const k = gaussianKernel(sigma);
    return correlateAxis(correlateAxis(img, H, W, k, 0), H, W, k, 1);
  }

  // Sobel gradient magnitude (same kernels and order as scipy.ndimage.sobel)
  function sobelMagnitude(img, H, W) {
    const d = [-1, 0, 1], s = [1, 2, 1];
    const gy = correlateAxis(correlateAxis(img, H, W, d, 0), H, W, s, 1);
    const gx = correlateAxis(correlateAxis(img, H, W, d, 1), H, W, s, 0);
    const out = new Float64Array(H * W);
    for (let i = 0; i < out.length; i++) out[i] = Math.sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
    return out;
  }

  // Standard deviation in a 3x3 window
  function localStd3(img, H, W) {
    const box = [1 / 3, 1 / 3, 1 / 3];
    const sq = new Float64Array(img.length);
    for (let i = 0; i < img.length; i++) sq[i] = img[i] * img[i];
    const mean = correlateAxis(correlateAxis(img, H, W, box, 0), H, W, box, 1);
    const mean2 = correlateAxis(correlateAxis(sq, H, W, box, 0), H, W, box, 1);
    const out = new Float64Array(img.length);
    for (let i = 0; i < img.length; i++) out[i] = Math.sqrt(Math.max(mean2[i] - mean[i] * mean[i], 0));
    return out;
  }

  function minMaxScale(img) {
    let lo = Infinity, hi = -Infinity;
    for (const v of img) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const out = new Float64Array(img.length);
    if (hi > lo) for (let i = 0; i < img.length; i++) out[i] = (img[i] - lo) / (hi - lo);
    return out;
  }

  // ---------------------------------------------------------------------
  // PCA
  // ---------------------------------------------------------------------

  const FEATURE_NAMES = ["intensity (smoothed)", "Gaussian σ=2", "Gaussian σ=4", "gradient |Sobel|", "local std 3×3"];

  function buildFeatures(norm, smooth, H, W) {
    return [smooth, gaussianBlur(norm, H, W, 2), gaussianBlur(norm, H, W, 4),
      sobelMagnitude(smooth, H, W), localStd3(smooth, H, W)];
  }

  function standardize(cols) {
    return cols.map((c) => {
      const n = c.length;
      let mean = 0;
      for (let i = 0; i < n; i++) mean += c[i];
      mean /= n;
      let v = 0;
      for (let i = 0; i < n; i++) v += (c[i] - mean) * (c[i] - mean);
      let std = Math.sqrt(v / n);
      if (std === 0) std = 1;
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) out[i] = (c[i] - mean) / std;
      return out;
    });
  }

  // Eigen-decomposition of a small symmetric matrix (cyclic Jacobi rotations)
  function jacobiEigen(matrix) {
    const d = matrix.length;
    const a = matrix.map((row) => row.slice());
    const v = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0;
      for (let p = 0; p < d; p++) for (let q = p + 1; q < d; q++) off += a[p][q] * a[p][q];
      if (off < 1e-22) break;
      for (let p = 0; p < d; p++) {
        for (let q = p + 1; q < d; q++) {
          if (Math.abs(a[p][q]) < 1e-300) continue;
          const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
          const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let k = 0; k < d; k++) { // rotate columns p, q
            const akp = a[k][p], akq = a[k][q];
            a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
          }
          for (let k = 0; k < d; k++) { // rotate rows p, q
            const apk = a[p][k], aqk = a[q][k];
            a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
          }
          for (let k = 0; k < d; k++) {
            const vkp = v[k][p], vkq = v[k][q];
            v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq;
          }
        }
      }
    }
    const values = a.map((row, i) => row[i]);
    const vectors = values.map((_, j) => v.map((row) => row[j])); // column j = eigenvector j
    return { values, vectors };
  }

  function pca(cols, nComponents) {
    const z = standardize(cols);
    const d = z.length, n = z[0].length;
    const cov = Array.from({ length: d }, () => new Array(d).fill(0));
    for (let a = 0; a < d; a++) {
      for (let b = a; b < d; b++) {
        let s = 0;
        const za = z[a], zb = z[b];
        for (let i = 0; i < n; i++) s += za[i] * zb[i];
        cov[a][b] = cov[b][a] = s / (n - 1);
      }
    }
    const { values, vectors } = jacobiEigen(cov);
    const order = values.map((_, i) => i).sort((i, j) => values[j] - values[i]);
    const total = values.reduce((a, b) => a + Math.max(b, 0), 0);
    const components = [], explained = [], scores = [];
    for (let c = 0; c < nComponents; c++) {
      const vec = vectors[order[c]].slice();
      // Eigenvectors are only defined up to sign: make the largest loading positive
      let big = 0;
      for (let j = 1; j < d; j++) if (Math.abs(vec[j]) > Math.abs(vec[big])) big = j;
      if (vec[big] < 0) for (let j = 0; j < d; j++) vec[j] = -vec[j];
      const score = new Float64Array(n);
      for (let j = 0; j < d; j++) {
        const w = vec[j], zj = z[j];
        for (let i = 0; i < n; i++) score[i] += w * zj[i];
      }
      components.push(vec);
      explained.push(Math.max(values[order[c]], 0) / total);
      scores.push(score);
    }
    return { scores, components, explained };
  }

  // ---------------------------------------------------------------------
  // Clustering: K-Means with deterministic quantile initialisation
  // ---------------------------------------------------------------------

  function kmeans(cols, k, maxIter, tol) {
    const d = cols.length, n = cols[0].length;
    const order = Array.from({ length: n }, (_, i) => i);
    const f0 = cols[0];
    order.sort((a, b) => f0[a] - f0[b] || a - b); // stable
    let centers = [];
    for (let c = 0; c < k; c++) {
      const idx = order[Math.floor(((c + 0.5) * n) / k)];
      centers.push(cols.map((col) => col[idx]));
    }
    const labels = new Int32Array(n);
    const assign = () => {
      for (let i = 0; i < n; i++) {
        let best = 0, bestD = Infinity;
        for (let c = 0; c < k; c++) {
          let dist = 0;
          for (let j = 0; j < d; j++) { const diff = cols[j][i] - centers[c][j]; dist += diff * diff; }
          if (dist < bestD) { bestD = dist; best = c; }
        }
        labels[i] = best;
      }
    };
    let iterations = 0;
    for (let it = 0; it < maxIter; it++) {
      iterations = it + 1;
      assign();
      const sums = Array.from({ length: k }, () => new Float64Array(d));
      const counts = new Float64Array(k);
      for (let i = 0; i < n; i++) {
        const c = labels[i];
        counts[c]++;
        for (let j = 0; j < d; j++) sums[c][j] += cols[j][i];
      }
      let shift = 0;
      const next = centers.map((old, c) => {
        if (counts[c] === 0) return old; // empty cluster keeps its centre
        const nc = Array.from(sums[c], (s) => s / counts[c]);
        let s2 = 0;
        for (let j = 0; j < d; j++) s2 += (nc[j] - old[j]) * (nc[j] - old[j]);
        if (s2 > shift) shift = s2;
        return nc;
      });
      centers = next;
      if (shift <= tol) break;
    }
    assign();
    // Relabel so that label 0 is the cluster with the lowest feature-0 centre
    const rank = centers.map((_, c) => c).sort((a, b) => centers[a][0] - centers[b][0] || a - b);
    const newLabel = new Int32Array(k);
    rank.forEach((c, r) => { newLabel[c] = r; });
    for (let i = 0; i < n; i++) labels[i] = newLabel[labels[i]];
    return { labels, centers: rank.map((c) => centers[c]), iterations };
  }

  // ---------------------------------------------------------------------
  // Segmentation
  // ---------------------------------------------------------------------

  function otsuThreshold(img) {
    const hist = new Float64Array(256);
    for (const v of img) hist[Math.min(Math.floor(v * 256), 255)]++;
    let total = 0, muT = 0;
    for (let i = 0; i < 256; i++) { total += hist[i]; muT += i * hist[i]; }
    let w0 = 0, mu = 0, best = 0, bestT = 0;
    for (let t = 0; t < 255; t++) {
      w0 += hist[t];
      mu += t * hist[t];
      const w1 = total - w0;
      if (w0 === 0 || w1 === 0) continue;
      const num = muT * w0 - mu * total;
      const sb = (num * num) / (w0 * w1);
      if (sb > best) { best = sb; bestT = t; }
    }
    return (bestT + 1) / 256;
  }

  function threshold(img, T) {
    const out = new Int32Array(img.length);
    for (let i = 0; i < img.length; i++) out[i] = img[i] >= T ? 1 : 0;
    return out;
  }

  function regionGrow(img, H, W, seedY, seedX, tol) {
    const out = new Int32Array(H * W);
    const seed = seedY * W + seedX;
    const ref = img[seed];
    const stack = [seed];
    out[seed] = 1;
    while (stack.length) {
      const p = stack.pop();
      const y = (p / W) | 0, x = p - y * W;
      const nbs = [y > 0 ? p - W : -1, y < H - 1 ? p + W : -1, x > 0 ? p - 1 : -1, x < W - 1 ? p + 1 : -1];
      for (const q of nbs) {
        if (q >= 0 && out[q] === 0 && Math.abs(img[q] - ref) <= tol) { out[q] = 1; stack.push(q); }
      }
    }
    return out;
  }

  // 4-connected component labelling, labels 1..count in raster order
  function connectedComponents(mask, H, W) {
    const labels = new Int32Array(H * W);
    const sizes = [0];
    let count = 0;
    for (let start = 0; start < mask.length; start++) {
      if (!mask[start] || labels[start]) continue;
      count++;
      let size = 0;
      const stack = [start];
      labels[start] = count;
      while (stack.length) {
        const p = stack.pop();
        size++;
        const y = (p / W) | 0, x = p - y * W;
        const nbs = [y > 0 ? p - W : -1, y < H - 1 ? p + W : -1, x > 0 ? p - 1 : -1, x < W - 1 ? p + 1 : -1];
        for (const q of nbs) if (q >= 0 && mask[q] && !labels[q]) { labels[q] = count; stack.push(q); }
      }
      sizes.push(size);
    }
    return { labels, count, sizes };
  }

  function watershedMarkers(img, H, W, low, high, minSize) {
    const markers = new Int32Array(H * W);
    const fg = new Uint8Array(H * W);
    for (let i = 0; i < img.length; i++) {
      if (img[i] < low) markers[i] = 1;
      else if (img[i] > high) fg[i] = 1;
    }
    const cc = connectedComponents(fg, H, W);
    const newId = new Int32Array(cc.count + 1);
    let next = 2;
    for (let c = 1; c <= cc.count; c++) if (cc.sizes[c] >= minSize) newId[c] = next++;
    for (let i = 0; i < img.length; i++) if (cc.labels[i] && newId[cc.labels[i]]) markers[i] = newId[cc.labels[i]];
    return { markers, objects: next - 2 };
  }

  // Binary min-heap keyed on (elevation, age)
  class Heap {
    constructor() { this.elev = []; this.age = []; this.idx = []; }
    get size() { return this.idx.length; }
    less(i, j) { return this.elev[i] < this.elev[j] || (this.elev[i] === this.elev[j] && this.age[i] < this.age[j]); }
    swap(i, j) {
      for (const a of [this.elev, this.age, this.idx]) { const t = a[i]; a[i] = a[j]; a[j] = t; }
    }
    push(e, a, p) {
      this.elev.push(e); this.age.push(a); this.idx.push(p);
      let i = this.idx.length - 1;
      while (i > 0) { const parent = (i - 1) >> 1; if (!this.less(i, parent)) break; this.swap(i, parent); i = parent; }
    }
    pop() {
      const top = this.idx[0];
      const last = this.idx.length - 1;
      this.swap(0, last);
      this.elev.pop(); this.age.pop(); this.idx.pop();
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < last && this.less(l, m)) m = l;
        if (r < last && this.less(r, m)) m = r;
        if (m === i) break;
        this.swap(i, m); i = m;
      }
      return top;
    }
  }

  function watershed(elevation, markers, H, W) {
    const labels = Int32Array.from(markers);
    const heap = new Heap();
    let age = 0;
    for (let p = 0; p < labels.length; p++) if (labels[p] > 0) heap.push(elevation[p], age++, p);
    while (heap.size) {
      const p = heap.pop();
      const y = (p / W) | 0, x = p - y * W;
      const nbs = [y > 0 ? p - W : -1, y < H - 1 ? p + W : -1, x > 0 ? p - 1 : -1, x < W - 1 ? p + 1 : -1];
      for (const q of nbs) {
        if (q >= 0 && labels[q] === 0) { labels[q] = labels[p]; heap.push(elevation[q], age++, q); }
      }
    }
    for (let i = 0; i < labels.length; i++) labels[i] = Math.max(labels[i] - 1, 0);
    return labels;
  }

  // ---------------------------------------------------------------------
  // Pipeline: config (shared JSON) -> labels + timings
  // ---------------------------------------------------------------------

  function runPipeline(img, H, W, config) {
    const timings = {};
    const info = {};
    let t0 = now();
    const start = t0;
    const lap = (name) => { const t = now(); timings[name] = t - t0; t0 = t; };

    const nrm = normalize(img, config.normalize.p_low, config.normalize.p_high);
    info.normalize_range = [nrm.lo, nrm.hi];
    lap("normalize");

    const smooth = gaussianBlur(nrm.image, H, W, config.smooth_sigma);
    lap("smooth");

    let work = smooth, pcaResult = null, pcaImages = [];
    if (config.pca.enabled) {
      const feats = buildFeatures(nrm.image, smooth, H, W);
      pcaResult = pca(feats, config.pca.n_components);
      pcaImages = pcaResult.scores.map(minMaxScale);
      work = pcaImages[0];
      info.explained_variance = pcaResult.explained;
    }
    lap("pca");

    let labels;
    const m = config.method;
    if (m === "kmeans") {
      const cols = pcaResult ? pcaResult.scores : [work];
      const km = kmeans(cols, config.kmeans.k, config.kmeans.max_iter, config.kmeans.tol);
      labels = km.labels;
      info.iterations = km.iterations;
    } else if (m === "threshold") {
      const T = config.threshold.mode === "otsu" ? otsuThreshold(work) : config.threshold.value;
      labels = threshold(work, T);
      info.threshold = T;
    } else if (m === "region_growing") {
      const g = config.region_growing;
      const sy = Math.min(Math.max(g.seed_y, 0), H - 1), sx = Math.min(Math.max(g.seed_x, 0), W - 1);
      labels = regionGrow(work, H, W, sy, sx, g.tolerance);
      info.seed_value = work[sy * W + sx];
    } else if (m === "watershed") {
      const w = config.watershed;
      const elevation = sobelMagnitude(work, H, W);
      const mk = watershedMarkers(work, H, W, w.low, w.high, w.min_marker_size);
      labels = watershed(elevation, mk.markers, H, W);
      info.object_markers = mk.objects;
    } else {
      throw new Error("Unknown method " + m);
    }
    lap("method");
    timings.total = now() - start;
    return { labels, work, pcaImages, pca: pcaResult, timings, info };
  }

  // ---------------------------------------------------------------------
  // Comparison metrics
  // ---------------------------------------------------------------------

  function contingency(a, b) {
    let ka = 0, kb = 0;
    for (let i = 0; i < a.length; i++) { if (a[i] > ka) ka = a[i]; if (b[i] > kb) kb = b[i]; }
    ka++; kb++;
    const table = Array.from({ length: ka }, () => new Float64Array(kb));
    for (let i = 0; i < a.length; i++) table[a[i]][b[i]]++;
    return table;
  }

  function adjustedRandIndex(a, b) {
    const table = contingency(a, b);
    const comb2 = (x) => (x * (x - 1)) / 2;
    const n = a.length;
    let sumIJ = 0;
    const rows = table.map((r) => r.reduce((s, v) => s + v, 0));
    const cols = new Float64Array(table[0].length);
    for (const r of table) r.forEach((v, j) => { sumIJ += comb2(v); cols[j] += v; });
    const sumA = rows.reduce((s, v) => s + comb2(v), 0);
    const sumB = cols.reduce((s, v) => s + comb2(v), 0);
    const expected = (sumA * sumB) / comb2(n);
    const maxIndex = (sumA + sumB) / 2;
    if (maxIndex === expected) return 1;
    return (sumIJ - expected) / (maxIndex - expected);
  }

  // Map every label of b to a label of a: greedy one-to-one matching by
  // largest overlap. Unmatched labels of b become -1.
  function matchLabels(a, b) {
    const table = contingency(a, b);
    const pairs = [];
    table.forEach((r, i) => r.forEach((v, j) => { if (v > 0) pairs.push([v, i, j]); }));
    pairs.sort((p, q) => q[0] - p[0] || p[1] - q[1] || p[2] - q[2]);
    const usedA = new Set();
    const mapping = new Int32Array(table[0].length).fill(-1);
    for (const [, i, j] of pairs) {
      if (usedA.has(i) || mapping[j] !== -1) continue;
      usedA.add(i);
      mapping[j] = i;
    }
    return Int32Array.from(b, (v) => mapping[v]);
  }

  function matchedAgreement(a, b) {
    const mapped = matchLabels(a, b);
    let same = 0;
    for (let i = 0; i < a.length; i++) if (mapped[i] === a[i]) same++;
    return same / a.length;
  }

  function labelStats(labels) {
    let max = 0, fg = 0;
    for (const v of labels) { if (v > max) max = v; if (v > 0) fg++; }
    return { nLabels: max + 1, foreground: fg / labels.length };
  }

  return {
    parseNpy, writeNpy, toVolume, getSlice, makePhantom,
    percentileSorted, normalize, gaussianBlur, sobelMagnitude, localStd3, minMaxScale,
    FEATURE_NAMES, buildFeatures, pca, kmeans, otsuThreshold, threshold, regionGrow,
    connectedComponents, watershedMarkers, watershed, runPipeline,
    adjustedRandIndex, matchLabels, matchedAgreement, labelStats,
  };
});
