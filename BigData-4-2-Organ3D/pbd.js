// Position-based dynamics (XPBD, "small steps" variant) for a closed triangle mesh.
// No three.js dependency: it works on a flat Float32Array of positions and an index array.
//
// Constraints:
//   - stretch: distance constraint on every mesh edge
//   - bend:    distance constraint between the two vertices opposite each shared edge
//   - volume:  one global constraint, V = pressure * V0  (needs a closed, consistently wound mesh)
//   - grab:    soft attachment of a vertex group to a moving target (the "surgical tool")
//   - tool:    hard spherical collider (the poke probe) that pushes tissue out of its way
//   - tether:  soft spring from every vertex to its rest position; this is what suspends the model
//              in place (it replaces gravity-and-floor support). Stiffness 0 = free floating.
//
// Each substep does one constraint pass, so XPBD lambdas never accumulate and are omitted.
 
export class SoftBody {
  /**
   * @param {Float32Array} positions  xyz triplets; used in place, so share it with the render buffer
   * @param {ArrayLike<number>} indices triangle indices (counter-clockwise seen from outside)
   */
  constructor(positions, indices) {
    this.n = positions.length / 3;
    this.pos = positions;
    this.restPos = positions.slice();
    this.prev = positions.slice();
    this.vel = new Float32Array(positions.length);
    this.invMass = new Float32Array(this.n).fill(1);
    this.tris = indices;
    this.grad = new Float32Array(positions.length);
    this.grab = null;
    this.tool = null;
 
    this._buildConstraints();
    this.restVolume = this.volume();
  }
 
  _buildConstraints() {
    const { n, tris } = this;
    const edges = new Map();
    for (let t = 0; t < tris.length; t += 3) {
      for (let e = 0; e < 3; e++) {
        const a = tris[t + e];
        const b = tris[t + ((e + 1) % 3)];
        const c = tris[t + ((e + 2) % 3)];
        const lo = Math.min(a, b);
        const hi = Math.max(a, b);
        const key = lo * n + hi;
        let ed = edges.get(key);
        if (!ed) {
          ed = { a: lo, b: hi, opp: [] };
          edges.set(key, ed);
        }
        ed.opp.push(c);
      }
    }
    const stretch = [];
    const bend = [];
    for (const ed of edges.values()) {
      stretch.push(ed.a, ed.b);
      if (ed.opp.length === 2 && ed.opp[0] !== ed.opp[1]) bend.push(ed.opp[0], ed.opp[1]);
    }
    this.stretchIdx = Uint32Array.from(stretch);
    this.bendIdx = Uint32Array.from(bend);
    this.stretchRest = this._restLengths(this.stretchIdx);
    this.bendRest = this._restLengths(this.bendIdx);
  }
 
  _restLengths(idx) {
    const r = new Float32Array(idx.length / 2);
    const p = this.restPos;
    for (let c = 0; c < r.length; c++) {
      const a = idx[2 * c] * 3;
      const b = idx[2 * c + 1] * 3;
      r[c] = Math.hypot(p[a] - p[b], p[a + 1] - p[b + 1], p[a + 2] - p[b + 2]);
    }
    return r;
  }
 
  /** Signed volume of the closed mesh (divergence theorem). */
  volume() {
    const { pos, tris } = this;
    let V = 0;
    for (let t = 0; t < tris.length; t += 3) {
      const a = tris[t] * 3, b = tris[t + 1] * 3, c = tris[t + 2] * 3;
      V +=
        (pos[a] * (pos[b + 1] * pos[c + 2] - pos[b + 2] * pos[c + 1]) +
          pos[a + 1] * (pos[b + 2] * pos[c] - pos[b] * pos[c + 2]) +
          pos[a + 2] * (pos[b] * pos[c + 1] - pos[b + 1] * pos[c])) / 6;
    }
    return V;
  }
 
  reset() {
    this.pos.set(this.restPos);
    this.prev.set(this.restPos);
    this.vel.fill(0);
    this.grab = null;
    this.tool = null;
  }
 
  setPinned(i, pinned) {
    this.invMass[i] = pinned ? 0 : 1;
  }
 
  clearPins() {
    this.invMass.fill(1);
  }
 
  /** Vertices within radius r of (x,y,z), with a smooth 1 -> 0 falloff weight. */
  queryRadius(x, y, z, r) {
    const { pos, n } = this;
    const idx = [];
    const w = [];
    const r2 = r * r;
    for (let i = 0; i < n; i++) {
      const k = i * 3;
      const dx = pos[k] - x, dy = pos[k + 1] - y, dz = pos[k + 2] - z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 >= r2) continue;
      const t = 1 - Math.sqrt(d2) / r;
      idx.push(i);
      w.push(t * t * (3 - 2 * t));
    }
    return { idx, w };
  }
 
  /** Add velocity along (dx,dy,dz) to vertices near (x,y,z). Used for poking. */
  applyImpulse(x, y, z, dx, dy, dz, radius, amount) {
    const { idx, w } = this.queryRadius(x, y, z, radius);
    const { vel, invMass } = this;
    for (let m = 0; m < idx.length; m++) {
      const i = idx[m];
      if (invMass[i] === 0) continue;
      const s = amount * w[m];
      vel[i * 3] += dx * s;
      vel[i * 3 + 1] += dy * s;
      vel[i * 3 + 2] += dz * s;
    }
  }
 
  /** Start dragging the vertices `idx` (with falloff weights `w`). */
  beginGrab(idx, w) {
    const base = new Float32Array(idx.length * 3);
    for (let m = 0; m < idx.length; m++) {
      base[m * 3] = this.pos[idx[m] * 3];
      base[m * 3 + 1] = this.pos[idx[m] * 3 + 1];
      base[m * 3 + 2] = this.pos[idx[m] * 3 + 2];
    }
    this.grab = { idx, w, base, dx: 0, dy: 0, dz: 0 };
  }
 
  setGrabDelta(dx, dy, dz) {
    if (!this.grab) return;
    this.grab.dx = dx;
    this.grab.dy = dy;
    this.grab.dz = dz;
  }
 
  endGrab() {
    this.grab = null;
  }
 
  /**
   * Spherical probe. It is a hard collider: vertices inside it are pushed out to its surface.
   * Move it with moveTool(); it is swept smoothly across the substeps of each frame, so the tissue
   * is pushed at the probe's real speed instead of being kicked.
   */
  beginTool(x, y, z, r) {
    this.tool = { px: x, py: y, pz: z, x, y, z, r };
  }
 
  moveTool(x, y, z) {
    const t = this.tool;
    if (!t) return;
    t.x = x;
    t.y = y;
    t.z = z;
  }
 
  endTool() {
    this.tool = null;
  }
 
  /**
   * Advance the simulation by dt seconds.
   * params: gravity, substeps, damping, floorY, friction, pressure,
   *         stretchCompliance, bendCompliance, volumeCompliance, grabCompliance, tetherStiffness
   */
  step(dt, p) {
    const substeps = Math.max(1, Math.round(p.substeps));
    const h = dt / substeps;
    const h2 = h * h;
    const aStretch = p.stretchCompliance / h2;
    const aBend = p.bendCompliance / h2;
    const aVolume = p.volumeCompliance / h2;
    const aTether = p.tetherStiffness > 0 ? 1 / p.tetherStiffness / h2 : 0;
    for (let s = 0; s < substeps; s++) {
      this._predict(h, p);
      this._solveDistance(this.stretchIdx, this.stretchRest, aStretch);
      this._solveDistance(this.bendIdx, this.bendRest, aBend);
      this._solveVolume(aVolume, p.pressure);
      this._solveGrab(p.grabCompliance, h2);
      if (aTether > 0) this._solveTether(aTether);
      this._collideTool((s + 1) / substeps);
      this._collideFloor(p);
      this._updateVelocity(h);
    }
    const t = this.tool;
    if (t) {
      t.px = t.x;
      t.py = t.y;
      t.pz = t.z;
    }
  }
 
  _predict(h, p) {
    const { pos, prev, vel, invMass, n } = this;
    const damp = Math.max(0, 1 - p.damping * h);
    const gh = p.gravity * h;
    for (let i = 0; i < n; i++) {
      const k = i * 3;
      prev[k] = pos[k];
      prev[k + 1] = pos[k + 1];
      prev[k + 2] = pos[k + 2];
      if (invMass[i] === 0) continue;
      const vx = vel[k] * damp;
      const vy = (vel[k + 1] + gh) * damp;
      const vz = vel[k + 2] * damp;
      vel[k] = vx;
      vel[k + 1] = vy;
      vel[k + 2] = vz;
      pos[k] += vx * h;
      pos[k + 1] += vy * h;
      pos[k + 2] += vz * h;
    }
  }
 
  _solveDistance(idx, rest, alpha) {
    const { pos, invMass } = this;
    for (let c = 0, m = rest.length; c < m; c++) {
      const i = idx[2 * c], j = idx[2 * c + 1];
      const wi = invMass[i], wj = invMass[j];
      const w = wi + wj;
      if (w === 0) continue;
      const a = i * 3, b = j * 3;
      const dx = pos[a] - pos[b];
      const dy = pos[a + 1] - pos[b + 1];
      const dz = pos[a + 2] - pos[b + 2];
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (len < 1e-9) continue;
      const k = -(len - rest[c]) / ((w + alpha) * len);
      pos[a] += wi * k * dx;
      pos[a + 1] += wi * k * dy;
      pos[a + 2] += wi * k * dz;
      pos[b] -= wj * k * dx;
      pos[b + 1] -= wj * k * dy;
      pos[b + 2] -= wj * k * dz;
    }
  }
 
  _solveVolume(alpha, pressure) {
    const { pos, tris, grad, invMass, n } = this;
    grad.fill(0);
    let V = 0;
    for (let t = 0; t < tris.length; t += 3) {
      const a = tris[t] * 3, b = tris[t + 1] * 3, c = tris[t + 2] * 3;
      const ax = pos[a], ay = pos[a + 1], az = pos[a + 2];
      const bx = pos[b], by = pos[b + 1], bz = pos[b + 2];
      const cx = pos[c], cy = pos[c + 1], cz = pos[c + 2];
      // b x c, c x a, a x b
      const bcx = by * cz - bz * cy, bcy = bz * cx - bx * cz, bcz = bx * cy - by * cx;
      const cax = cy * az - cz * ay, cay = cz * ax - cx * az, caz = cx * ay - cy * ax;
      const abx = ay * bz - az * by, aby = az * bx - ax * bz, abz = ax * by - ay * bx;
      V += (ax * bcx + ay * bcy + az * bcz) / 6;
      grad[a] += bcx / 6; grad[a + 1] += bcy / 6; grad[a + 2] += bcz / 6;
      grad[b] += cax / 6; grad[b + 1] += cay / 6; grad[b + 2] += caz / 6;
      grad[c] += abx / 6; grad[c + 1] += aby / 6; grad[c + 2] += abz / 6;
    }
    const C = V - pressure * this.restVolume;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const w = invMass[i];
      if (w === 0) continue;
      const k = i * 3;
      sum += w * (grad[k] * grad[k] + grad[k + 1] * grad[k + 1] + grad[k + 2] * grad[k + 2]);
    }
    const denom = sum + alpha;
    if (denom < 1e-12) return;
    const dl = -C / denom;
    for (let i = 0; i < n; i++) {
      const w = invMass[i];
      if (w === 0) continue;
      const k = i * 3;
      pos[k] += w * dl * grad[k];
      pos[k + 1] += w * dl * grad[k + 1];
      pos[k + 2] += w * dl * grad[k + 2];
    }
  }
 
  _solveGrab(compliance, h2) {
    const g = this.grab;
    if (!g) return;
    const { pos, invMass } = this;
    for (let m = 0; m < g.idx.length; m++) {
      const i = g.idx[m];
      const wi = invMass[i];
      if (wi === 0) continue;
      const a = i * 3;
      const dx = pos[a] - (g.base[m * 3] + g.dx);
      const dy = pos[a + 1] - (g.base[m * 3 + 1] + g.dy);
      const dz = pos[a + 2] - (g.base[m * 3 + 2] + g.dz);
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (len < 1e-9) continue;
      // vertices near the edge of the tool radius are held more loosely
      const alpha = compliance / Math.max(g.w[m], 0.05) / h2;
      const k = -1 / (wi + alpha); // dl = -C/(w+alpha), n = d/len, C = len
      pos[a] += wi * k * dx;
      pos[a + 1] += wi * k * dy;
      pos[a + 2] += wi * k * dz;
    }
  }
 
  _solveTether(alpha) {
    const { pos, restPos, invMass, n } = this;
    for (let i = 0; i < n; i++) {
      const w = invMass[i];
      if (w === 0) continue;
      const k = i * 3;
      const s = -w / (w + alpha); // pull the vertex back towards where it started
      pos[k] += s * (pos[k] - restPos[k]);
      pos[k + 1] += s * (pos[k + 1] - restPos[k + 1]);
      pos[k + 2] += s * (pos[k + 2] - restPos[k + 2]);
    }
  }
 
  _collideTool(f) {
    const t = this.tool;
    if (!t) return;
    const cx = t.px + (t.x - t.px) * f;
    const cy = t.py + (t.y - t.py) * f;
    const cz = t.pz + (t.z - t.pz) * f;
    const { pos, invMass, n } = this;
    const r = t.r;
    const r2 = r * r;
    for (let i = 0; i < n; i++) {
      if (invMass[i] === 0) continue;
      const k = i * 3;
      const dx = pos[k] - cx;
      const dy = pos[k + 1] - cy;
      const dz = pos[k + 2] - cz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 >= r2) continue;
      const d = Math.sqrt(d2);
      if (d < 1e-9) {
        pos[k + 1] = cy + r;
        continue;
      }
      const sc = r / d;
      pos[k] = cx + dx * sc;
      pos[k + 1] = cy + dy * sc;
      pos[k + 2] = cz + dz * sc;
    }
  }
 
  _collideFloor(p) {
    const { pos, prev, n } = this;
    for (let i = 0; i < n; i++) {
      const k = i * 3;
      if (pos[k + 1] < p.floorY) {
        pos[k + 1] = p.floorY;
        pos[k] += (prev[k] - pos[k]) * p.friction;
        pos[k + 2] += (prev[k + 2] - pos[k + 2]) * p.friction;
      }
    }
  }
 
  _updateVelocity(h) {
    const { pos, prev, vel, invMass, n } = this;
    const inv = 1 / h;
    for (let i = 0; i < n; i++) {
      const k = i * 3;
      if (invMass[i] === 0) {
        vel[k] = vel[k + 1] = vel[k + 2] = 0;
        continue;
      }
      vel[k] = (pos[k] - prev[k]) * inv;
      vel[k + 1] = (pos[k + 1] - prev[k + 1]) * inv;
      vel[k + 2] = (pos[k + 2] - prev[k + 2]) * inv;
    }
  }
}
 