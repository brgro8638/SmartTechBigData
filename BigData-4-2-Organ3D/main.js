import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import GUI from 'lil-gui';
import { SoftBody } from './pbd.js';
 
// ---------------------------------------------------------------- parameters
const params = {
  // tool
  tool: 'Grab',
  toolRadius: 0.4,
  grabStiffness: 0.5,
  pokeSpeed: 0.8, // how fast the probe sinks in (units per second)
  pokeDepth: 0.35, // how far it can sink in
  // physics
  gravity: 0, // zero by default: the model is suspended, so world gravity does not act on it
  suspension: 150, // soft elastic hold on the rest pose that keeps it in place (0 = free floating)
  substeps: 16,
  damping: 25, // velocity damping per second: high enough that the tissue settles without wobbling
  pressure: 1.0,
  stretchStiffness: 0.63,
  bendStiffness: 0.55,
  volumeStiffness: 0.5,
  friction: 0.4,
  // mesh
  segments: 24,
  // view
  wireframe: false,
  paused: false,
};
 
// Stiffness slider (0..1) -> XPBD compliance (smaller = stiffer). Defaults are guesses; tune them.
const toCompliance = (s, scale) => scale * Math.pow(10, -6 * s);
function solverParams() {
  return {
    gravity: params.gravity,
    tetherStiffness: params.suspension,
    substeps: params.substeps,
    damping: params.damping,
    pressure: params.pressure,
    floorY: 0,
    friction: params.friction,
    stretchCompliance: toCompliance(params.stretchStiffness, 1e-2),
    bendCompliance: toCompliance(params.bendStiffness, 1e-2),
    volumeCompliance: toCompliance(params.volumeStiffness, 1e-4),
    grabCompliance: toCompliance(params.grabStiffness, 1e-3),
  };
}
 
// ---------------------------------------------------------------- scene
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
const canvas = renderer.domElement;
 
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1b1e24);
 
const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(5, 3.2, 9);
 
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 2.5, 0);
controls.enableDamping = true;
controls.update();
 
scene.add(new THREE.HemisphereLight(0xffffff, 0x333344, 1.2));
const sun = new THREE.DirectionalLight(0xffffff, 2.0);
sun.position.set(3, 6, 4);
scene.add(sun);
scene.add(new THREE.GridHelper(10, 20, 0x555a66, 0x2e323b));
 
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
 
// ---------------------------------------------------------------- body
// Procedural gallbladder: one closed surface swept along a curved centre line.
//   fundus (rounded tip) -> body -> Hartmann's pouch -> infundibulum -> cystic duct (closed end)
// It stands upright (fundus at the bottom, duct on top), with a slightly flattened cross-section
// and a neck that curls sideways. Ring spacing follows the
// ring circumference, so the triangles come out roughly even, which keeps PBD well behaved.
// params.segments is now the number of vertices around each ring (24 is a good default).
 
const TISSUE = new THREE.Color(0.40, 0.58, 0.30); // bile-filled wall
const DUCT = new THREE.Color(0.84, 0.70, 0.55); // paler cystic duct
const PINNED = new THREE.Color(0.2, 0.4, 1.0);
 
const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.35,
    metalness: 0,
    side: THREE.DoubleSide,
});
 
let mesh = null;
let body = null;
let baseColors = null; // un-pinned vertex colours, so unpinning restores the gradient
 
// --- generator start (pure maths, no three.js)
function buildGallbladder(S) {
    const LEN = 4.0; // fundus tip -> end of the duct, along the axis
    const LIFT = 0.8; // how far the neck curls up (+y)
    const FLAT = 0.8; // flattens the cross-section into an oval (narrower in the plane of the neck's curl)
    const HOVER = 2.5; // height of the model's centre above the grid
    const POUCH = 0.1; // extra bulge of Hartmann's pouch on the outer (lower) side
    const TWO_PI = Math.PI * 2;
  
    // radius control points (u = 0..1 along the axis); both ends are closed by rounded caps
    const PTS = [
      [0.16, 0.6], [0.3, 0.66], [0.48, 0.58], [0.62, 0.46],
      [0.72, 0.38], [0.8, 0.25], [0.89, 0.15], [0.96, 0.13],
    ];
    const U0 = PTS[0][0];
    const U1 = PTS[PTS.length - 1][0];
    const tang = PTS.map((p, k) =>
      k === 0 || k === PTS.length - 1 ? 0 : (PTS[k + 1][1] - PTS[k - 1][1]) / (PTS[k + 1][0] - PTS[k - 1][0])
    );
 
  function radius(u) {
      if (u <= 0 || u >= 1) return 0;
      if (u < U0) {
        const t = (U0 - u) / U0;
        return PTS[0][1] * Math.sqrt(Math.max(0, 1 - t * t));
      }
      if (u > U1) {
        const t = (u - U1) / (1 - U1);
        return PTS[PTS.length - 1][1] * Math.sqrt(Math.max(0, 1 - t * t));
      }
      let k = 0;
      while (k < PTS.length - 2 && u > PTS[k + 1][0]) k++;
      const [ua, ra] = PTS[k];
      const [ub, rb] = PTS[k + 1];
      const h = ub - ua;
      const t = (u - ua) / h;
      const t2 = t * t;
      const t3 = t2 * t;
      return (2 * t3 - 3 * t2 + 1) * ra + (t3 - 2 * t2 + t) * h * tang[k] +
        (-2 * t3 + 3 * t2) * rb + (t3 - t2) * h * tang[k + 1];
    }
  
    // centre line in the xy plane: straight at the fundus, then curling upward towards the neck
    const CURL0 = 0.35;
    const CURLW = 0.65;
    const cx = (u) => LEN * u;
    const cy = (u) => {
      const t = Math.min(1, Math.max(0, (u - CURL0) / CURLW));
      return LIFT * t * t * (3 - 2 * t);
    };
    const dcy = (u) => {
      const t = (u - CURL0) / CURLW;
      return t <= 0 || t >= 1 ? 0 : (LIFT * 6 * t * (1 - t)) / CURLW;
    };
  
    // arc length along the profile, so rings can be spaced evenly on the surface
    const NS = 4000;
    const cum = new Float64Array(NS + 1);
    for (let i = 1; i <= NS; i++) {
      const u = i / NS;
      const p = (i - 1) / NS;
      cum[i] = cum[i - 1] + Math.hypot(cx(u) - cx(p), cy(u) - cy(p), radius(u) - radius(p));
    }
    const total = cum[NS];
    const uAt = (s) => {
      let lo = 0;
      let hi = NS;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (cum[mid] <= s) lo = mid;
        else hi = mid;
      }
      return (lo + (s - cum[lo]) / (cum[hi] - cum[lo] || 1)) / NS;
    };
  
    const minStep = (TWO_PI * 0.13) / S; // stops the thin duct from getting a ring every hair's breadth
    const stepFor = (r) => Math.max((TWO_PI * r) / S, minStep);
  
    const P = [0, cy(0), 0, cx(1), cy(1), 0]; // vertex 0 = fundus pole, vertex 1 = duct pole
    const U = [0, 1];
    let rings = 0;
    for (let s = stepFor(0) * 0.8; s < total - minStep * 0.8; s += stepFor(radius(uAt(s)))) {
      const u = uAt(s);
      const r = radius(u);
      const ty = dcy(u);
      const tl = Math.hypot(LEN, ty);
      const nx = ty / tl; // in-plane normal pointing to the outside of the curl
      const ny = -LEN / tl;
      const bump = POUCH * Math.exp(-(((u - 0.7) / 0.07) ** 2));
      for (let j = 0; j < S; j++) {
        const th = (TWO_PI * j) / S;
        const c = Math.cos(th);
        const rr = r + bump * 0.5 * (1 + c);
        P.push(cx(u) + rr * c * FLAT * nx, cy(u) + rr * c * FLAT * ny, rr * Math.sin(th));
        U.push(u);
      }
      rings++;
    }
  
    const ri = (i, j) => 2 + i * S + (j % S);
    const idx = [];
    for (let j = 0; j < S; j++) idx.push(0, ri(0, j + 1), ri(0, j));
    for (let i = 0; i < rings - 1; i++) {
      for (let j = 0; j < S; j++) {
        const a = ri(i, j), b = ri(i, j + 1), c = ri(i + 1, j + 1), d = ri(i + 1, j);
        idx.push(a, b, c, a, c, d);
      }
    }
    for (let j = 0; j < S; j++) idx.push(1, ri(rings - 1, j), ri(rings - 1, j + 1));
  
    // make sure the winding points outwards (signed volume must be positive)
    let V = 0;
    for (let t = 0; t < idx.length; t += 3) {
      const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
      V += (P[a] * (P[b + 1] * P[c + 2] - P[b + 2] * P[c + 1]) +
        P[a + 1] * (P[b + 2] * P[c] - P[b] * P[c + 2]) +
        P[a + 2] * (P[b] * P[c + 1] - P[b + 1] * P[c])) / 6;
    }
    if (V < 0) {
      for (let t = 0; t < idx.length; t += 3) {
        const tmp = idx[t + 1];
        idx[t + 1] = idx[t + 2];
        idx[t + 2] = tmp;
      }
    }
  
    // stand it upright: long axis vertical, fundus at the bottom, cystic duct on top (90 degrees about z)
    for (let i = 0; i < P.length; i += 3) {
      const x = P[i];
      P[i] = -P[i + 1];
      P[i + 1] = x;
    }
  
    // centre it on the origin in x/z and at height HOVER, so it floats in space
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < P.length; i += 3) {
      minX = Math.min(minX, P[i]); maxX = Math.max(maxX, P[i]);
      minY = Math.min(minY, P[i + 1]); maxY = Math.max(maxY, P[i + 1]);
      minZ = Math.min(minZ, P[i + 2]); maxZ = Math.max(maxZ, P[i + 2]);
    }
    const ox = -(minX + maxX) / 2;
    const oy = HOVER - (minY + maxY) / 2;
    const oz = -(minZ + maxZ) / 2;
    for (let i = 0; i < P.length; i += 3) {
        P[i] += ox;
        P[i + 1] += oy;
        P[i + 2] += oz;
    }
  
    return { positions: Float32Array.from(P), indices: Uint32Array.from(idx), u: Float32Array.from(U) };
}
// --- generator end
 
function createGallbladderGeometry(segments) {
    const g = buildGallbladder(Math.max(6, Math.round(segments)));
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(g.positions, 3));
    geo.setIndex(new THREE.BufferAttribute(g.indices, 1));
  
    // wall colour fades into the pale duct towards the neck
    const colors = new Float32Array(g.u.length * 3);
    const c = new THREE.Color();
    for (let i = 0; i < g.u.length; i++) {
        const t = Math.min(1, Math.max(0, (g.u[i] - 0.72) / 0.18));
        c.copy(TISSUE).lerp(DUCT, t * t * (3 - 2 * t)).toArray(colors, i * 3);
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    return geo;
}
 
function build() {
    endInteraction();
    if (mesh) {
        scene.remove(mesh);
        mesh.geometry.dispose();
    }
    const geo = createGallbladderGeometry(params.segments);
    baseColors = geo.attributes.color.array.slice();
    // The solver works directly on the render buffer, so no copying per frame.
    body = new SoftBody(geo.attributes.position.array, geo.index.array);
    refMesh.geometry.dispose();
    refMesh.geometry = new THREE.BufferGeometry();
    refMesh.geometry.setAttribute('position', new THREE.BufferAttribute(geo.attributes.position.array.slice(), 3));
    refMesh.geometry.setIndex(geo.index);
    mesh = new THREE.Mesh(geo, material);
    mesh.frustumCulled = false;
    scene.add(mesh);
    stats.vertices = body.n;
    statsCtrl?.updateDisplay();
}
 
function paintVertex(i, pinned) {
  const col = mesh.geometry.attributes.color;
  if (pinned) PINNED.toArray(col.array, i * 3);
  else col.array.set(baseColors.subarray(i * 3, i * 3 + 3), i * 3);
  col.needsUpdate = true;
}
 
// ---------------------------------------------------------------- GUI
const stats = { vertices: 0 };
const gui = new GUI({ title: 'Soft body' });
 
const fTool = gui.addFolder('Tool');
fTool.add(params, 'tool', ['Grab', 'Poke', 'Pin']).name('tool');
fTool.add(params, 'toolRadius', 0.25, 1.2, 0.01).name('radius');
fTool.add(params, 'grabStiffness', 0, 1, 0.01).name('grab stiffness');
fTool.add(params, 'pokeSpeed', 0.2, 3, 0.05).name('poke speed');
fTool.add(params, 'pokeDepth', 0.1, 0.8, 0.05).name('poke depth');
 
const fPhys = gui.addFolder('Physics');
fPhys.add(params, 'suspension', 0, 1000, 1).name('suspension stiffness');
fPhys.add(params, 'substeps', 1, 40, 1);
fPhys.add(params, 'damping', 0, 60, 0.5);
fPhys.add(params, 'pressure', 0.5, 1.5, 0.01);
fPhys.add(params, 'stretchStiffness', 0, 1, 0.01).name('stretch stiffness');
fPhys.add(params, 'bendStiffness', 0, 1, 0.01).name('bend stiffness');
fPhys.add(params, 'volumeStiffness', 0, 1, 0.01).name('volume stiffness');
fPhys.add(params, 'friction', 0, 1, 0.01);
 
const fMesh = gui.addFolder('Mesh');
fMesh.add(params, 'segments', 8, 40, 1).name('mesh resolution').onFinishChange(build);
const statsCtrl = fMesh.add(stats, 'vertices').name('vertices').disable().listen();
 
const fView = gui.addFolder('View');
fView.add(params, 'wireframe').onChange((v) => (material.wireframe = v));
fView.add(params, 'paused');
 
gui.add({ reset: () => { endInteraction(); body.reset(); resetColors(); } }, 'reset').name('Reset shape');
gui.add({ clear: () => { body.clearPins(); resetColors(); } }, 'clear').name('Clear pins');
 
function resetColors() {
  for (let i = 0; i < body.n; i++) paintVertex(i, body.invMass[i] === 0);
}

fPhys.close()
fMesh.close()

// ---------------------------------------------------------------- interaction
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
const tmp = new THREE.Vector3();
const dragPlane = new THREE.Plane();
 
let drag = null; // { tool, pointerId, hit0?: Vector3 }
 
// Poking follows the cursor over the surface. The tissue is dented under the probe, so the cursor ray
// is cast against a snapshot of the surface taken when the poke starts (an invisible mesh that is never
// added to the scene), not against the live, dented mesh. That would feed the dent back into the probe.
const refMesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
const POKE_FOLLOW = 30; // how quickly the probe chases the cursor (1/s)
const POKE_MAX_SPEED = 8; // speed limit, so a quick flick of the mouse cannot teleport it through the tissue
const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _c = new THREE.Vector3();
const _v0 = new THREE.Vector3();
const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _n = new THREE.Vector3();
 
function snapshotSurface() {
  const g = refMesh.geometry;
  g.attributes.position.array.set(body.pos);
  g.attributes.position.needsUpdate = true;
  g.computeVertexNormals();
  g.computeBoundingBox();
  g.computeBoundingSphere();
}
 
/** Cast the cursor ray at the snapshot; writes the hit point and a smooth surface normal into `out`. */
function sampleSurface(out) {
  raycaster.setFromCamera(ndc, camera);
  const hit = raycaster.intersectObject(refMesh, false)[0];
  if (!hit) return false;
  const pos = refMesh.geometry.attributes.position;
  const nor = refMesh.geometry.attributes.normal;
  const { a, b, c } = hit.face;
  _a.fromBufferAttribute(pos, a);
  _b.fromBufferAttribute(pos, b);
  _c.fromBufferAttribute(pos, c);
  // barycentric weights of the hit point, to blend the vertex normals (face normals would make the probe jitter)
  _v0.subVectors(_b, _a);
  _v1.subVectors(_c, _a);
  _v2.subVectors(hit.point, _a);
  const d00 = _v0.dot(_v0), d01 = _v0.dot(_v1), d11 = _v1.dot(_v1), d20 = _v2.dot(_v0), d21 = _v2.dot(_v1);
  const den = d00 * d11 - d01 * d01;
  let v = 0, w = 0;
  if (Math.abs(den) > 1e-12) {
    v = (d11 * d20 - d01 * d21) / den;
    w = (d00 * d21 - d01 * d20) / den;
  }
  const u = 1 - v - w;
  out.n.set(0, 0, 0);
  out.n.addScaledVector(_n.fromBufferAttribute(nor, a), u);
  out.n.addScaledVector(_n.fromBufferAttribute(nor, b), v);
  out.n.addScaledVector(_n.fromBufferAttribute(nor, c), w);
  if (out.n.lengthSq() < 1e-12) out.n.copy(hit.face.normal);
  out.n.normalize();
  if (out.n.dot(raycaster.ray.direction) > 0) out.n.negate(); // always the side facing the camera
  out.p.copy(hit.point);
  return true;
}
 
function setNDC(e) {
  const r = canvas.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
}
 
function pick() {
  raycaster.setFromCamera(ndc, camera);
  // geometry is deforming, so cached bounds must be refreshed before raycasting
  mesh.geometry.computeBoundingBox();
  mesh.geometry.computeBoundingSphere();
  return raycaster.intersectObject(mesh, false)[0] ?? null;
}
 
function endInteraction() {
  if (drag && body) {
    body.endGrab();
    body.endTool();
  }
  if (drag && canvas.hasPointerCapture?.(drag.pointerId)) canvas.releasePointerCapture(drag.pointerId);
  drag = null;
}
 
// Capture phase + stopImmediatePropagation: when the press lands on the organ it
// belongs to the tool and OrbitControls never sees it. Missing the organ orbits as usual.
canvas.addEventListener(
  'pointerdown',
  (e) => {
    if (e.button !== 0) return; // right-drag pans, wheel zooms (OrbitControls)
    setNDC(e);
    const hit = pick();
    if (!hit) return;
    e.stopImmediatePropagation();
 
    const { idx, w } = body.queryRadius(hit.point.x, hit.point.y, hit.point.z, params.toolRadius);
 
    if (params.tool === 'Pin') {
      // if the closest vertex is already pinned, unpin the whole patch, otherwise pin it
      let nearest = idx[0];
      let best = -1;
      idx.forEach((v, m) => { if (w[m] > best) { best = w[m]; nearest = v; } });
      const pin = nearest === undefined ? false : body.invMass[nearest] !== 0;
      for (const v of idx) {
        body.setPinned(v, pin);
        paintVertex(v, pin);
      }
      return;
    }
 
    canvas.setPointerCapture(e.pointerId);
    drag = { tool: params.tool, pointerId: e.pointerId, hit0: hit.point.clone() };
 
    if (params.tool === 'Grab') {
      // drag plane faces the camera and passes through the grab point
      camera.getWorldDirection(tmp);
      dragPlane.setFromNormalAndCoplanarPoint(tmp, hit.point);
      body.beginGrab(idx, w);
    } else if (params.tool === 'Poke') {
      snapshotSurface();
      drag.target = { p: hit.point.clone(), n: hit.face.normal.clone() }; // where the cursor is on the surface
      if (drag.target.n.dot(raycaster.ray.direction) > 0) drag.target.n.negate();
      sampleSurface(drag.target); // same spot, but with a smooth normal
      drag.r = params.toolRadius;
      drag.depth = 0;
      drag.p = drag.target.p.clone(); // where the probe touches the surface; it chases the target
      drag.n = drag.target.n.clone();
      tmp.copy(drag.n).multiplyScalar(drag.r).add(drag.p); // probe resting on the surface
      body.beginTool(tmp.x, tmp.y, tmp.z, drag.r);
    }
  },
  { capture: true }
);
 
canvas.addEventListener('pointermove', (e) => {
  setNDC(e);
  if (drag?.tool === 'Grab') {
    raycaster.setFromCamera(ndc, camera);
    if (raycaster.ray.intersectPlane(dragPlane, tmp)) {
      body.setGrabDelta(tmp.x - drag.hit0.x, tmp.y - drag.hit0.y, tmp.z - drag.hit0.z);
    }
  } else if (drag?.tool === 'Poke') {
    sampleSurface(drag.target); // cursor off the organ: no hit, so the probe stays where it was
  }
});
canvas.addEventListener('pointerup', endInteraction);
canvas.addEventListener('pointercancel', endInteraction);
 
// While the button is held the probe sinks into the surface, up to pokeDepth, and follows the cursor
// across it, so dragging moves the indent. Releasing removes it and the tissue settles back.
function updatePoke(dt) {
  if (drag?.tool !== 'Poke') return;
  drag.depth = Math.min(params.pokeDepth, drag.depth + params.pokeSpeed * dt);
  const follow = 1 - Math.exp(-dt * POKE_FOLLOW);
  tmp.subVectors(drag.target.p, drag.p);
  const dist = tmp.length();
  if (dist > 1e-9) drag.p.addScaledVector(tmp, Math.min(follow, (POKE_MAX_SPEED * dt) / dist));
  drag.n.lerp(drag.target.n, follow).normalize();
  tmp.copy(drag.n).multiplyScalar(drag.r - drag.depth).add(drag.p);
  body.moveTool(tmp.x, tmp.y, tmp.z);
}
 
// ---------------------------------------------------------------- loop
build();
let last = performance.now();
renderer.setAnimationLoop(() => {
  const now = performance.now();
  const dt = Math.min((now - last) / 1000, 1 / 30);
  last = now;
 
  if (!params.paused) {
    updatePoke(dt);
    body.step(dt, solverParams());
    mesh.geometry.attributes.position.needsUpdate = true;
    mesh.geometry.computeVertexNormals();
  }
 
  controls.update();
  renderer.render(scene, camera);
});
 