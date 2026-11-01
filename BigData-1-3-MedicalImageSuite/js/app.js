/*
 * Browser GUI for the JavaScript implementation (mia-core.js).
 * Reads the controls into the shared config JSON, runs the pipeline on the
 * selected slice, and draws the results on canvases.
 */
"use strict";

const $ = (id) => document.getElementById(id);

// Same colours as PALETTE in python/app.py
const PALETTE = [
  [0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
  [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128],
];
const METHODS = { kmeans: "K-Means clustering", threshold: "Thresholding", region_growing: "Region growing", watershed: "Watershed" };
const DEFAULT_CONFIG = {
  slice: 5,
  normalize: { p_low: 0.5, p_high: 99.5 },
  smooth_sigma: 1.0,
  pca: { enabled: false, n_components: 2 },
  method: "kmeans",
  kmeans: { k: 4, max_iter: 100, tol: 1e-6 },
  threshold: { mode: "otsu", value: 0.5 },
  region_growing: { seed_x: 128, seed_y: 128, tolerance: 0.12 },
  watershed: { low: 0.15, high: 0.6, min_marker_size: 20 },
};

const state = { volume: null, groundTruth: null, name: "", result: null, config: null };

// ---------------------------------------------------------------------------
// Config <-> controls
// ---------------------------------------------------------------------------

function writeControls(c) {
  // Raise the slider maximum first, otherwise the browser clamps the value
  $("slice").max = Math.max(Number($("slice").max), c.slice);
  $("slice").value = c.slice;
  $("p_low").value = c.normalize.p_low;
  $("p_high").value = c.normalize.p_high;
  $("smooth_sigma").value = c.smooth_sigma;
  $("pca_enabled").checked = c.pca.enabled;
  $("n_components").value = c.pca.n_components;
  $("method").value = c.method;
  $("k").value = c.kmeans.k;
  $("max_iter").value = c.kmeans.max_iter;
  $("th_mode").value = c.threshold.mode;
  $("th_value").value = c.threshold.value;
  $("seed_x").value = c.region_growing.seed_x;
  $("seed_y").value = c.region_growing.seed_y;
  $("tolerance").value = c.region_growing.tolerance;
  $("ws_low").value = c.watershed.low;
  $("ws_high").value = c.watershed.high;
  $("ws_min").value = c.watershed.min_marker_size;
  state.tol = c.kmeans.tol;
}

function readControls() {
  const num = (id) => Number($(id).value);
  return {
    slice: num("slice"),
    normalize: { p_low: num("p_low"), p_high: num("p_high") },
    smooth_sigma: num("smooth_sigma"),
    pca: { enabled: $("pca_enabled").checked, n_components: num("n_components") },
    method: $("method").value,
    kmeans: { k: num("k"), max_iter: num("max_iter"), tol: state.tol },
    threshold: { mode: $("th_mode").value, value: num("th_value") },
    region_growing: { seed_x: num("seed_x"), seed_y: num("seed_y"), tolerance: num("tolerance") },
    watershed: { low: num("ws_low"), high: num("ws_high"), min_marker_size: num("ws_min") },
  };
}

function updateLabels(c) {
  $("out-slice").textContent = c.slice;
  $("out-sigma").textContent = c.smooth_sigma.toFixed(2);
  $("out-ncomp").textContent = c.pca.n_components;
  $("out-k").textContent = c.kmeans.k;
  $("out-thv").textContent = c.threshold.value.toFixed(2);
  $("out-tol").textContent = c.region_growing.tolerance.toFixed(2);
  $("n_components").disabled = !c.pca.enabled;
  $("th_value").disabled = c.threshold.mode === "otsu";
  document.querySelectorAll("fieldset[data-method]").forEach((f) => {
    f.classList.toggle("inactive", f.dataset.method !== c.method);
  });
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

function drawRGB(canvas, W, H, pixel) {
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  const im = ctx.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    const [r, g, b] = pixel(i);
    im.data[4 * i] = r; im.data[4 * i + 1] = g; im.data[4 * i + 2] = b; im.data[4 * i + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}

const grey = (v) => { const g = Math.round(Math.min(Math.max(v, 0), 1) * 255); return [g, g, g]; };

function drawGrey(canvas, img, W, H) { drawRGB(canvas, W, H, (i) => grey(img[i])); }

// Coloured labels on top of the grey image; label 0 is left uncoloured.
// K-Means labels start at 0, so they are shifted by one to colour every cluster.
function drawOverlay(canvas, img, labels, W, H, shift = 0, alpha = 0.45) {
  drawRGB(canvas, W, H, (i) => {
    const g = grey(img[i]);
    const lab = labels[i] + shift;
    if (lab <= 0) return g;
    const c = PALETTE[((lab - 1) % (PALETTE.length - 1)) + 1];
    return g.map((v, k) => Math.round((1 - alpha) * v + alpha * c[k]));
  });
}

function drawSeed(canvas, config) {
  if (config.method !== "region_growing") return;
  const ctx = canvas.getContext("2d");
  const { seed_x: x, seed_y: y } = config.region_growing;
  ctx.fillStyle = "#ffff00";
  ctx.fillRect(x - 3, y, 7, 1);
  ctx.fillRect(x, y - 3, 1, 7);
}

function metric(label, value) {
  return `<div class="metric"><span>${label}</span><b>${value}</b></div>`;
}

function table(el, header, rows) {
  el.innerHTML = "<tr>" + header.map((h) => `<th>${h}</th>`).join("") + "</tr>" +
    rows.map((r) => `<tr${r.cls ? ` class="${r.cls}"` : ""}>` + r.cells.map((c) => `<td>${c}</td>`).join("") + "</tr>").join("");
}

const fmt = (v, d = 2) => (typeof v === "number" ? v.toFixed(d) : v);

// ---------------------------------------------------------------------------
// Running the pipeline
// ---------------------------------------------------------------------------

function currentSlice(config) {
  const [, H, W] = state.volume.shape;
  return { img: MIA.getSlice(state.volume, config.slice), H, W };
}

function groundTruthSlice(index) {
  if (!state.groundTruth) return null;
  const [, H, W] = state.groundTruth.shape;
  return Int32Array.from(state.groundTruth.data.subarray(index * H * W, (index + 1) * H * W));
}

function displayImage(img, config) {
  return MIA.normalize(img, config.normalize.p_low, config.normalize.p_high).image;
}

function run() {
  if (!state.volume) return;
  const config = readControls();
  updateLabels(config);
  state.config = config;
  const { img, H, W } = currentSlice(config);
  const result = MIA.runPipeline(img, H, W, config);
  state.result = result;
  const shown = displayImage(img, config);
  const shift = config.method === "kmeans" ? 1 : 0;

  drawGrey($("cv-input"), shown, W, H);
  drawSeed($("cv-input"), config);
  drawGrey($("cv-work"), result.work, W, H);
  drawSeed($("cv-work"), config);
  drawOverlay($("cv-result"), shown, result.labels, W, H, shift);
  $("cap-input").textContent = `Input – slice ${config.slice} (normalized)`;
  $("cap-work").textContent = `Work image (${config.pca.enabled ? "PC1" : "smoothed"})`;
  $("cap-result").textContent = `${METHODS[config.method]} – JavaScript`;

  const stats = MIA.labelStats(result.labels);
  let html = metric("Total runtime", `${result.timings.total.toFixed(1)} ms`) +
    metric("Labels", stats.nLabels) + metric("Foreground", `${(stats.foreground * 100).toFixed(1)} %`);
  const gt = groundTruthSlice(config.slice);
  if (gt) {
    html += metric("ARI vs ground truth", MIA.adjustedRandIndex(gt, result.labels).toFixed(3));
    html += metric("Matched accuracy vs GT", `${(MIA.matchedAgreement(gt, result.labels) * 100).toFixed(1)} %`);
  }
  $("metrics").innerHTML = html;

  const steps = Object.keys(result.timings);
  table($("tbl-timings"), steps, [{ cells: steps.map((s) => fmt(result.timings[s])) }]);
  $("info").textContent = JSON.stringify(result.info, null, 2);

  $("pca-panel").hidden = !config.pca.enabled;
  if (config.pca.enabled) {
    table($("tbl-pca"), ["", "explained", ...MIA.FEATURE_NAMES],
      result.pca.components.map((comp, i) => ({
        cells: [`PC${i + 1}`, `${(result.pca.explained[i] * 100).toFixed(1)} %`, ...comp.map((v) => fmt(v, 3))],
      })));
    const box = $("pca-images");
    box.innerHTML = "";
    result.pcaImages.forEach((pc, i) => {
      const fig = document.createElement("figure");
      const cv = document.createElement("canvas");
      fig.append(cv);
      fig.insertAdjacentHTML("beforeend", `<figcaption>PC${i + 1}</figcaption>`);
      box.append(fig);
      drawGrey(cv, pc, W, H);
    });
  }
  if (state.reference) compareReference();
  setStatus(`Ran ${METHODS[config.method]} on ${state.name}, slice ${config.slice} (${W}×${H}).`);
}

let pending = null;
function scheduleRun() {
  // Small delay so dragging a slider does not queue dozens of runs
  clearTimeout(pending);
  pending = setTimeout(run, 60);
}

function runAll() {
  if (!state.volume) return;
  const base = readControls();
  const { img, H, W } = currentSlice(base);
  const shown = displayImage(img, base);
  const gt = groundTruthSlice(base.slice);
  const pcaOptions = $("both-pca").checked ? [false, true] : [base.pca.enabled];
  const variants = [];
  for (const usePca of pcaOptions) {
    for (const method of Object.keys(METHODS)) {
      for (const mode of method === "threshold" ? ["otsu", "manual"] : [null]) {
        const c = JSON.parse(JSON.stringify(base));
        c.method = method;
        c.pca.enabled = usePca;
        if (mode) c.threshold.mode = mode;
        variants.push({ label: METHODS[method] + (mode ? ` (${mode})` : "") + (usePca ? " + PCA" : ""), config: c });
      }
    }
  }
  const rows = [];
  const box = $("all-images");
  box.innerHTML = "";
  for (const v of variants) {
    const r = MIA.runPipeline(img, H, W, v.config);
    const s = MIA.labelStats(r.labels);
    const row = { label: v.label, labels: s.nLabels, fg: s.foreground, ms: r.timings.total, method: r.timings.method };
    if (gt) { row.ari = MIA.adjustedRandIndex(gt, r.labels); row.acc = MIA.matchedAgreement(gt, r.labels); }
    rows.push(row);
    const fig = document.createElement("figure");
    const cv = document.createElement("canvas");
    fig.append(cv);
    fig.insertAdjacentHTML("beforeend", `<figcaption>${v.label}</figcaption>`);
    box.append(fig);
    drawOverlay(cv, shown, r.labels, W, H, v.config.method === "kmeans" ? 1 : 0);
  }
  const best = gt ? rows.reduce((a, b) => (b.ari > a.ari ? b : a)) : null;
  const header = ["method", "labels", "foreground %", "runtime ms", "method ms"].concat(gt ? ["ARI vs GT", "accuracy vs GT"] : []);
  table($("tbl-all"), header, rows.map((r) => ({
    cls: r === best ? "best" : "",
    cells: [r.label, r.labels, fmt(r.fg * 100, 1), fmt(r.ms, 1), fmt(r.method, 1)]
      .concat(gt ? [fmt(r.ari, 3), fmt(r.acc * 100, 1) + " %"] : []),
  })));
  $("best").textContent = best ? `Best agreement with ground truth: ${best.label} (ARI ${best.ari.toFixed(3)})` : "";
}

function compareReference() {
  const res = state.result;
  const ref = state.reference;
  const [, H, W] = state.volume.shape;
  if (!res || ref.length !== H * W) {
    $("ref-metrics").innerHTML = metric("Error", "size mismatch");
    return;
  }
  const config = state.config;
  const shown = displayImage(currentSlice(config).img, config);
  const shift = config.method === "kmeans" ? 1 : 0;
  // Python labels are matched to the JS labels first, then compared pixel by pixel
  const mapped = MIA.matchLabels(res.labels, ref);
  const ari = MIA.adjustedRandIndex(res.labels, ref);
  const diff = new Int32Array(H * W);
  let differing = 0;
  for (let i = 0; i < H * W; i++) if (mapped[i] !== res.labels[i]) { diff[i] = 1; differing++; }
  const agreement = 1 - differing / (H * W);

  $("ref-metrics").innerHTML = metric("Matched agreement", `${(agreement * 100).toFixed(3)} %`) +
    metric("ARI", ari.toFixed(5)) + metric("Differing pixels", differing);
  drawOverlay($("cv-ref-js"), shown, res.labels, W, H, shift);
  drawOverlay($("cv-ref-py"), shown, ref, W, H, shift);
  drawOverlay($("cv-ref-diff"), shown, diff, W, H, 0, 0.9);
  $("cap-diff").textContent = `Differences (${differing} px, red)`;
}

// ---------------------------------------------------------------------------
// Loading / saving files
// ---------------------------------------------------------------------------

function setStatus(text) { $("status").textContent = text; }

function setVolume(volume, groundTruth, name) {
  state.volume = volume;
  state.groundTruth = groundTruth;
  state.name = name;
  state.reference = null;
  const [slices, H, W] = volume.shape;
  $("slice").max = slices - 1;
  if (Number($("slice").value) > slices - 1) $("slice").value = Math.floor(slices / 2);
  $("seed_x").max = W - 1;
  $("seed_y").max = H - 1;
  $("dataset-info").textContent = `${name} · ${slices} × ${H} × ${W}` + (groundTruth ? " · with ground truth" : "");
  run();
}

function loadImageFile(file) {
  const url = URL.createObjectURL(file);
  const image = new Image();
  image.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0);
    const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const n = canvas.width * canvas.height;
    const g = new Float64Array(n);
    for (let i = 0; i < n; i++) g[i] = 0.299 * rgba[4 * i] + 0.587 * rgba[4 * i + 1] + 0.114 * rgba[4 * i + 2];
    URL.revokeObjectURL(url);
    setVolume({ data: g, shape: [1, canvas.height, canvas.width] }, null, file.name);
  };
  image.src = url;
}

function download(name, data, type) {
  const blob = new Blob([data], { type });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function bind() {
  $("btn-phantom").onclick = () => {
    setStatus("Generating phantom…");
    setTimeout(() => {
      const ph = MIA.makePhantom();
      setVolume(ph.volume, ph.groundTruth, "Synthetic phantom");
    }, 10);
  };
  $("file-data").onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      if (file.name.toLowerCase().endsWith(".npy")) {
        setVolume(MIA.toVolume(MIA.parseNpy(await file.arrayBuffer())), null, file.name);
      } else {
        loadImageFile(file);
      }
    } catch (err) {
      setStatus("Could not load file: " + err.message);
    }
    e.target.value = "";
  };
  $("file-config").onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const loaded = JSON.parse(await file.text());
    writeControls({ ...DEFAULT_CONFIG, ...loaded });
    e.target.value = "";
    run();
  };
  $("file-reference").onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const npy = MIA.parseNpy(await file.arrayBuffer());
    state.reference = Int32Array.from(npy.data);
    e.target.value = "";
    compareReference();
  };

  document.querySelectorAll("aside input, aside select").forEach((el) => {
    if (el.type !== "file") el.addEventListener("input", scheduleRun);
  });

  $("cv-input").addEventListener("click", (e) => {
    if (!state.volume) return;
    const rect = e.target.getBoundingClientRect();
    const [, H, W] = state.volume.shape;
    $("seed_x").value = Math.floor(((e.clientX - rect.left) / rect.width) * W);
    $("seed_y").value = Math.floor(((e.clientY - rect.top) / rect.height) * H);
    $("method").value = "region_growing";
    run();
  });

  $("btn-run-all").onclick = () => {
    setStatus("Running all algorithms…");
    setTimeout(() => { runAll(); setStatus("Done."); }, 10);
  };
  $("btn-export-config").onclick = () => download("config.json", JSON.stringify(readControls(), null, 2), "application/json");
  $("btn-export-labels").onclick = () => {
    if (!state.result) return;
    const [, H, W] = state.volume.shape;
    download("labels.npy", MIA.writeNpy(state.result.labels, [H, W]), "application/octet-stream");
  };
  $("btn-export-result").onclick = () => {
    if (!state.result) return;
    const [, H, W] = state.volume.shape;
    download("result.json", JSON.stringify({
      implementation: `JavaScript (browser: ${navigator.userAgent})`,
      dataset: state.name,
      config: state.config,
      timings_ms: state.result.timings,
      info: state.result.info,
      shape: [H, W],
    }, null, 2), "application/json");
  };

  document.querySelectorAll(".tabs button").forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("active", x === b));
      document.querySelectorAll(".tab").forEach((t) => { t.hidden = t.id !== "tab-" + b.dataset.tab; });
    };
  });
}

writeControls(DEFAULT_CONFIG);
updateLabels(DEFAULT_CONFIG);
bind();
$("btn-phantom").click();
