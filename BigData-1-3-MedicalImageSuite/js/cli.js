#!/usr/bin/env node
/*
 * Runs the JavaScript implementation from the command line, so the Python
 * app can compare both languages on exactly the same input.
 *
 *   node cli.js <image.npy | phantom> <config.json> <out_dir> [repeats]
 *
 * Writes out_dir/labels.npy (int32), out_dir/work.npy (float64, the image
 * the segmentation ran on) and out_dir/result.json (timings of every run).
 * With "phantom" as input, the volume and ground truth are written as well
 * (used by the test that checks the two phantom generators are identical).
 */
"use strict";
const fs = require("fs");
const path = require("path");
const MIA = require("./mia-core.js");

function readNpy(file) {
  const buf = fs.readFileSync(file);
  return MIA.parseNpy(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}

function main() {
  const [input, configFile, outDir, repeatsArg] = process.argv.slice(2);
  if (!input || !configFile || !outDir) {
    console.error("usage: node cli.js <image.npy | phantom> <config.json> <out_dir> [repeats]");
    process.exit(2);
  }
  const config = JSON.parse(fs.readFileSync(configFile, "utf8"));
  const repeats = Math.max(1, parseInt(repeatsArg || "1", 10));
  fs.mkdirSync(outDir, { recursive: true });

  let volume;
  if (input === "phantom") {
    const ph = MIA.makePhantom();
    volume = ph.volume;
    fs.writeFileSync(path.join(outDir, "phantom.npy"), MIA.writeNpy(ph.volume.data, ph.volume.shape));
    fs.writeFileSync(path.join(outDir, "phantom_gt.npy"), MIA.writeNpy(ph.groundTruth.data, ph.groundTruth.shape));
  } else {
    volume = MIA.toVolume(readNpy(input));
  }
  const [slices, H, W] = volume.shape;
  const index = Math.min(Math.max(config.slice || 0, 0), slices - 1);
  const img = MIA.getSlice(volume, index);

  const runs = [];
  let result;
  for (let r = 0; r < repeats; r++) {
    result = MIA.runPipeline(img, H, W, config);
    runs.push(result.timings);
  }
  fs.writeFileSync(path.join(outDir, "labels.npy"), MIA.writeNpy(result.labels, [H, W]));
  fs.writeFileSync(path.join(outDir, "work.npy"), MIA.writeNpy(result.work, [H, W]));
  fs.writeFileSync(path.join(outDir, "result.json"), JSON.stringify({
    implementation: `JavaScript (Node ${process.version})`,
    config,
    timings_ms: runs[runs.length - 1],
    runs_ms: runs,
    info: result.info,
    shape: [H, W],
  }, null, 2));
}

main();
