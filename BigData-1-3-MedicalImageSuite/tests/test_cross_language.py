"""Cross-language test: Python (own), Python (libraries) and JavaScript (Node).

Run from the project folder:   python tests/test_cross_language.py

1. The two phantom generators must be bit-identical (same input data).
2. For every method, with and without PCA, the three implementations are run
   on the same slice and their label maps compared. Own Python vs JavaScript
   must agree on at least 99.9% of the pixels.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mia import compare, library, own  # noqa: E402
from mia.js_bridge import js_phantom, node_available, run_js  # noqa: E402
from mia.phantom import make_phantom  # noqa: E402
from mia.pipeline import default_config, run_pipeline  # noqa: E402

MIN_AGREEMENT = 0.999


def configs():
    for method in ["kmeans", "threshold", "region_growing", "watershed"]:
        for use_pca in (False, True):
            config = default_config()
            config["method"] = method
            config["pca"]["enabled"] = use_pca
            if method == "threshold":
                for mode in ("otsu", "manual"):
                    c = default_config()
                    c.update(method="threshold", pca={"enabled": use_pca, "n_components": 2},
                             threshold={"mode": mode, "value": 0.5})
                    yield c
                continue
            if method == "region_growing" and use_pca:
                config["region_growing"]["tolerance"] = 0.08
            yield config


def main():
    if not node_available():
        sys.exit("Node.js is not installed – cannot run the JavaScript implementation.")

    volume, gt = make_phantom()
    js_volume, js_gt = js_phantom()
    assert np.array_equal(volume, js_volume), "phantom volumes differ between Python and JavaScript"
    assert np.array_equal(gt, js_gt), "phantom ground truth differs between Python and JavaScript"
    print("Phantom: Python and JavaScript volumes are bit-identical", volume.shape)

    failures = 0
    header = f"{'method':<28}{'PCA':<5}{'own↔JS':>9}{'own↔lib':>9}{'max|Δwork|':>12}{'ARI GT':>8}" \
             f"{'own ms':>9}{'lib ms':>9}{'JS ms':>9}"
    print(header)
    print("-" * len(header))
    for config in configs():
        img = volume[config["slice"]]
        r_own = run_pipeline(img, config, own)
        r_lib = run_pipeline(img, config, library)
        js_labels, js_work, js_result = run_js(img, config, repeats=3)

        a_js = compare.agreement(r_own["labels"], js_labels)
        a_lib = compare.agreement(r_own["labels"], r_lib["labels"])
        work_diff = float(np.abs(r_own["work"] - js_work).max())
        ari_gt = compare.agreement(gt[config["slice"]], r_own["labels"])["ARI"]
        name = config["method"] + (f" ({config['threshold']['mode']})" if config["method"] == "threshold" else "")
        js_ms = min(run["total"] for run in js_result["runs_ms"])
        print(f"{name:<28}{'yes' if config['pca']['enabled'] else 'no':<5}"
              f"{a_js['matched agreement']:>9.5f}{a_lib['matched agreement']:>9.5f}{work_diff:>12.2e}{ari_gt:>8.3f}"
              f"{r_own['timings']['total']:>9.1f}{r_lib['timings']['total']:>9.1f}{js_ms:>9.1f}")
        if a_js["matched agreement"] < MIN_AGREEMENT:
            failures += 1
            print(f"  FAIL: own Python and JavaScript agree on less than {MIN_AGREEMENT:.1%} of pixels")

    if failures:
        sys.exit(f"{failures} configuration(s) failed")
    print("All configurations passed.")


if __name__ == "__main__":
    main()
