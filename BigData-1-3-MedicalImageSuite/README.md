# Medical Image Analysis Suite

Cross-platform application for medical image analysis, implemented twice:

| | Python | JavaScript |
|---|---|---|
| GUI | Streamlit web app (`python/app.py`) | Browser app (`js/index.html`), works on desktop and mobile |
| Algorithms | `python/mia/own.py` (own, NumPy) and `python/mia/library.py` (SciPy / scikit-learn / scikit-image) | `js/mia-core.js` (own, plain JS) |
| Also runs in | – | Node.js (`js/cli.js`), used by the Python app for side-by-side comparison |

Both follow the same specification ([SPEC.md](SPEC.md)), read the same data
(`.npy`, images; DICOM is converted by the Python app), use the same
config JSON and write the same result format, so results can be compared
pixel by pixel.

## Pipeline

```
load slice → normalize (percentiles) → Gaussian smoothing → [PCA on 5 per-pixel features]
          → K-Means | thresholding (manual / Otsu) | region growing | watershed
```

## Run

Python app (from this folder):

```bash
python -m pip install -r python/requirements.txt
streamlit run python/app.py
```

JavaScript app: open `js/index.html` in a browser (double-click works, no
server or install needed). To open it on a phone, serve the folder and
browse to it from the same network:

```bash
python -m http.server 8602 --directory js
```

Cross-language test (needs Node.js):

```bash
python tests/test_cross_language.py
```

Data: the synthetic phantom is built in. Put your own files (`.zip` with
DICOM, `.dcm`, `.npy`, images) in `data/` and they show up in the Python
app. `data/` is ignored by git – DICOM files contain patient data.

## Comparing

- **Compare algorithms** (both apps): all methods, with and without PCA,
  on the same slice: runtime, number of labels, foreground fraction and,
  for the phantom, agreement with ground truth (ARI, matched accuracy).
- **Compare implementations** (Python app): own Python, library Python and
  JavaScript (Node) on the same slice and config: runtime per step, matched
  pixel agreement, ARI, max difference of the work image, difference maps.
  A `labels.npy` exported from the JavaScript browser app can be added.
- **Compare with Python result** (JavaScript app): load a `labels.npy`
  exported from the Python app.

## Results (phantom slice 5, 256×256; and MR angiography slice, 640×640)

`tests/test_cross_language.py` on the phantom:

- The two phantom generators are bit-identical.
- Own Python and JavaScript agree on **100 %** of the pixels for every
  method, with and without PCA. Without PCA the work images are
  bit-identical; with PCA they differ by ≈2·10⁻¹⁵ (different summation
  order and eigen-solver: NumPy `eigh` vs Jacobi rotations).
- Own vs library implementations agree on 99.98–100 %. The small
  differences are expected: `threshold_otsu` uses bin centres between the
  image min and max (own: fixed bins on [0, 1]), and scikit-learn's K-Means
  uses a relative tolerance and a different stopping rule.
- Best method against ground truth: watershed and K-Means (ARI ≈ 0.94).

On the real MR series (`dicoms.zip`, 640×640) own Python and JavaScript also
agree on 100 % for all eight method/PCA combinations. Runtimes show the
expected pattern: vectorized steps (filters, K-Means) are fast in NumPy,
pixel-by-pixel loops (region growing, watershed) are 3–10× faster in
JavaScript than in pure Python, and the compiled library code is fastest.

## Files

- `SPEC.md` – shared algorithm specification (the contract between languages)
- `config.example.json` – example of the shared config format
- `python/app.py` – Streamlit GUI
- `python/mia/` – `own.py`, `library.py`, `pipeline.py`, `compare.py`,
  `dataio.py` (DICOM/npy/images), `phantom.py`, `js_bridge.py` (Node runner)
- `js/index.html`, `js/app.js`, `js/style.css` – browser GUI
- `js/mia-core.js` – JavaScript algorithms (browser and Node)
- `js/cli.js` – command-line runner used by the Python app and the test
- `tests/test_cross_language.py` – cross-language test and benchmark
