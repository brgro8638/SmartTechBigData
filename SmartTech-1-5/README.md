# SmartTech 1-5: Data Coder

A GUI that reads image data in different ways and shows how Huffman coding, arithmetic
coding and CABAC encode it, and how they perform. All three coders are written from
scratch and are verified lossless on every run.

## Run

```bash
cd SmartTech-1-5
python -m pip install -r requirements.txt
streamlit run app.py
```

The Harnverhalt data is patient data and is not in the repository. The app finds
`Harnverhalt2.zip` in `SmartTech-1-5/data/` or in any parent folder, or you can upload it.
Tests: `python -m pytest tests`.

## What the GUI does

**Data sources** (sidebar):

| Source | Reading |
|---|---|
| Generated | Seven patterns: shapes with noise, gradient, checkerboard, Gaussian and uniform noise, skewed binary, constant |
| Medical | Harnverhalt DICOM through pydicom, as stored 12-bit values or an 8-bit soft-tissue window. Only pixels are read into the app; patient metadata is never used |
| Real-life | Bundled photos or your own PNG/JPEG, as grayscale |
| Raw file | Any file as 8- or 16-bit samples, with optional width |

**Tabs:** Input (image, residuals, histograms, entropy), Results (size, time, round-trip
check, a generated explanation), one tab per coder showing the actual encoding, and a
benchmark that runs all data through all settings.

## Design

```
sources.py            load every data type into a 2D integer array
prediction.py         reversible prediction: none, left neighbour, MED (JPEG-LS)
coders/huffman.py     canonical static Huffman
coders/arithmetic.py  static order-0 arithmetic coder, 32-bit integer interval
coders/cabac.py       binarization + context modelling + H.264 M-coder
benchmark.py          encode, decode from bytes alone, invert prediction, compare
app.py                the Streamlit GUI
```

Decisions that matter for the comparison:

- **Same input.** All coders receive the same symbols, so differences come from the coders.
- **Honest size.** Each stream is self-contained, including its header, and the decoder gets only those bytes.
- **Real CABAC engine.** The arithmetic core uses the H.264 tables (`rangeTabLPS`, state
  transitions), renormalization, bypass mode and termination.
- **Own binarization and contexts.** H.264's are built for transform coefficients, so these are adapted to pixels:
  - Residuals: significance, truncated unary to 14, then Exp-Golomb escape and sign. The context class comes from `|left| + |top|`.
  - Raw values: an MSB-first bit tree, split by the top 3 bits of the neighbours' mean.
  - The Exp-Golomb prefix is context-coded, unlike H.264, because CT residuals often reach it.
  - All contexts start at p = 0.5, because the H.264 init tables are tuned to video.

## Results

Bits per pixel on a 256 x 256 center crop. Full table: [`results/benchmark_256.csv`](results/benchmark_256.csv).
Regenerate with `python make_results.py`, or use the Benchmark tab. All 108 runs were verified lossless.

| Source | Prediction | H0 | Huffman | Arithmetic | CABAC |
|---|---|---:|---:|---:|---:|
| CT window 8 bit | none | 6.243 | 6.326 | 6.321 | **4.317** |
| CT window 8 bit | MED | 4.486 | 4.557 | 4.531 | **3.986** |
| CT stored 12 bit | none | 8.935 | 9.396 | 9.387 | **7.379** |
| CT stored 12 bit | MED | 6.064 | 6.191 | 6.176 | **6.117** |
| Photo (astronaut) | none | 7.572 | 7.668 | 7.656 | **5.322** |
| Photo (astronaut) | MED | 4.760 | 4.844 | 4.823 | **4.468** |
| Photo (elephant) | MED | 3.503 | 3.592 | 3.554 | **3.126** |
| Shapes with noise | MED | 3.681 | 3.723 | **3.710** | 3.744 |
| Skewed binary, 2 % ones | none | 0.140 | 1.001 | **0.142** | 0.150 |
| Constant | none | 0.000 | 1.001 | **0.002** | 0.030 |
| Uniform noise | none | 7.998 | **8.063** | 8.092 | 8.125 |

Full CT slice 141 (512 x 512) with MED prediction:

| | Huffman | Arithmetic | CABAC |
|---|---:|---:|---:|
| Stored 12 bit, total bytes | 197,557 | 196,412 | **189,111** |
| Stored 12 bit, ratio | 1.99 | 2.00 | **2.08** |
| Window 8 bit, total bytes | 103,214 | 101,775 | **76,455** |
| Window 8 bit, ratio | 2.54 | 2.58 | **3.43** |
| Encode / decode time, stored | 0.03 / 0.45 s | 0.75 / 1.13 s | 2.22 / 2.04 s |

For reference, the DICOM file for that slice is 220,830 bytes. It uses JPEG Lossless with
first-order prediction and also contains the metadata, so it is not a like-for-like comparison.

## Conclusions

1. **Prediction matters most on real images.** On the astronaut photo, MED takes Huffman from 7.67 to 4.84 bits/pixel. No entropy coder gets there alone with an order-0 model.
2. **Arithmetic coding barely beats Huffman on natural images** (about 1 % at most across the CT and photo runs), because no symbol there is frequent enough to need under 1 bit. Part of the gain goes to its larger header.
3. **Arithmetic coding wins by far on skewed data.** Huffman has a floor of 1 bit per symbol: 1.001 against 0.142 bits/pixel on the skewed binary image, where H0 is 0.140.
4. **CABAC wins on real data because of context, not arithmetic.** It beats H0 itself, since H0 only describes an order-0 model: 3.986 against H0 = 4.486 on the windowed CT.
5. **The CABAC gain depends on the data.** It is 12 % on the windowed CT but only 1 % on the stored 12-bit values with MED, where low bits are mostly noise that contexts cannot predict.
6. **CABAC can lose.** On noise it pays for learning without gaining from contexts (9.08 against 8.87 bits/pixel, uniform noise with MED). On constant data its probability floor of about 0.02 costs 0.03 bits per bin, against 0.002 bits/pixel for static arithmetic coding.
7. **Speed is the price.** Total over the benchmark: Huffman 0.3 s encode and 2.7 s decode, arithmetic 4.7 and 7.4 s, CABAC 15.3 and 12.5 s.

## Limitations

- Timings are for pure Python and only show relative cost. Huffman's encode is flattered because it joins precomputed strings.
- This is entropy coding of pixels, not a full image or video codec: no transform, no motion, no quantization.
- Photos are compared as grayscale.
- Windowed CT is lossless relative to the 8-bit image only. Windowing clips values, so only the stored values are the full clinical data.
- One CT slice is shown in the results. The slider covers all 281, but the numbers above should not be read as a series average.
