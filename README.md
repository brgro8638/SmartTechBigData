# SmartTechBigData
A repository for collaboration in the subjects DTE-2803 Store datasystemer and DTE-2804 Smart Teknologi. Her finnes oversikt over alle innleveringer og de ansvarlige for hver av dem, i tillegg til relaterte filer.

--  --  -- Check backlog: --   --  --
https://github.com/users/brgro8638/projects/1

--  --  --  --  --  --  --  --  --  --  

## Lossless coding benchmark

The `excel_compressor` folder contains the benchmarking application for issue
[#26](https://github.com/brgro8638/SmartTechBigData/issues/26). It provides a
Streamlit interface that accepts Excel workbooks and compares these lossless
coders:

- Simple RLE
- PackBits
- Huffman coding
- Arithmetic coding
- CABAC (tree-based)

Run the application from the repository root:

```bash
python -m pip install -r excel_compressor/requirements.txt
streamlit run excel_compressor/app.py
```

The default `insurance.csv` dataset is included in `excel_compressor/`, so the
benchmark and roundtrip checks work without any other repository checkout.
The regression checks can be run with:

```bash
python excel_compressor/test_roundtrip.py
```

## 2D→3D Clustering Visualizer

The `BigData-1-6-Visualizer3D` folder contains the application for issue
[#9](https://github.com/brgro8638/SmartTechBigData/issues/9). It clusters a
DICOM series (CT or MR) or the differential frames of a video with six
algorithms and compares them in interactive 3D plots. See
`BigData-1-6-Visualizer3D/README.md` for details.

```bash
python -m pip install -r BigData-1-6-Visualizer3D/requirements.txt
streamlit run BigData-1-6-Visualizer3D/app.py
```

## Medical Image Analysis Suite

The `BigData-1-3-MedicalImageSuite` folder contains the application for issue
[#6](https://github.com/brgro8638/SmartTechBigData/issues/6): a medical image analysis
application implemented in both Python (Streamlit) and JavaScript (browser,
also mobile). Pipeline: normalization, Gaussian smoothing, PCA, K-Means,
thresholding (manual/Otsu), region growing and watershed, with comparison of
algorithms and of implementations. See
`BigData-1-3-MedicalImageSuite/README.md` for details.

```bash
python -m pip install -r BigData-1-3-MedicalImageSuite/python/requirements.txt
streamlit run BigData-1-3-MedicalImageSuite/python/app.py
```

## Data Coder (Huffman, arithmetic coding, CABAC)

The `SmartTech-1-5` folder contains a Streamlit GUI that reads generated images,
Harnverhalt CT slices, real photos and raw files, and compares the encoding and
performance of the three coders. See [`SmartTech-1-5/README.md`](SmartTech-1-5/README.md).
