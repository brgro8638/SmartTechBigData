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
