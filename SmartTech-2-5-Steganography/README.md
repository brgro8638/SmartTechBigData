# Wavelet Steganography

Streamlit+Python app for å gjemme ren tekst eller filer i bilder.
Appen lagrer resultatet som PNG for å sikre lossless bevaring av den hemmelige meldingen. 

## Installation

1. Sett opp virtual environment:

```bash
python -m venv .venv
```
alternativt:
```bash
python3 -m venv .venv
```

2. Aktiver virtual environment:

Windows
```bash
.venv/Scripts/Activate
```
Kjør alternativt denne først, hvis scripts er disabled:
```bash
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

Linux
```bash
source .venv/bin/activate
```

3. Installer dependencies

```bash
pip install -r requirements.txt
```

4. Kjør

```bash
streamlit run app.py
```

Dersom browser ikke åpnes automatisk kan du manuelt gå til http://localhost:8501