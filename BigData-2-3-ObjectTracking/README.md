# Installation

Create and activate a virtual environment. Then run one of the following:

## CPU (Windows or Linux)
```bash
pip install ultralytics numpy opencv-python Pillow pandas PyYAML
```

## CUDA (You have a NVIDIA GPU) (Windows or Linux)
Change `cu128` if you need a different CUDA version.
```bash
pip install ultralytics numpy opencv-python Pillow pands PyYAML --index-url https://download.pytorch.org/whl/cu128
```

Get the exact command for your CUDA version at https://pytorch.org/get-started/locally/ if needed.
