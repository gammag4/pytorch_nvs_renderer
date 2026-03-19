# PyTorch NVS Renderer

A renderer for PyTorch NVS models that takes CUDA tensors directly and renders them with OpenGL.

It uses the Python.h API to integrate the Python code with the OpenGL C++ code.

## Usage

First, create the conda env:

```bash
conda create -n nvs_renderer python=3.13
conda activate nvs_renderer
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Then, run with `./run.sh nvs_renderer ./tensor.py`.
