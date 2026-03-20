# PyTorch NVS Renderer

A renderer for PyTorch NVS models that takes CUDA tensors directly and renders them with OpenGL.

It uses the Python.h API to integrate the Python code with the OpenGL C++ code.

For now, it can render LVSM scenes.

## Usage

Clone this repo:

```bash
git clone --recursive https://github.com/gammag4/pytorch_nvs_renderer
```

Follow instructions to configure LVSM for inference from [here](https://github.com/gammag4/LVSM) (use this version because the original code is not compatible with this).

Install required libraries:

```bash
sudo apt install libglfw3 libglfw3-dev libglm-dev
```

Create conda environment:

```bash
conda create -n nvs_renderer python=3.13
conda activate nvs_renderer
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Build and run:

```bash
./run.sh nvs_renderer ./tensor.py
```
