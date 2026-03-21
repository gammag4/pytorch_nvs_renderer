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
./run.sh ./render.py render_lvsm
```

## Adding models

Crate a script similar to render_lvsm.py that outputs:

- `device (str)`: The CUDA device used for rendering
- `initial_cam_state`: Tuple `(x, y, z, rotX, rotY)` With initial camera position and rotation in x and y axis
- `render(T: torch.Tensor): torch.Tensor`: Function that receives a 4x4 camera transformation tensor and returns the view with shape `(3, h, w)` rendered by the model at that position

Then do the usual configuration, build and run:

```bash
./run.sh ./render.py <your_script_name>
```
