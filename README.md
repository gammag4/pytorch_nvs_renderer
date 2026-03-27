# PyTorch NVS Renderer

| English | [Português](README_PT.md) |

An OpenGL renderer for 3D Novel View Synthesis (NVS) PyTorch models that takes CUDA tensors and renders them directly.

It uses the Python.h API to integrate the Python code with the OpenGL C++ code.

For now, it can render [LVSM](https://github.com/Haian-Jin/LVSM) scenes.

Some examples:

https://github.com/user-attachments/assets/16de9309-e82a-4c30-a4e0-e8285eb4e954

https://github.com/user-attachments/assets/7071a7dc-cba4-410b-ab48-004afeadcf46

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
pip install -r requirements.txt
```

Build and run:

```bash
python render.py render_lvsm
```

Controls are WASD for forward, left, backward, right, left ctrl/space for down/up and mouse for camera movement.
Press ESC once to unlock the mouse and press twice to close.

## Adding models

Crate a script similar to render_lvsm.py that outputs:

- `device (str)`: The CUDA device used for rendering
- `initial_cam_state`: Tuple `(x, y, z, rotX, rotY)` with initial camera position and rotation in x and y axis
- `render(T: torch.Tensor): torch.Tensor`: Function that receives a 4x4 camera transformation tensor and returns the view with shape `(3, h, w)` rendered by the model at that pose

Then do the usual configuration, build and run:

```bash
python render.py <your_script_name>
```
