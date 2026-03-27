# PyTorch NVS Renderer

| English | [Português](README_PT.md) |

TODO
A library that allows rendering...

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
python render.py --module render_lvsm
```

Controls are WASD for forward, left, backward, right, left ctrl/space for down/up and mouse for camera movement.
Press ESC once to unlock the mouse and press twice to close.

### Using with custom models

#### As a module

To render using a custom model, import and use the function `render_model` with the following format:

```py
render_model(n_frames, initial_T, render, device, render_resolution, window_resolution=(800, 800))
```

Where:

- `n_frames: int`: Number of frames in the scene (should be 1 in the case of static NVS)
- `initial_T: tensor`: Initial 4x4 camera transformation matrix
- `render(T: tensor, frame_index: int) -> I: tensor`: A function that receives the current 4x4 camera transformation matrix and current frame index (which will be always zero in static NVS) and returns the rendered image (shape `(C=3, H, W)`) at that position
- `device: str`: Which device to use (currently only works with `cuda`)
- `render_resolution: (int, int)`: Which resolution to use for rendering images (should be the same as the shape of `render(T)` output)
- `window_resolution: (int, int)`: (optional) Which resolution to use for the window

#### As a script

Crate a module similar to render_lvsm.py that exports all the parameters described in the previous section.

Then do the usual configuration, build and run:

```bash
python render.py --module <your_module_name>
```
