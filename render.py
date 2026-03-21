from importlib import import_module
import sys
import math
from easydict import EasyDict as edict
import torch

module_name = sys.argv[2]  # 'render_lvsm'
sys.argv.clear()
sys.argv.append('')
module = import_module(module_name)
initial_cam_state, render, device = [getattr(module, n) for n in ['initial_cam_state', 'render', 'device']]


current_state = initial_cam_state # TODO use rotation too
current_pos = torch.tensor(current_state[:3], dtype=torch.float32, device=device).reshape(3, 1)


def compute_transform_matrix(controls, current_pos, device):
    controls = edict(controls)
    z, x, y = -controls.forward, -controls.right, controls.up
    mX, mY = controls.angleX, controls.angleY

    speed = 0.05
    mouse_sensitivity = 0.001
    
    theta = torch.pi * mY * mouse_sensitivity
    ct, st = math.cos(theta), math.sin(theta)
    RX = torch.tensor([[1, 0, 0], [0, ct, -st], [0, st, ct]], device=device)

    theta = torch.pi * -mX * mouse_sensitivity
    ct, st = math.cos(theta), math.sin(theta)
    RY = torch.tensor([[ct, 0, st], [0, 1, 0], [-st, 0, ct]], device=device)
    
    R = RX @ RY

    current_pos = current_pos + R.T @ torch.tensor([[x, y, z]], device=device).T * speed
    
    R = torch.concat([torch.concat([R, torch.tensor([[0, 0, 0]], device=device).T], dim=1), torch.tensor([[0, 0, 0, 1]], device=device)])
    T = torch.eye(4, dtype=torch.float32, device=device)
    T[:3, 3:] = current_pos
    
    T = (R @ T).inverse()
    T = T.reshape(1, 1, *T.shape)
    
    return T, current_pos


def get_tensor_info(tensor: torch.Tensor) -> dict:
    """
    Get metadata about a tensor for C++ side to know how to interpret it.
    
    Returns:
        Dictionary with shape, dtype, and device pointer
    """

    if not tensor.is_cuda:
        raise RuntimeError("Tensor must be on CUDA device")
    
    if tensor.dtype == torch.float32:
        tensor = ((tensor * 0.5 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8)

    if tensor.shape[0] == 3 and len(tensor.shape) == 3:
        tensor = tensor.permute(1, 2, 0)
    
    if tensor.shape[2] == 3:
        ones = torch.ones((*tensor.shape[:2], 1), dtype=tensor.dtype, device=tensor.device) * 255
        tensor = torch.cat([tensor, ones], dim=2)
    
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    h, w, c = tensor.shape
    row_bytes = w * c * tensor.element_size()
    rows = h

    res = {
        "tensor": tensor,
        "pointer": tensor.data_ptr(),
        "row_bytes": row_bytes,
        "rows": rows,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device_id": tensor.device.index or 0,
    }
    
    return res


def update(controls):
    """
    Returns a GPU tensor of shape (3, 512, 512) in CUDA memory.
    This should not be changed bc it will be used later with images this shape.
    Implement this with your actual tensor generation logic.
    """
    
    global device
    global current_pos

    T, current_pos = compute_transform_matrix(controls, current_pos, device)
    result = render(T)
    tensor_info = get_tensor_info(result)
    return tensor_info
