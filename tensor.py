import torch
from torchvision.io import decode_image
import torchvision.transforms as T
import ctypes

def get_tensor(controls):
    """
    Returns a GPU tensor of shape (3, 512, 512) in CUDA memory.
    This should not be changed bc it will be used later with images this shape.
    Implement this with your actual tensor generation logic.
    """
    with open('Untitled.jpg', 'rb') as f:
        img_bytes = bytearray(f.read())

    print(controls)
    
    img_tensor = decode_image(torch.frombuffer(img_bytes, dtype=torch.uint8)).float() / 255.0
    img_tensor = T.Resize((512, 512))(img_tensor)
    return img_tensor.to(device='cuda') + torch.randn(3, 512, 512, device='cuda', dtype=torch.float32) * 0.3


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
