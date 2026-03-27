import math
from easydict import EasyDict as edict
import torch
import pygame
import profiler


def compute_transform_matrix(controls, current_state, device):
    z, x, y = -controls.forward, -controls.right, controls.up
    rotX, rotY = current_state[3:]
    rotX += -controls.mouseDelta[0]
    rotY += controls.mouseDelta[1]

    theta = rotY
    ct, st = math.cos(theta), math.sin(theta)
    RX = torch.tensor([[1, 0, 0], [0, ct, -st], [0, st, ct]])

    theta = rotX
    ct, st = math.cos(theta), math.sin(theta)
    RY = torch.tensor([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    
    R = RX @ RY

    current_pos = torch.tensor(current_state[:3], dtype=torch.float32).reshape(3, 1)
    current_pos = current_pos + R.T @ torch.tensor([[x, y, z]], dtype=torch.float32).T
    
    R = torch.concat([torch.concat([R, torch.tensor([[0, 0, 0]]).T], dim=1), torch.tensor([[0, 0, 0, 1]])])
    T = torch.eye(4, dtype=torch.float32)
    T[:3, 3:] = current_pos
    
    T = (R @ T).inverse()
    T = T.reshape(1, 1, *T.shape)
    
    current_state = (*current_pos.squeeze().tolist(), rotX, rotY)
    
    return T.to(device), current_state


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

    res = edict(
        tensor=tensor,
        pointer=tensor.data_ptr(),
        row_bytes=row_bytes,
        rows=rows,
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        device_id=tensor.device.index or 0,
    )
    
    return res


def update(controls, current_state, render, device):
    """
    Returns a GPU tensor of shape (3, 512, 512) in CUDA memory.
    This should not be changed bc it will be used later with images this shape.
    Implement this with your actual tensor generation logic.
    """
    
    T, current_state = compute_transform_matrix(controls, current_state, device)
    result = render(T)
    tensor_info = get_tensor_info(result)
    return tensor_info, current_state


def get_controls(dt):
    speed = 1.0
    mouse_sensitivity = 0.001
    
    forward = 0.0
    right = 0.0
    up = 0.0
    mouseDelta = pygame.mouse.get_rel()
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        forward += 1
    if keys[pygame.K_s]:
        forward -= 1
    if keys[pygame.K_a]:
        right -= 1
    if keys[pygame.K_d]:
        right += 1
    if keys[pygame.K_LCTRL]:
        up -= 1
    if keys[pygame.K_SPACE]:
        up += 1

    controls = edict()
    
    vmp, mmp = speed * dt, mouse_sensitivity * torch.pi

    controls.right = right * vmp
    controls.up = up * vmp
    controls.forward = forward * vmp
    controls.mouseDelta = (mouseDelta[0] * mmp, mouseDelta[1] * mmp)
    
    return controls


def render_model(initial_T, render, device, screen_res):
    w, h = screen_res
    
    R = initial_T[:3, :3]
    x, y, z = (-R.T @ initial_T[:3, 3:]).squeeze().tolist()
    rotX, rotY = 0.0, 0.0 # TODO compute from R.T
    current_state = (x, y, z, rotX, rotY)
    
    # Setup pygame for keyboard input
    pygame.init()
    screen = pygame.display.set_mode((w * 3, h * 3))
    pygame.display.set_caption("NVS Renderer")

    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    surface = pygame.Surface((w, h))
    
    # Control loopimport asyncio
    clock = pygame.time.Clock()
    while True:
        dt = clock.tick() / 1000.0
        # print(f'fps: {clock.get_fps()}')
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            pygame.quit()
            return
        
        with profiler.RegionProfiler('create_tensor_from_controls'):
            controls = get_controls(dt)
            tensor_info, current_state = update(
                controls, current_state, render, device)

        with profiler.RegionProfiler('get_tensor_result'):
            canvas = tensor_info.tensor.permute(1, 0, 2)[:, :, :3].cpu().numpy()
        pygame.surfarray.blit_array(surface, canvas)

        # Render camera feed
        screen.blit(pygame.transform.scale(surface, (w * 3, h * 3)), (0, 0))
        pygame.display.flip()
        
        profiler.step()


if __name__ == '__main__':
    from importlib import import_module
    import sys

    module_name = sys.argv[1]  # 'render_lvsm'
    module = import_module(module_name)
    initial_T, render, device, screen_resolution = [getattr(module, n) for n in ['initial_T', 'render', 'device', 'screen_resolution']]

    profiler.start(warmup=20)
    
    # T (4, 4), render(T) -> img (c, h, w), device, resolution
    render_model(initial_T, render, device, screen_resolution)
    
    profiler.stop()
    profiler.print_results()
