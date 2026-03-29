import asyncio
import queue
import argparse
import math
from easydict import EasyDict as edict
import torch
import pygame
import pygame_gui
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


def render_func(frame_index, render, T):
    """
    Returns a GPU tensor of shape (3, 512, 512) in CUDA memory.
    This should not be changed bc it will be used later with images this shape.
    Implement this with your actual tensor generation logic.
    """

    res = render(T, frame_index)
    res = get_tensor_info(res)
    
    return res


def get_controls(dt, is_navigating):
    if not is_navigating:
        return edict(right=0.0, up=0.0, forward=0.0, mouseDelta=(0.0, 0.0))

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


def render_model(n_frames, initial_T, render, device, render_resolution, window_resolution=(800, 800)):
    w, h = render_resolution
    win_w, win_h = window_resolution
    
    R = initial_T[:3, :3]
    x, y, z = (-R.T @ initial_T[:3, 3:]).squeeze().tolist()
    rotX, rotY = 0.0, 0.0 # TODO compute from R.T
    current_state = (x, y, z, rotX, rotY)
    
    # Setup pygame for keyboard input
    pygame.init()
    screen = pygame.display.set_mode(window_resolution)
    manager = pygame_gui.UIManager(window_resolution)
    pygame.display.set_caption("NVS Renderer")
    
    is_navigating = True
    is_playing = False

    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    surface = pygame.Surface((w, h))

    frame_index = 0
    time_slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((0, -50), (win_w / 2, 30)),
        start_value=0,
        value_range=(0, 1 if n_frames == 1 else n_frames - 1),
        manager=manager,
        anchors={
            'centerx': 'centerx',
            'bottom': 'bottom'
        }
    )
    play_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((20, -50), (100, 30)), # Position and size (x, y), (width, height)
        text='Play',
        manager=manager,
        anchors={
            'left': 'left',
            'bottom': 'bottom'
        }
    )
    if n_frames == 1:
        time_slider.hide()
        play_button.hide()
    
    processing = queue.Queue(4)
    streams_cache = queue.Queue()
    current = None
    
    # Control loopimport asyncio
    clock = pygame.time.Clock()
    render_clock = pygame.time.Clock()
    while True:
        with profiler.RegionProfiler('main_loop'):
            dt = clock.tick() / 1000.0
            # print(f'fps: {clock.get_fps()}')
            print(f'fps: {render_clock.get_fps()}')
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if not is_navigating:
                            pygame.quit()
                            return
                        
                        is_navigating = False
                        pygame.mouse.set_visible(True)
                        pygame.event.set_grab(False)

                consumed = False

                if not is_navigating:
                    # Time slider
                    if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                        if event.ui_element == time_slider:
                            frame_index = int(event.value)
                    
                    # Play button
                    if event.type == pygame_gui.UI_BUTTON_PRESSED:
                        if event.ui_element == play_button:
                            is_playing = not is_playing
                            play_button.set_text('Pause' if is_playing else 'Play')
                    
                    consumed = manager.process_events(event)

                # Check click on screen
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # left mouse
                        if not consumed:
                            is_navigating = True
                            pygame.mouse.set_visible(False)
                            pygame.event.set_grab(True)
                            
                            # discards current mouse delta to prevent mouse drift when switching back
                            _ = pygame.mouse.get_rel()
            
            if is_playing:
                frame_index = (frame_index + 1) % n_frames
                time_slider.set_current_value(frame_index)

            manager.update(dt)
            
            # print(f'{processing.qsize()} {streams_cache.qsize()}')
            
            controls = get_controls(dt, is_navigating)
            T, current_state = compute_transform_matrix(controls, current_state, device)

            if not processing.full():
                event, stream = streams_cache.get() if not streams_cache.empty() else (torch.cuda.Event(), torch.cuda.Stream())
                with torch.cuda.stream(stream):
                    tensor = render_func(frame_index, render, T).tensor.permute(1, 0, 2)[:, :, :3]
                    event.record(stream)
                processing.put((tensor, event, stream))
            
            if not processing.empty() and current is None:
                current = processing.get()

            if current is not None:
                tensor, event, stream = current
                if event.query():
                    event.synchronize()
                    canvas = tensor.cpu().numpy()
                    
                    pygame.surfarray.blit_array(surface, canvas)

                    # Render camera feed
                    screen.blit(pygame.transform.scale(surface, window_resolution), (0, 0))
                    
                    current = None
                    streams_cache.put((event, stream))
                    render_clock.tick()
            
            if not is_navigating:
                manager.draw_ui(screen)
            
            pygame.display.flip()
        
        profiler.step()


async def main():
    from importlib import import_module
    import sys

    parser = argparse.ArgumentParser()

    parser.add_argument("--module", required=True)
    args = parser.parse_args()

    module = import_module(args.module) # 'render_lvsm'
    (
        n_frames,
        initial_T,
        render,
        device,
        render_resolution
    ) = [getattr(module, n) for n in [
        'n_frames',
        'initial_T',
        'render',
        'device',
        'render_resolution'
    ]]
    
    window_resolution = getattr(module, 'window_resolution', None)
    
    render_model(n_frames, initial_T, render, device, render_resolution, window_resolution or (800, 800))


if __name__ == '__main__':
    with profiler.Profiler(should_profile=True, warmup=20):
        # T (4, 4), render(T) -> img (c, h, w), device, resolution
        asyncio.run(main())
    
    profiler.print_results()
    # profiler.dump('profiler_dump.pt')
