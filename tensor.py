import sys
import os
import shlex

sys.path.append(os.path.abspath('LVSM'))
os.chdir('LVSM')

args = '''
--config "configs/LVSM_scene_decoder_only.yaml"
training.dataset_path = "./preprocessed_data/test/full_list.txt"
training.batch_size_per_gpu = 1
training.target_has_input =  false
training.num_views = 3
training.square_crop = true
training.num_input_views = 2
training.num_target_views = 1
inference.if_inference = true
'''
# sys.argv.pop()
sys.argv.extend(shlex.split(args))
sys.argv


from importlib import import_module
import os
import math
from easydict import EasyDict as edict
import torch
from torch.utils.data import DataLoader
import einops
from setup import init_config


# Init stuff

amp_dtypes = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    'tf32': torch.float32
}
config = init_config()
config.training.amp_dtype = amp_dtypes[config.training.amp_dtype]

torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32

device = torch.device('cuda:0')
torch.cuda.set_device(device)

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Dataset

dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, dname = dataset_name.rsplit(".", 1)
dataset = getattr(import_module(module), dname)(config)

dataloader = DataLoader(
    dataset,
    batch_size=config.training.batch_size_per_gpu,
    prefetch_factor=config.training.prefetch_factor,
    num_workers=config.training.num_workers,
    persistent_workers=True,
    shuffle=False,
    pin_memory=False,
    drop_last=True
)

# Model

module, lvsm = config.model.class_name.rsplit(".", 1)
model = getattr(import_module(module), lvsm)(config).to(device)
model.load_ckpt(config.training.checkpoint_dir)
model.eval()

# Get specific scene to render

batch = next(iter(dataloader))
batch = edict({k: v.to(device) if type(v) == torch.Tensor else v for k, v in batch.items()})
fxfycxcy_t, c2w_t = batch.fxfycxcy[:1, 2:3], batch.c2w[:1, 2:3]
images, fxfycxcy, c2w = batch.image[:1, :2], batch.fxfycxcy[:1, :2], batch.c2w[:1, :2]

# Renders a single frame given inputs and target poses
@torch.no_grad()
def render_single_frame(model, imgs, fxfycxcy, c2w, fxfycxcy_t, c2w_t, config):
    # imgs: (1, 2, 3, 256, 256)
    # fxfycxcy: (1, 2, 4)
    # c2w: (1, 2, 4, 4)
    # fxfycxcy_t: (1, 1, 4)
    # c2w_t: (1, 1, 4, 4)

    # assert c2w_t[-3] == fxfycxcy_t[-2] == 1, 'should only generate one target for each view'

    device = imgs.device
    h, w = imgs.shape[-2], imgs.shape[-1]
    patch_size = config.model.target_pose_tokenizer.patch_size

    # inputs
    o, d = model.process_data.compute_rays(c2w, fxfycxcy, h, w, device)
    posed_sources = model.get_posed_input(images=imgs, ray_o=o, ray_d=d)
    b, n_sources, c, h, w = posed_sources.shape

    source_tokens = model.image_tokenizer(posed_sources)
    _, n_patches, d = source_tokens.shape
    source_tokens = source_tokens.reshape(b, n_sources * n_patches, d)

    # targets
    o_t, d_t = model.process_data.compute_rays(c2w_t, fxfycxcy_t, h, w, device)
    target_poses = model.get_posed_input(ray_o=o_t, ray_d=d_t)
    n_targets = target_poses.shape[-4]

    target_tokens = model.target_pose_tokenizer(target_poses)
    target_tokens = target_tokens.reshape(b, n_targets, n_patches, d)

    # repeat source tokens for all targets (when rendering multiple targets at once)
    source_tokens = einops.repeat(
        source_tokens, 'b np d -> b v_target np d', v_target=n_targets)

    # transformer
    concat_tokens = torch.cat((source_tokens, target_tokens), dim=-2)
    b, n_views, n_tokens, _ = concat_tokens.shape
    concat_tokens = concat_tokens.reshape(b, n_views * n_tokens, d)
    concat_tokens = model.transformer_input_layernorm(concat_tokens)
    transformer_output = model.pass_layers(
        concat_tokens, gradient_checkpoint=False)
    transformer_output = transformer_output.reshape(b, n_views, n_tokens, d)

    # decode
    _, target_image_tokens = transformer_output.split(
        [n_sources * n_patches, n_patches], dim=-2)
    rendered = model.image_token_decoder(target_image_tokens)

    rendered = einops.rearrange(
        rendered, "b v (h_p w_p) (p1 p2 c) -> b v c (h_p p1) (w_p p2)",
        # v=n_targets,
        h_p=h // patch_size,
        w_p=w // patch_size,
        p1=patch_size,
        p2=patch_size,
        c=3
    )
    return rendered  # (1, 1, 3, 256, 256)


def compute_transform_matrix(controls, device):
    controls = edict(controls)
    z, x, y = -controls.forward, -controls.right, controls.up
    mX, mY = controls.mouseDeltaX, controls.mouseDeltaY

    speed = 0.01
    mouse_sensitivity = 0.0001

    T = torch.eye(4, dtype=torch.float32, device=device)
    T[:3, 3] = torch.tensor([x, y, z], device=device) * speed
    
    theta = torch.pi * mY * mouse_sensitivity
    ct, st = math.cos(theta), math.sin(theta)
    RX = torch.tensor([[1, 0, 0], [0, ct, -st], [0, st, ct]], device=device)

    theta = torch.pi * -mX * mouse_sensitivity
    ct, st = math.cos(theta), math.sin(theta)
    RY = torch.tensor([[ct, 0, st], [0, 1, 0], [-st, 0, ct]], device=device)
    
    R = RY @ RX
    R = torch.concat([torch.concat([R, torch.tensor([[0, 0, 0]], device=device).T], dim=1), torch.tensor([[0, 0, 0, 1]], device=device)])
    
    return T @ R


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
    global images
    global fxfycxcy
    global c2w
    global c2w_t
    global fxfycxcy_t

    T = compute_transform_matrix(controls, device)
    c2w_t = c2w_t @ T.inverse()

    with torch.no_grad(), torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=config.training.amp_dtype,
    ):
        result = render_single_frame(
            model, images, fxfycxcy, c2w, fxfycxcy_t, c2w_t, config)

    result = result[0, 0].float()
    tensor_info = get_tensor_info(result)

    # + torch.randn(3, 512, 512, device='cuda', dtype=torch.float32) * 0.3
    return tensor_info
