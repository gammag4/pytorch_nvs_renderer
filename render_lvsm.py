import os
import sys
sys.path.append(os.path.abspath('LVSM'))

from importlib import import_module
import shlex
from pathlib import Path
from easydict import EasyDict as edict
import torch
from torch.utils.data import DataLoader
import einops
from setup import init_config


def is_valid_path(path):
    try:
        Path(path).resolve()
        return True
    except (OSError, RuntimeError):
        return False


def change_relative_paths(config, new_root):
    for k in config.keys():
        if type(config[k]) is str and config[k].startswith('./') and is_valid_path(config[k]):
            config[k] = os.path.join(new_root, config[k])

        if type(config[k]) in [dict, list, edict]:
            config[k] = change_relative_paths(config[k], new_root)

    return config


# Init stuff

args = '''
--config "./LVSM/configs/LVSM_scene_decoder_only.yaml"
training.dataset_path = "./preprocessed_data/test/full_list.txt"
training.batch_size_per_gpu = 1
training.target_has_input =  false
training.num_views = 3
training.square_crop = true
training.num_input_views = 2
training.num_target_views = 1
inference.if_inference = true
'''

sys.argv = ['']
sys.argv.extend(shlex.split(args))

amp_dtypes = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    'tf32': torch.float32
}
config = init_config()
config = change_relative_paths(config, './LVSM')
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
    shuffle=True,
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

initial_T = c2w_t[0, 0, :, :]


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
    
    mx, my = 1, 1

    # inputs
    o, d = model.process_data.compute_rays(c2w, fxfycxcy, h * mx, w * my, device)
    o, d = [einops.rearrange(i, 'b n c (h p1) (w p2) -> (b p1 p2) n c h w', p1=mx, p2=my) for i in (o, d)]
    posed_sources = model.get_posed_input(images=einops.repeat(imgs, 'b n c h w -> (b p) n c h w', p=mx * my), ray_o=o, ray_d=d)
    b, n_sources, c, h, w = posed_sources.shape

    source_tokens = model.image_tokenizer(posed_sources)
    _, n_patches, d = source_tokens.shape
    source_tokens = source_tokens.reshape(b, n_sources * n_patches, d)

    # targets
    o_t, d_t = model.process_data.compute_rays(c2w_t, fxfycxcy_t, h * mx, w * my, device)
    o_t, d_t = [einops.rearrange(i, 'b n c (h p1) (w p2) -> (b p1 p2) n c h w', p1=mx, p2=my) for i in (o_t, d_t)]
    target_poses = model.get_posed_input(ray_o=o_t, ray_d=d_t)
    n_targets = target_poses.shape[-4]

    target_tokens = model.target_pose_tokenizer(target_poses)
    target_tokens = target_tokens.reshape(b, n_targets, n_patches, d)

    # repeat source tokens for all targets (when rendering multiple targets at once)
    source_tokens = einops.repeat(source_tokens, 'b np d -> b n_targets np d', n_targets=n_targets)

    # transformer
    concat_tokens = torch.cat((source_tokens, target_tokens), dim=-2)
    b, n_views, n_tokens, _ = concat_tokens.shape
    concat_tokens = concat_tokens.reshape(b, n_views * n_tokens, d)
    concat_tokens = model.transformer_input_layernorm(concat_tokens)
    transformer_output = model.pass_layers(concat_tokens, gradient_checkpoint=False)
    transformer_output = transformer_output.reshape(b, n_views, n_tokens, d)

    # decode
    _, target_image_tokens = transformer_output.split([n_sources * n_patches, n_patches], dim=-2)
    rendered = model.image_token_decoder(target_image_tokens)

    rendered = einops.rearrange(
        rendered, "b n (h_p w_p) (p1 p2 c) -> b n c (h_p p1) (w_p p2)",
        # v=n_targets,
        h_p=h // patch_size,
        w_p=w // patch_size,
        p1=patch_size,
        p2=patch_size,
        c=3
    )
    rendered = einops.rearrange(rendered, '(b p1 p2) n c h w -> b n c (h p1) (w p2)', p1=mx, p2=my)
    
    return rendered  # (1, 1, 3, 256, 256)


def render(T):
    global images
    global fxfycxcy
    global c2w
    global fxfycxcy_t
    
    with torch.no_grad(), torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=config.training.amp_dtype,
    ):
        result = render_single_frame(
            model,
            images, fxfycxcy, c2w,
            fxfycxcy_t, T,
            config
        )

    return result[0, 0].float()
