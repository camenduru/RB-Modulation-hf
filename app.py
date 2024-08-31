import sys
import os
from pathlib import Path
import gc

# Add the StableCascade and CSD directories to the Python path
app_dir = Path(__file__).parent
sys.path.extend([
    str(app_dir),
    str(app_dir / "third_party" / "StableCascade"),
    str(app_dir / "third_party" / "CSD")
])

import yaml
import torch
from tqdm import tqdm
from accelerate.utils import set_module_tensor_to_device
import torch.nn.functional as F
import torchvision.transforms as T
from lang_sam import LangSAM
from inference.utils import *
from core.utils import load_or_fail
from train import WurstCoreC, WurstCoreB
from gdf_rbm import RBM
from stage_c_rbm import StageCRBM
from utils import WurstCoreCRBM
from gdf.schedulers import CosineSchedule
from gdf import VPScaler, CosineTNoiseCond, DDPMSampler, P2LossWeight, AdaptiveLossWeight
from gdf.targets import EpsilonTarget

# Enable mixed precision
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Flag for low VRAM usage
low_vram = True  # Set to True to enable low VRAM optimizations

# Function to clear GPU cache
def clear_gpu_cache():
    torch.cuda.empty_cache()
    gc.collect()

# Function to move model to CPU
def to_cpu(model):
    return model.cpu()

# Function to move model to GPU
def to_gpu(model):
    return model.cuda()

# Function definition for low VRAM usage
if low_vram:
    def models_to(model, device="cpu", excepts=None):
        """
        Change the device of nn.Modules within a class, skipping specified attributes.
        """
        for attr_name in dir(model):
            if attr_name.startswith('__') and attr_name.endswith('__'):
                continue  # skip special attributes

            attr_value = getattr(model, attr_name, None)

            if isinstance(attr_value, torch.nn.Module):
                if excepts and attr_name in excepts:
                    print(f"Except '{attr_name}'")
                    continue
                print(f"Change device of '{attr_name}' to {device}")
                attr_value.to(device)
        
        clear_gpu_cache()

# ... (rest of your setup code remains the same)

def infer(style_description, ref_style_file, caption):
    clear_gpu_cache()  # Clear cache before inference

    height=1024
    width=1024
    batch_size=1
    output_file='output.png'
    
    stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

    extras.sampling_configs['cfg'] = 4
    extras.sampling_configs['shift'] = 2
    extras.sampling_configs['timesteps'] = 20
    extras.sampling_configs['t_start'] = 1.0

    extras_b.sampling_configs['cfg'] = 1.1
    extras_b.sampling_configs['shift'] = 1
    extras_b.sampling_configs['timesteps'] = 10
    extras_b.sampling_configs['t_start'] = 1.0

    ref_style = resize_image(PIL.Image.open(ref_style_file).convert("RGB")).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)

    batch = {'captions': [caption] * batch_size}
    batch['style'] = ref_style

    x0_style_forward = models_rbm.effnet(extras.effnet_preprocess(ref_style.to(device)))

    conditions = core.get_conditions(batch, models_rbm, extras, is_eval=True, is_unconditional=False, eval_image_embeds=True, eval_style=True, eval_csd=False) 
    unconditions = core.get_conditions(batch, models_rbm, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)    
    conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
    unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)

    if low_vram:
        # The sampling process uses more vram, so we offload everything except two modules to the cpu.
        models_to(models_rbm, device="cpu", excepts=["generator", "previewer"])

    # Stage C reverse process.
    with torch.cuda.amp.autocast():  # Use mixed precision
        sampling_c = extras.gdf.sample(
            models_rbm.generator, conditions, stage_c_latent_shape,
            unconditions, device=device,
            **extras.sampling_configs,
            x0_style_forward=x0_style_forward,
            apply_pushforward=False, tau_pushforward=8,
            num_iter=3, eta=0.1, tau=20, eval_csd=True,
            extras=extras, models=models_rbm,
            lam_style=1, lam_txt_alignment=1.0,
            use_ddim_sampler=True,
        )
        for (sampled_c, _, _) in tqdm(sampling_c, total=extras.sampling_configs['timesteps']):
            sampled_c = sampled_c

    clear_gpu_cache()  # Clear cache between stages

    # Stage B reverse process.
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):                
        conditions_b['effnet'] = sampled_c
        unconditions_b['effnet'] = torch.zeros_like(sampled_c)
        
        sampling_b = extras_b.gdf.sample(
            models_b.generator, conditions_b, stage_b_latent_shape,
            unconditions_b, device=device, **extras_b.sampling_configs,
        )
        for (sampled_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs['timesteps']):
            sampled_b = sampled_b
        sampled = models_b.stage_a.decode(sampled_b).float()

    sampled = torch.cat([
        torch.nn.functional.interpolate(ref_style.cpu(), size=height),
        sampled.cpu(),
        ],
        dim=0)

    # Save the sampled image to a file
    sampled_image = T.ToPILImage()(sampled.squeeze(0))  # Convert tensor to PIL image
    sampled_image.save(output_file)  # Save the image

    clear_gpu_cache()  # Clear cache after inference

    return output_file  # Return the path to the saved image

import gradio as gr

gr.Interface(
    fn = infer,
    inputs=[gr.Textbox(label="style description"), gr.Image(label="Ref Style File", type="filepath"), gr.Textbox(label="caption")],
    outputs=[gr.Image()]
).launch()