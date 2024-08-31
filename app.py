import sys
import os
from pathlib import Path
import gc
import traceback

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
import PIL

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

# Load configurations
config_file = 'third_party/StableCascade/configs/inference/stage_c_3b.yaml'
with open(config_file, "r", encoding="utf-8") as file:
    loaded_config = yaml.safe_load(file)

config_file_b = 'third_party/StableCascade/configs/inference/stage_b_3b.yaml'
with open(config_file_b, "r", encoding="utf-8") as file:
    config_file_b = yaml.safe_load(file)

def initialize_models():
    global models_rbm, models_b, extras, extras_b, core, core_b
    
    # Clear any existing models from memory
    models_rbm = None
    models_b = None
    extras = None
    extras_b = None
    
    # Clear GPU cache
    clear_gpu_cache()
    
    # Initialize models
    core = WurstCoreCRBM(config_dict=loaded_config, device=device, training=False)
    core_b = WurstCoreB(config_dict=config_file_b, device=device, training=False)
    
    extras = core.setup_extras_pre()
    models = core.setup_models(extras)
    
    extras_b = core_b.setup_extras_pre()
    models_b = core_b.setup_models(extras_b, skip_clip=True)
    models_b = WurstCoreB.Models(
        **{**models_b.to_dict(), 'tokenizer': models.tokenizer, 'text_model': models.text_model}
    )
    
    # Initialize models_rbm
    generator_rbm = StageCRBM()
    for param_name, param in load_or_fail(core.config.generator_checkpoint_path).items():
        set_module_tensor_to_device(generator_rbm, param_name, "cpu", value=param)
    
    generator_rbm = generator_rbm.to(getattr(torch, core.config.dtype)).to(device)
    generator_rbm = core.load_model(generator_rbm, 'generator')
    
    models_rbm = core.Models(
        effnet=models.effnet,
        previewer=models.previewer,
        generator=generator_rbm,
        generator_ema=models.generator_ema,
        tokenizer=models.tokenizer,
        text_model=models.text_model,
        image_model=models.image_model
    )
    
    # Move models to appropriate devices
    models_rbm.generator.to(device).eval().requires_grad_(False)
    models_b.generator.to(device).eval().requires_grad_(False)
    
    clear_gpu_cache()

def infer(style_description, ref_style_file, caption):
    try:
        # Initialize (or reinitialize) models before each inference
        initialize_models()
        
        height = 1024
        width = 1024
        batch_size = 1
        output_file = 'output.png'
        
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

        x0_style_forward = models_rbm.effnet(extras.effnet_preprocess(ref_style))

        conditions = core.get_conditions(batch, models_rbm, extras, is_eval=True, is_unconditional=False, eval_image_embeds=True, eval_style=True, eval_csd=False) 
        unconditions = core.get_conditions(batch, models_rbm, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)    
        conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
        unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)

        if low_vram:
            # Offload non-essential models to CPU for memory savings
            models_to(models_rbm, device="cpu", excepts=["generator", "previewer"])

        # Stage C reverse process
        with torch.cuda.amp.autocast():
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

        # Stage B reverse process
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

        # Post-process and save the image
        sampled = sampled.cpu()  # Move to CPU before processing

        # Ensure the tensor is in [C, H, W] format
        if sampled.dim() == 4 and sampled.size(0) == 1:
            sampled = sampled.squeeze(0)
        
        if sampled.dim() == 3 and sampled.shape[0] == 3:
            sampled_image = T.ToPILImage()(sampled)  # Convert tensor to PIL image
            sampled_image.save(output_file)  # Save the image as a PNG
        else:
            raise ValueError(f"Expected tensor of shape [3, H, W] but got {sampled.shape}")

    except Exception as e:
        print(f"An error occurred during inference: {str(e)}")
        traceback.print_exc()  # This will print the full traceback
        return None

    finally:
        clear_gpu_cache()  # Always clear cache after inference

    return output_file  # Return the path to the saved image

import gradio as gr

def gradio_interface(style_description, ref_style_file, caption):
    return infer(style_description, ref_style_file, caption)

gr.Interface(
    fn=gradio_interface,
    inputs=[gr.Textbox(label="style description"), gr.Image(label="Ref Style File", type="filepath"), gr.Textbox(label="caption")],
    outputs=[gr.Image()]
).launch()