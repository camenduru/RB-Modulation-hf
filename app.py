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
import PIL

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Flag for low VRAM usage
low_vram = True

# Function definition for low VRAM usage
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
    
    torch.cuda.empty_cache()
    gc.collect()

# Stage C model configuration
config_file = 'third_party/StableCascade/configs/inference/stage_c_3b.yaml'
with open(config_file, "r", encoding="utf-8") as file:
    loaded_config = yaml.safe_load(file)

core = WurstCoreCRBM(config_dict=loaded_config, device=device, training=False)

# Stage B model configuration
config_file_b = 'third_party/StableCascade/configs/inference/stage_b_3b.yaml'
with open(config_file_b, "r", encoding="utf-8") as file:
    config_file_b = yaml.safe_load(file)
    
core_b = WurstCoreB(config_dict=config_file_b, device=device, training=False)

# Setup extras and models for Stage C
extras = core.setup_extras_pre()

gdf_rbm = RBM(
    schedule=CosineSchedule(clamp_range=[0.0001, 0.9999]),
    input_scaler=VPScaler(), target=EpsilonTarget(),
    noise_cond=CosineTNoiseCond(),
    loss_weight=AdaptiveLossWeight(),
)

sampling_configs = {
    "cfg": 5,
    "sampler": DDPMSampler(gdf_rbm),
    "shift": 1,
    "timesteps": 20
}

extras = core.Extras(
    gdf=gdf_rbm,
    sampling_configs=sampling_configs,
    transforms=extras.transforms,
    effnet_preprocess=extras.effnet_preprocess,
    clip_preprocess=extras.clip_preprocess
)

models = core.setup_models(extras)
models.generator.eval().requires_grad_(False)

# Setup extras and models for Stage B
extras_b = core_b.setup_extras_pre()
models_b = core_b.setup_models(extras_b, skip_clip=True)
models_b = WurstCoreB.Models(
    **{**models_b.to_dict(), 'tokenizer': models.tokenizer, 'text_model': models.text_model}
)
models_b.generator.bfloat16().eval().requires_grad_(False)

if low_vram:
    # Off-load old generator (which is not used in models_rbm)
    models.generator.to("cpu")
    torch.cuda.empty_cache()

generator_rbm = StageCRBM()
for param_name, param in load_or_fail(core.config.generator_checkpoint_path).items():
    set_module_tensor_to_device(generator_rbm, param_name, "cpu", value=param)
generator_rbm = generator_rbm.to(getattr(torch, core.config.dtype)).to(device)
generator_rbm = core.load_model(generator_rbm, 'generator')

models_rbm = core.Models(
        effnet=models.effnet, previewer=models.previewer,
        generator=generator_rbm, generator_ema=models.generator_ema,
        tokenizer=models.tokenizer, text_model=models.text_model, image_model=models.image_model
    )
models_rbm.generator.eval().requires_grad_(False)

sam_model = LangSAM()

def infer(ref_style_file, style_description, caption, progress):
    global models_rbm, models_b, device
    if low_vram:
        models_to(models_rbm, device=device, excepts=["generator", "previewer"])
    try:
        
        caption = f"{caption} in {style_description}"
        height=1024
        width=1024
        batch_size=1
        
        stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

        extras.sampling_configs['cfg'] = 4
        extras.sampling_configs['shift'] = 2
        extras.sampling_configs['timesteps'] = 20
        extras.sampling_configs['t_start'] = 1.0

        extras_b.sampling_configs['cfg'] = 1.1
        extras_b.sampling_configs['shift'] = 1
        extras_b.sampling_configs['timesteps'] = 10
        extras_b.sampling_configs['t_start'] = 1.0

        progress(0.1, "Loading style reference image")
        ref_style = resize_image(PIL.Image.open(ref_style_file).convert("RGB")).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)

        batch = {'captions': [caption] * batch_size}
        batch['style'] = ref_style

        progress(0.2, "Processing style reference image")
        x0_style_forward = models_rbm.effnet(extras.effnet_preprocess(ref_style.to(device)))

        progress(0.3, "Generating conditions")
        conditions = core.get_conditions(batch, models_rbm, extras, is_eval=True, is_unconditional=False, eval_image_embeds=True, eval_style=True, eval_csd=False) 
        unconditions = core.get_conditions(batch, models_rbm, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)    
        conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
        unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)

        if low_vram:
            # The sampling process uses more vram, so we offload everything except two modules to the cpu.
            models_to(models_rbm, device="cpu", excepts=["generator", "previewer"])

        progress(0.4, "Starting Stage C reverse process")
        # Stage C reverse process.
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
        for (sampled_c, _, _) in progress.tqdm(tqdm(sampling_c, total=extras.sampling_configs['timesteps']), desc="Stage C reverse process"):
        #for i, (sampled_c, _, _) in enumerate(sampling_c, 1):
        #    if i % 5 == 0:  # Update progress every 5 steps
        #        progress(0.4 + 0.3 * (i / extras.sampling_configs['timesteps']), f"Stage C reverse process: step {i}/{extras.sampling_configs['timesteps']}")
            sampled_c = sampled_c

        progress(0.7, "Starting Stage B reverse process")
        # Stage B reverse process.
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):                
            conditions_b['effnet'] = sampled_c
            unconditions_b['effnet'] = torch.zeros_like(sampled_c)
            
            sampling_b = extras_b.gdf.sample(
                models_b.generator, conditions_b, stage_b_latent_shape,
                unconditions_b, device=device, **extras_b.sampling_configs,
            )
            for sampled_b, _, _ in progress.tqdm(tqdm(sampling_b, total=extras_b.sampling_configs['timesteps']), desc="Stage B reverse process"):
            #for i, (sampled_b, _, _) in enumerate(sampling_b, 1):
            #    if i % 1 == 0:  # Update progress every 1 step
            #        progress(0.7 + 0.2 * (i / extras_b.sampling_configs['timesteps']), f"Stage B reverse process: step {i}/{extras_b.sampling_configs['timesteps']}")
                sampled_b = sampled_b
            sampled = models_b.stage_a.decode(sampled_b).float()

        progress(0.9, "Finalizing the output image")
        sampled = torch.cat([
            torch.nn.functional.interpolate(ref_style.cpu(), size=(height, width)),
            sampled.cpu(),
        ], dim=0)

        # Remove the batch dimension and keep only the generated image
        sampled = sampled[1]  # This selects the generated image, discarding the reference style image

        # Ensure the tensor values are in the correct range
        sampled = torch.clamp(sampled, 0, 1)

        # Ensure the tensor is in [C, H, W] format
        if sampled.dim() == 3 and sampled.shape[0] == 3:
            sampled_image = T.ToPILImage()(sampled)  # Convert tensor to PIL image
        else:
            raise ValueError(f"Expected tensor of shape [3, H, W] but got {sampled.shape}")

        progress(1.0, "Inference complete")
        return sampled_image # Return the sampled_image PIL image

    finally:
        # Clear CUDA cache
        torch.cuda.empty_cache()
        gc.collect()

def infer_compo(style_description, ref_style_file, caption, ref_sub_file, progress):
    global models_rbm, models_b, device, sam_model
    if low_vram:
        models_to(models_rbm, device=device, excepts=["generator", "previewer"])
        models_to(sam_model, device=device)
        models_to(sam_model.sam, device=device)
    try:
        caption = f"{caption} in {style_description}"
        sam_prompt = f"{caption}"
        use_sam_mask = False
        
        batch_size = 1
        height, width = 1024, 1024
        stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

        extras.sampling_configs['cfg'] = 4
        extras.sampling_configs['shift'] = 2
        extras.sampling_configs['timesteps'] = 20
        extras.sampling_configs['t_start'] = 1.0
        extras_b.sampling_configs['cfg'] = 1.1
        extras_b.sampling_configs['shift'] = 1
        extras_b.sampling_configs['timesteps'] = 10
        extras_b.sampling_configs['t_start'] = 1.0

        progress(0.1, "Loading style and subject reference images")
        ref_style = resize_image(PIL.Image.open(ref_style_file).convert("RGB")).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)
        ref_images = resize_image(PIL.Image.open(ref_sub_file).convert("RGB")).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)
        
        batch = {'captions': [caption] * batch_size}
        batch['style'] = ref_style
        batch['images'] = ref_images

        progress(0.2, "Processing reference images")
        x0_forward = models_rbm.effnet(extras.effnet_preprocess(ref_images.to(device)))
        x0_style_forward = models_rbm.effnet(extras.effnet_preprocess(ref_style.to(device)))
        
        ## SAM Mask for sub
        use_sam_mask = False
        x0_preview = models_rbm.previewer(x0_forward)

        x0_preview_pil = T.ToPILImage()(x0_preview[0].cpu())
        sam_mask, boxes, phrases, logits = sam_model.predict(x0_preview_pil, sam_prompt)
        # sam_mask, boxes, phrases, logits = sam_model.predict(transform(x0_preview[0]), sam_prompt)
        sam_mask = sam_mask.detach().unsqueeze(dim=0).to(device)

        progress(0.3, "Generating conditions")
        conditions = core.get_conditions(batch, models_rbm, extras, is_eval=True, is_unconditional=False, eval_image_embeds=True, eval_subject_style=True, eval_csd=False)
        unconditions = core.get_conditions(batch, models_rbm, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False, eval_subject_style=True)    
        conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
        unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)

        if low_vram:
            models_to(models_rbm, device="cpu", excepts=["generator", "previewer"])
            models_to(sam_model, device="cpu")
            models_to(sam_model.sam, device="cpu")

        progress(0.4, "Starting Stage C reverse process")
        # Stage C reverse process.
        sampling_c = extras.gdf.sample(
            models_rbm.generator, conditions, stage_c_latent_shape,
            unconditions, device=device,
            **extras.sampling_configs,
            x0_style_forward=x0_style_forward, x0_forward=x0_forward,
            apply_pushforward=False, tau_pushforward=5, tau_pushforward_csd=10, 
            num_iter=3, eta=1e-1, tau=20, eval_sub_csd=True,
            extras=extras, models=models_rbm,  
            use_attn_mask=use_sam_mask,
            save_attn_mask=False,
            lam_content=1, lam_style=1,
            sam_mask=sam_mask, use_sam_mask=use_sam_mask,
            sam_prompt=sam_prompt
        )

        for sampled_c, _, _ in progress.tqdm(tqdm(sampling_c, total=extras.sampling_configs['timesteps']), desc="Stage C reverse process"):
        #for i, (sampled_c, _, _) in enumerate(sampling_c, 1):
        #    if i % 5 == 0:  # Update progress every 5 steps
        #        progress(0.4 + 0.3 * (i / extras.sampling_configs['timesteps']), f"Stage C reverse process: step {i}/{extras.sampling_configs['timesteps']}")
            sampled_c = sampled_c

        progress(0.7, "Starting Stage B reverse process")
        # Stage B reverse process.
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):                
            conditions_b['effnet'] = sampled_c
            unconditions_b['effnet'] = torch.zeros_like(sampled_c)
            
            sampling_b = extras_b.gdf.sample(
                models_b.generator, conditions_b, stage_b_latent_shape,
                unconditions_b, device=device, **extras_b.sampling_configs,
            )
            for sampled_b, _, _ in progress.tqdm(tqdm(sampling_b, total=extras_b.sampling_configs['timesteps']), desc="Stage B reverse process"):
            #for i, (sampled_b, _, _) in enumerate(sampling_b, 1):
            #    if i % 5 == 0:  # Update progress every 5 steps
            #        progress(0.7 + 0.2 * (i / extras_b.sampling_configs['timesteps']), f"Stage B reverse process: step {i}/{extras_b.sampling_configs['timesteps']}")
                sampled_b = sampled_b
            sampled = models_b.stage_a.decode(sampled_b).float()

        progress(0.9, "Finalizing the output image")
        sampled = torch.cat([
            torch.nn.functional.interpolate(ref_images.cpu(), size=(height, width)),
            torch.nn.functional.interpolate(ref_style.cpu(), size=(height, width)),
            sampled.cpu(),
        ], dim=0)

        # Remove the batch dimension and keep only the generated image
        sampled = sampled[2]  # This selects the generated image, discarding the reference images

        # Ensure the tensor values are in the correct range
        sampled = torch.clamp(sampled, 0, 1)

        # Ensure the tensor is in [C, H, W] format
        if sampled.dim() == 3 and sampled.shape[0] == 3:
            sampled_image = T.ToPILImage()(sampled)  # Convert tensor to PIL image
        else:
            raise ValueError(f"Expected tensor of shape [3, H, W] but got {sampled.shape}")

        progress(1.0, "Inference complete")
        return sampled_image  # Return the sampled_image PIL image

    finally:
        # Clear CUDA cache
        torch.cuda.empty_cache()
        gc.collect()

def run(style_reference_image, style_description, subject_prompt, subject_reference, use_subject_ref):
    result = None
    progress = gr.Progress(track_tqdm=True)
    if use_subject_ref is True:
        result = infer_compo(style_description, style_reference_image, subject_prompt, subject_reference, progress)
    else:
        result = infer(style_reference_image, style_description, subject_prompt, progress)
    return result

def show_hide_subject_image_component(use_subject_ref):
    if use_subject_ref is True:
        return gr.update(open=True)
    else:
        return gr.update(open=False)

import gradio as gr

with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("# RB-Modulation")
        gr.Markdown("## Training-Free Personalization of Diffusion Models using Stochastic Optimal Control")
        gr.HTML("""
        <div style="display:flex;column-gap:4px;">
            <a href='https://rb-modulation.github.io'>
                <img src='https://img.shields.io/badge/Project-Page-Green'>
            </a> 
            <a href='https://arxiv.org/pdf/2405.17401'>
                <img src='https://img.shields.io/badge/Paper-Arxiv-red'>
            </a>
        </div>
        """)
        with gr.Row():
            with gr.Column():
                style_reference_image = gr.Image(
                    label = "Style Reference Image",
                    type = "filepath"
                )
                style_description = gr.Textbox(
                    label ="Style Description"
                )
                subject_prompt = gr.Textbox(
                    label = "Subject Prompt"
                )
                use_subject_ref = gr.Checkbox(label="Use Subject Image as Reference", value=False)
                
                with gr.Accordion("Advanced Settings", open=False) as sub_img_panel:
                    subject_reference = gr.Image(label="Subject Reference", type="filepath")
                    
                submit_btn = gr.Button("Submit")

                
            with gr.Column():
                output_image = gr.Image(label="Output Image")
                gr.Examples(
                    examples = [
                        ["./data/cyberpunk.png", "cyberpunk art style", "a car", None, False],
                        ["./data/mosaic.png", "mosaic art style", "a lighthouse", None, False],
                        ["./data/glowing.png", "glowing style", "a dwarf", None, False],
                        ["./data/melting_gold.png", "melting golden 3D rendering style", "a dog", "./data/dog.jpg", True]
                    ],
                    fn=run,
                    inputs=[style_reference_image, style_description, subject_prompt, subject_reference, use_subject_ref],
                    outputs=[output_image],
                    cache_examples="lazy"
                
                )

    use_subject_ref.input(
        fn = show_hide_subject_image_component,
        inputs = [use_subject_ref],
        outputs = [sub_img_panel],
        queue = False
    )
    
    submit_btn.click(
        fn = run,
        inputs = [style_reference_image, style_description, subject_prompt, subject_reference, use_subject_ref],
        outputs = [output_image],
        show_api = False
    )

demo.queue().launch(show_error=True, show_api=False)