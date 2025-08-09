import random
import os
from argparse import ArgumentParser
from datetime import datetime
import subprocess
import sys
import gc

import gradio as gr
import torch
import torchvision.transforms as transforms
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from dotenv import load_dotenv
from PIL import Image
from huggingface_hub import hf_hub_download

from HYPIR.enhancer.sd2 import SD2Enhancer
from HYPIR.utils.captioner import EmptyCaptioner, GPTCaptioner, Florence2Captioner


load_dotenv()
error_image = Image.open(os.path.join("assets", "gradio_error_img.png"))

# --- Create an output directory for saved images ---
if not os.path.exists("outputs"):
    os.makedirs("outputs")

parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--local", action="store_true")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--captioner_type", type=str, default="florence2", choices=["none", "gpt", "florence2"], help="Type of captioner to use for the 'auto' prompt feature.")
parser.add_argument("--max_size", type=str, default=None, help="Comma-seperated image size")
args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"✅ Automatically selected device: {device.upper()}")

max_size = args.max_size
if max_size is not None:
    max_size = tuple(int(x) for x in max_size.split(","))
    if len(max_size) != 2:
        raise ValueError(f"Invalid max size: {max_size}")
    print(f"Max size set to {max_size}, max pixels: {max_size[0] * max_size[1]}")

# --- Initialize the selected captioner ---
captioner = None

if args.captioner_type == "florence2":
    try:
        captioner = Florence2Captioner(device=device)
    except ImportError as e:
        print(f"Warning: Could not initialize Florence2Captioner. {e}")
        captioner = EmptyCaptioner(device=device)

elif args.captioner_type == "gpt":
    if (
        "GPT_API_KEY" not in os.environ
        or "GPT_BASE_URL" not in os.environ
        or "GPT_MODEL" not in os.environ
    ):
        print(
            "Warning: To use gpt-generated captions, please specify `GPT_API_KEY`, "
            "`GPT_BASE_URL`, and `GPT_MODEL` in your .env file. "
            "Falling back to no captioner."
        )
        captioner = EmptyCaptioner(device=device)
    else:
        captioner = GPTCaptioner(
            api_key=os.getenv("GPT_API_KEY"),
            base_url=os.getenv("GPT_BASE_URL"),
            model=os.getenv("GPT_MODEL"),
        )
else:
    captioner = EmptyCaptioner(device=device)

to_tensor = transforms.ToTensor()

# --- Pre-download and cache the HYPIR model ---
print("Checking for HYPIR model weights...")
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "HYPIR_sd2.pth")

if not os.path.exists(model_path):
    print(f"Downloading HYPIR model to {model_path}...")
    try:
        hf_hub_download(
            repo_id="lxq007/HYPIR",
            filename="HYPIR_sd2.pth",
            local_dir=model_dir
        )
        print("Download complete.")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        # Exit if the model is essential for the app to run
        sys.exit(1)
else:
    print("HYPIR model already exists.")
        
config = OmegaConf.load(args.config)
if config.base_model_type == "sd2":
    model = SD2Enhancer(
        base_model_path=config.base_model_path,
        weight_path=config.weight_path,
        lora_modules=config.lora_modules,
        lora_rank=config.lora_rank,
        model_t=config.model_t,
        coeff_t=config.coeff_t,
        device=device,
    )
    model.init_models()
else:
    raise ValueError(config.base_model_type)

def save_image(image, status_message):
    """Saves the image with a timestamp and returns an updated status message."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"outputs/img_{timestamp}.png"
    image.save(filename)
    updated_message = f"{status_message}\n\n💾 Saved to: {filename}"
    return updated_message, filename

def open_output_folder():
    """Opens the 'outputs' directory in the default file explorer."""
    folder_path = os.path.abspath("outputs")
    if sys.platform == "win32":
        os.startfile(folder_path)
    elif sys.platform == "darwin":  # macOS
        subprocess.run(["open", folder_path])
    else:  # Linux and other Unix-like OS
        subprocess.run(["xdg-open", folder_path])

def toggle_autosave(choice):
    """Updates the visibility of the manual save button based on the checkbox."""
    return gr.update(visible=not choice)


def process(
    image,
    prompt,
    upscale,
    patch_size,
    stride,
    seed,
    autosave,
    caption_detail_toggle,
    prompt_suffix,
    progress=gr.Progress(track_tqdm=True),
):
    pil_image = error_image  # Default to the error image
    status_message = "An unknown error occurred."
    
    if image is None:
        status_message = "❌ Failed: Please provide an input image."
        return [error_image, error_image], status_message, None

    original_image = image.copy()

    try:
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        set_seed(seed)
        image = image.convert("RGB")

        if max_size is not None:
            out_w, out_h = tuple(int(x * upscale) for x in image.size)
            if out_w * out_h > max_size[0] * max_size[1]:
                status_message = (
                    f"❌ Failed: The requested resolution ({out_h}, {out_w}) exceeds the maximum pixel limit."
                )
                return [original_image, error_image], status_message, original_image

        if prompt == "auto":
            if captioner and captioner.is_functional:
                generated_prompt = captioner(image, use_more_detailed=caption_detail_toggle)
                if prompt_suffix and prompt_suffix.strip():
                    prompt = f"{generated_prompt}, {prompt_suffix.strip()}"
                else:
                    prompt = generated_prompt
            else:
                status_message = "❌ Failed: This app was not launched with a functional auto-captioner."
                return [original_image, error_image], status_message, original_image

        image_tensor = to_tensor(image).unsqueeze(0)
        
        pil_image = model.enhance(
            lq=image_tensor,
            prompt=prompt,
            upscale=upscale,
            patch_size=patch_size,
            stride=stride,
            return_type="pil",
        )[0]
        
        status_message = f"✅ Success!\n\nUsed prompt:\n{prompt}"

        if autosave:
            status_message, _ = save_image(pil_image, status_message)

    except Exception as e:
        status_message = f"❌ Failed: {e} :("
    
    finally:
        print("Cleaning up memory...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return [original_image, pil_image], status_message, pil_image
            
def batch_process(
    files,
    prompt,
    upscale,
    patch_size,
    stride,
    seed,
    caption_detail_toggle,
    prompt_suffix,
    progress=gr.Progress(track_tqdm=True),
):
    summary = "❌ Batch process failed unexpectedly."

    try:
        if not files:
            return "Please upload images to process."

        batch_dir = os.path.join("outputs", f"batch_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(batch_dir)

        if seed != -1:
            set_seed(seed)

        success_count = 0
        failed_files = []
        
        for file_obj in progress.tqdm(files, desc="Starting Batch Process"):
            original_filename = os.path.basename(file_obj.name)
            try:
                if seed == -1:
                    set_seed(random.randint(0, 2**32 - 1))

                image = Image.open(file_obj.name).convert("RGB")
                
                current_prompt = prompt
                if current_prompt.lower() == "auto":
                    if captioner and captioner.is_functional:
                        generated_prompt = captioner(image, use_more_detailed=caption_detail_toggle)
                        if prompt_suffix and prompt_suffix.strip():
                            current_prompt = f"{generated_prompt}, {prompt_suffix.strip()}"
                        else:
                            current_prompt = generated_prompt
                    else:
                        failed_files.append(f"{original_filename} (Error: Auto-captioning is disabled)")
                        continue

                image_tensor = to_tensor(image).unsqueeze(0)
                pil_image = model.enhance(
                    lq=image_tensor,
                    prompt=current_prompt,
                    upscale=upscale,
                    patch_size=patch_size,
                    stride=stride,
                    return_type="pil",
                )[0]

                output_filename = os.path.join(batch_dir, original_filename)
                pil_image.save(output_filename)
                success_count += 1

            except Exception as e:
                failed_files.append(f"{original_filename} (Error: {e})")
        
        summary = f"## ✅ Batch Processing Complete\n\n"
        summary += f"Successfully upscaled **{success_count}** of **{len(files)}** images.  \n"
        summary += f"Results saved in directory: `{batch_dir}`"

        if failed_files:
            summary += "\n\n### ❌ The following files could not be processed:\n"
            for filename in failed_files:
                summary += f"- {filename}\n"

    except Exception as e:
        summary = f"❌ Batch process failed critically: {e}"

    finally:
        print("Cleaning up memory after batch process...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return summary
    

MARKDOWN = """
## HYPIR: Harnessing Diffusion-Yielded Score Priors for Image Restoration

[GitHub](https://github.com/XPixelGroup/HYPIR) | [Paper](TODO) | [Project Page](TODO)

If HYPIR is helpful for you, please help star the GitHub Repo. Thanks!
"""

block = gr.Blocks(theme=gr.themes.Default()).queue()
with block:
    # This invisible component will hold the last generated image
    # so the manual save button can access it.
    image_state = gr.State()
    
    with gr.Row():
        gr.Markdown(MARKDOWN)

    with gr.Row():
        with gr.Tabs():
            # --- SINGLE IMAGE TAB ---
            with gr.TabItem("Single Image"):
                with gr.Row():
                    with gr.Column():
                        image = gr.Image(type="pil", label="Input Image")
                        run_single = gr.Button(value="Run", variant="primary")
                    with gr.Column():
                        result_slider = gr.ImageSlider(label="Before | After", interactive=False, show_label=True, max_height=900)
                        with gr.Group():
                            autosave_toggle = gr.Checkbox(label="Autosave", value=True)
                            manual_save_button = gr.Button(value="Save Image", variant="primary",visible=False)

            # --- BATCH PROCESS TAB ---
            with gr.TabItem("Batch Process"):
                with gr.Row():
                    with gr.Column():
                        batch_files = gr.File(
                            label="Upload Images",
                            file_count="multiple",
                            type="filepath",
                            file_types=["image"]
                        )
                        run_batch = gr.Button(value="Run Batch Process", variant="primary")
                    with gr.Column():
                        # user info for batch processing
                        gr.Markdown(
                            """
                            ### How to Batch Process

                            📥 **Drag & drop** your images into the upload box.

                            ⚙️ All settings (Prompt, Upscale Factor, Patch Size, etc.) will be applied to every image.

                            🎲 Leave the Seed at **-1** to get a unique, random result for each image.

                            📂 Your results are **auto-saved** into a new folder inside the outputs directory.
                            """
                        )
                
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label=(
                "Prompt (Input 'auto' for an AI-generated caption)"
                if captioner.is_functional else "Prompt (Auto-captioning disabled)"
            ))
            with gr.Row():
                upscale = gr.Slider(minimum=1, maximum=8, value=1, scale=3, label="Upscale Factor", step=1)
                seed = gr.Number(label="Seed", value=-1, scale=1, info="-1 = random")
            with gr.Row():
                with gr.Accordion("Options", open=False):
                    with gr.Group():
                        gr.Markdown("Auto-Prompt Settings (only used if prompt is 'auto')")
                        keep_captioner_loaded_toggle = gr.Checkbox(
                            label="Keep Captioner Model in VRAM",
                            value=False,
                            info="Speeds up repeated 'auto' prompts but occupies VRAM."
                        )
                        caption_detail_toggle = gr.Checkbox(
                            label="MORE_DETAILED_CAPTION", 
                            value=True,
                            info="If checked, uses Florence-2's MORE_DETAILED_CAPTION. If unchecked, uses DETAILED_CAPTION."
                        )
                        prompt_suffix = gr.Textbox(
                            label="Append to Auto-Prompt", 
                            placeholder="e.g., cinematic lighting, 4k, high quality",
                            info="Adds extra keywords to the end of the AI-generated prompt."
                        )
                    patch_size = gr.Slider(minimum=512, maximum=1024, value=512, label="Patch Size", step=128)
                    stride = gr.Slider(minimum=256, maximum=1024, value=256, label="Patch Stride", step=128)
                    
        with gr.Column(scale=1):     
            status = gr.Textbox(label="Status", interactive=False, lines=4)
            open_folder_button = gr.Button(value="Open Output Folder", variant="huggingface")   

    # --- Accordion for Help/Documentation ---
    with gr.Accordion("What do the 'Patch' settings mean?", open=False):
        gr.Markdown(
            """
            ### Understanding the Patch-Based Process

            To handle images of any size, this model breaks your input image into many smaller, square sections called **"patches"**. It processes each patch one by one and then stitches them all back together to create your final, upscaled image.

            The settings below control how this happens.
            ---
            #### **Patch Size**
            This slider determines the size of each square patch (e.g., 512x512 pixels).
            -   **Larger Size**: Allows the AI to "see" more of the image at once. This can lead to more consistent and coherent results, especially on images with large subjects.
            -   **The Trade-Off**: Larger patches require significantly more VRAM!

            #### **Patch Stride**
            This slider controls the distance the "window" moves before processing the next patch. This creates an **overlap** between the patches. The amount of overlap is simply `Patch Size - Patch Stride`.
            -   **Smaller Stride**: Increases the overlap between neighboring patches. This helps create a smoother, more seamless blend between the stitched-together sections, reducing visible edges or inconsistencies.
            -   **The Trade-Off**: A smaller stride means more patches have to be processed, which will make the upscaling take longer.

            **In short: For higher quality, try increasing `Patch Size` and/or decreasing `Patch Stride`, but be mindful of VRAM usage and processing time.**
            """
        )
        
    # --- Event Listeners ---

    run_single.click(
        fn=process,
        inputs=[image, prompt, upscale, patch_size, stride, seed, autosave_toggle, caption_detail_toggle, prompt_suffix],
        outputs=[result_slider, status, image_state],
    )
    autosave_toggle.change(
        fn=toggle_autosave,
        inputs=autosave_toggle,
        outputs=manual_save_button,
    )
    manual_save_button.click(
        fn=lambda img, msg: save_image(img, msg)[0],
        inputs=[image_state, status],
        outputs=status,
    )
    open_folder_button.click(fn=open_output_folder)

    # Batch Process Listener
    run_batch.click(
        fn=batch_process,
        inputs=[batch_files, prompt, upscale, patch_size, stride, seed, caption_detail_toggle, prompt_suffix],
        outputs=[status]
    )
    
    if isinstance(captioner, Florence2Captioner):
        keep_captioner_loaded_toggle.change(
            fn=captioner.set_keep_loaded,
            inputs=keep_captioner_loaded_toggle,
            outputs=None, # This action doesn't directly update a UI component
        )

    open_folder_button.click(fn=open_output_folder)

# --- Pre-download and cache the Florence-2 model if selected ---
if isinstance(captioner, Florence2Captioner):
    captioner.prime_model_cache()
    
block.launch(server_name="0.0.0.0" if not args.local else "127.0.0.1", server_port=args.port)
