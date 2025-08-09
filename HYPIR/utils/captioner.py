from typing import overload, Literal
import re
import base64
from io import BytesIO
import gc

from PIL import Image
import torch
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

# Conditional import for transformers, only if needed
try:
    from transformers import AutoModelForCausalLM, AutoProcessor
except ImportError:
    AutoModelForCausalLM = None
    AutoProcessor = None


class Captioner:

    def __init__(self, device: torch.device) -> "Captioner":
        self.device = device
        # A flag to check if the captioner is functional
        self.is_functional = True

    @overload
    def __call__(self, image: Image.Image) -> str: ...


class EmptyCaptioner(Captioner):

    def __init__(self, device: torch.device) -> "EmptyCaptioner":
        super().__init__(device)
        self.is_functional = False

    def __call__(self, image: Image.Image) -> str:
        return ""


class Florence2Captioner(Captioner):
    """
    A captioner that uses Florence-2 and can either dynamically load/unload the model
    or keep it in memory to conserve VRAM for other processes.
    """
    MODEL_ID = "microsoft/Florence-2-large"
    DETAILED_PROMPT = "<DETAILED_CAPTION>"
    MORE_DETAILED_PROMPT = "<MORE_DETAILED_CAPTION>"

    def __init__(self, device: str):
        super().__init__(device)
        if AutoModelForCausalLM is None:
            raise ImportError("Please install `transformers` and `einops` to use Florence2Captioner: pip install transformers einops")
        
        self.model = None
        self.processor = None
        self.keep_loaded = False
        print("Florence-2 Captioner initialized.")

    def _load_model(self):
        if self.model is None:
            print("Loading Florence-2 model into memory...")
            # This now assumes the model is cached and will load locally.
            model_args = {"torch_dtype": torch.float16, "trust_remote_code": True}
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID, **model_args, local_files_only=True
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(
                self.MODEL_ID, trust_remote_code=True, local_files_only=True
            )
            print("Florence-2 model loaded.")

    def _unload_model(self):
        if self.model is not None:
            print("Unloading Florence-2 model from memory...")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Florence-2 model unloaded and memory cache cleared.")
    
    def prime_model_cache(self):
        """
        Checks if the model is cached locally. If not, it downloads it.
        This should be called at startup to avoid a download during user interaction.
        """
        print("\nChecking for Florence-2 model cache...")
        try:
            # Try to load from local cache first. This is a fast check.
            _ = AutoProcessor.from_pretrained(self.MODEL_ID, trust_remote_code=True, local_files_only=True)
            print("Florence-2 model is already cached locally.")
        except (OSError, ValueError):
            # If it fails, the model is not cached.
            print("-----------------------------------------------------------------------")
            print("Florence-2 model not found locally.")
            print("Starting a one-time download of the model (~2 GB).")
            print("The UI will launch after the download is complete. Please be patient.")
            print("-----------------------------------------------------------------------")
            model_args = {"torch_dtype": torch.float16, "trust_remote_code": True}
            AutoModelForCausalLM.from_pretrained(self.MODEL_ID, **model_args)
            AutoProcessor.from_pretrained(self.MODEL_ID, trust_remote_code=True)
            print("Download complete. Florence-2 model is now cached.")
        finally:
            print("Cache check complete. The app is ready to launch.\n")

    def set_keep_loaded(self, keep_loaded: bool):
        self.keep_loaded = keep_loaded
        if self.keep_loaded:
            print("Captioner policy set to 'Keep Loaded'. Loading model now...")
            self._load_model()
        else:
            print("Captioner policy set to 'On-Demand'. Unloading model now...")
            self._unload_model()

    def __call__(self, image: Image.Image, use_more_detailed: bool = True) -> str:
        if self.model is None and self.processor is None:
            self._load_model()
        
        try:
            task_prompt = self.MORE_DETAILED_PROMPT if use_more_detailed else self.DETAILED_PROMPT
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
            
            input_ids = inputs["input_ids"].to(self.device)
            pixel_values = inputs["pixel_values"].to(self.device, torch.float16)
            
            generated_ids = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            caption_result = self.processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(image.width, image.height)
            )
            return caption_result.get(task_prompt, "Caption generation failed.")
        finally:
            if not self.keep_loaded:
                self._unload_model()


class GPTCaptioner(Captioner):
    # This class remains unchanged for users who still want to use it.
    DEFAULT_PROMPT = "Provide a detailed description of this image without exceeding 100 words and without line breaks."

    def __init__(self, api_key, base_url, model):
        super().__init__(None) # GPT captioner doesn't use a local torch device
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    @staticmethod
    def pil_image_to_base64(image, format="PNG"):
        buffered = BytesIO()
        image.save(buffered, format=format)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return img_base64

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
    def get_response(self, base64_image, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            stream=False,
        )
        return response.choices[0].message.content

    def __call__(self, image, prompt=None):
        base64_image = self.pil_image_to_base64(image)
        if prompt is None:
            prompt = self.DEFAULT_PROMPT
        caption = self.get_response(base64_image, prompt=prompt)
        return caption