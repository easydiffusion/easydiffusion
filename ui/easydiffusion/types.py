from typing import Any

from pydantic import BaseModel


class GenerateImageRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""

    seed: int = 42
    width: int = 512
    height: int = 512

    num_outputs: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    init_image: Any = None
    init_image_mask: Any = None
    prompt_strength: float = 0.8
    preserve_init_image_color_profile = False

    sampler_name: str = None  # "ddim", "plms", "heun", "euler", "euler_a", "dpm2", "dpm2_a", "lms"
    hypernetwork_strength: float = 0
    lora_alpha: float = 0


class TaskData(BaseModel):
    request_id: str = None
    session_id: str = "session"
    save_to_disk_path: str = None
    vram_usage_level: str = "balanced"  # or "low" or "medium"

    use_face_correction: str = None  # or "GFPGANv1.3"
    use_upscale: str = None  # or "RealESRGAN_x4plus" or "RealESRGAN_x4plus_anime_6B" or "latent_upscaler"
    upscale_amount: int = 4  # or 2
    latent_upscaler_steps: int = 10
    use_stable_diffusion_model: str = "sd-v1-4"
    # use_stable_diffusion_config: str = "v1-inference"
    use_vae_model: str = None
    use_hypernetwork_model: str = None
    use_lora_model: str = None

    show_only_filtered_image: bool = False
    block_nsfw: bool = False
    output_format: str = "jpeg"  # or "png" or "webp"
    output_quality: int = 75
    output_lossless: bool = False
    metadata_output_format: str = "txt"  # or "json"
    stream_image_progress: bool = False
    stream_image_progress_interval: int = 5
    clip_skip: bool = False


class MergeRequest(BaseModel):
    model0: str = None
    model1: str = None
    ratio: float = None
    out_path: str = "mix"
    use_fp16 = True


class Image:
    data: str  # base64
    seed: int
    is_nsfw: bool
    path_abs: str = None

    def __init__(self, data, seed):
        self.data = data
        self.seed = seed

    def json(self):
        return {
            "data": self.data,
            "seed": self.seed,
            "path_abs": self.path_abs,
        }


class Response:
    render_request: GenerateImageRequest
    task_data: TaskData
    images: list

    def __init__(self, render_request: GenerateImageRequest, task_data: TaskData, images: list):
        self.render_request = render_request
        self.task_data = task_data
        self.images = images

    def json(self):
        del self.render_request.init_image
        del self.render_request.init_image_mask

        res = {
            "status": "succeeded",
            "render_request": self.render_request.dict(),
            "task_data": self.task_data.dict(),
            "output": [],
        }

        for image in self.images:
            res["output"].append(image.json())

        return res


class UserInitiatedStop(Exception):
    pass
