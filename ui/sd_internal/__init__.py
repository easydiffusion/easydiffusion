import json

class Request:
    session_id: str = "session"
    prompt: str = ""
    negative_prompt: str = ""
    init_image: str = None # base64
    mask: str = None # base64
    num_outputs: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: int = 42
    prompt_strength: float = 0.8
    sampler: str = None # "ddim", "plms", "heun", "euler", "euler_a", "dpm2", "dpm2_a", "lms"
    # allow_nsfw: bool = False
    precision: str = "autocast" # or "full"
    save_to_disk_path: str = None
    turbo: bool = True
    use_full_precision: bool = False
    use_face_correction: str = None # or "GFPGANv1.3"
    use_upscale: str = None # or "RealESRGAN_x4plus" or "RealESRGAN_x4plus_anime_6B"
    use_stable_diffusion_model: str = "sd-v1-4"
    use_vae_model: str = None
    use_hypernetwork_model: str = None
    hypernetwork_strength: float = 1
    show_only_filtered_image: bool = False
    output_format: str = "jpeg" # or "png"
    output_quality: int = 75

    stream_progress_updates: bool = False
    stream_image_progress: bool = False

    def json(self):
        return {
            "session_id": self.session_id,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "num_outputs": self.num_outputs,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "prompt_strength": self.prompt_strength,
            "sampler": self.sampler,
            "use_face_correction": self.use_face_correction,
            "use_upscale": self.use_upscale,
            "use_stable_diffusion_model": self.use_stable_diffusion_model,
            "use_vae_model": self.use_vae_model,
            "use_hypernetwork_model": self.use_hypernetwork_model,
            "hypernetwork_strength": self.hypernetwork_strength,
            "output_format": self.output_format,
            "output_quality": self.output_quality,
        }

    def __str__(self):
        return f'''
    session_id: {self.session_id}
    prompt: {self.prompt}
    negative_prompt: {self.negative_prompt}
    seed: {self.seed}
    num_inference_steps: {self.num_inference_steps}
    sampler: {self.sampler}
    guidance_scale: {self.guidance_scale}
    w: {self.width}
    h: {self.height}
    precision: {self.precision}
    save_to_disk_path: {self.save_to_disk_path}
    turbo: {self.turbo}
    use_full_precision: {self.use_full_precision}
    use_face_correction: {self.use_face_correction}
    use_upscale: {self.use_upscale}
    use_stable_diffusion_model: {self.use_stable_diffusion_model}
    use_vae_model: {self.use_vae_model}
    use_hypernetwork_model: {self.use_hypernetwork_model}
    hypernetwork_strength: {self.hypernetwork_strength}
    show_only_filtered_image: {self.show_only_filtered_image}
    output_format: {self.output_format}
    output_quality: {self.output_quality}

    stream_progress_updates: {self.stream_progress_updates}
    stream_image_progress: {self.stream_image_progress}'''

class Image:
    data: str # base64
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
    request: Request
    images: list

    def json(self):
        res = {
            "status": 'succeeded',
            "request": self.request.json(),
            "output": [],
        }

        for image in self.images:
            res["output"].append(image.json())

        return res
