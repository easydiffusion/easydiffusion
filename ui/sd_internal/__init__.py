import json

class Request:
    prompt: str = ""
    init_image: str = None # base64
    mask: str = None # base64
    num_outputs: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: int = 42
    prompt_strength: float = 0.8
    # allow_nsfw: bool = False
    precision: str = "autocast" # or "full"
    save_to_disk_path: str = None
    turbo: bool = True
    use_cpu: bool = False
    use_full_precision: bool = False
    use_face_correction: str = None # or "GFPGANv1.3"
    use_upscale: str = None # or "RealESRGAN_x4plus" or "RealESRGAN_x4plus_anime_6B"
    show_only_filtered_image: bool = False

    def to_string(self):
        return f'''
    prompt: {self.prompt}
    seed: {self.seed}
    num_inference_steps: {self.num_inference_steps}
    guidance_scale: {self.guidance_scale}
    w: {self.width}
    h: {self.height}
    precision: {self.precision}
    save_to_disk_path: {self.save_to_disk_path}
    turbo: {self.turbo}
    use_cpu: {self.use_cpu}
    use_full_precision: {self.use_full_precision}
    use_face_correction: {self.use_face_correction}
    use_upscale: {self.use_upscale}
    show_only_filtered_image: {self.show_only_filtered_image}'''

class Image:
    data: str # base64
    seed: int
    is_nsfw: bool

    def __init__(self, data, seed):
        self.data = data
        self.seed = seed

    def json(self):
        return {
            "data": self.data,
            "seed": self.seed,
        }

class Response:
    images: list

    def json(self):
        res = {
            "status": 'succeeded',
            "output": [],
        }

        for image in self.images:
            res["output"].append(image.json())

        return res
