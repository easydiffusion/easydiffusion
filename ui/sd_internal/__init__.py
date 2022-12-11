from pydantic import BaseModel

from modules.types import GenerateImageRequest

class TaskData(BaseModel):
    request_id: str = None
    session_id: str = "session"
    save_to_disk_path: str = None
    turbo: bool = True
    use_face_correction: str = None # or "GFPGANv1.3"
    use_upscale: str = None # or "RealESRGAN_x4plus" or "RealESRGAN_x4plus_anime_6B"
    use_stable_diffusion_model: str = "sd-v1-4"
    use_vae_model: str = None
    use_hypernetwork_model: str = None
    show_only_filtered_image: bool = False
    output_format: str = "jpeg" # or "png"
    output_quality: int = 75
    stream_image_progress: bool = False

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
    render_request: GenerateImageRequest
    task_data: TaskData
    images: list

    def __init__(self, render_request: GenerateImageRequest, task_data: TaskData, images: list):
        self.render_request = render_request
        self.task_data = task_data
        self.images = images

    def json(self):
        res = {
            "status": 'succeeded',
            "render_request": self.render_request.dict(),
            "task_data": self.task_data.dict(),
            "output": [],
        }

        for image in self.images:
            res["output"].append(image.json())

        return res

class UserInitiatedStop(Exception):
    pass
