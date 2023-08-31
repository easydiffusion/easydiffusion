from typing import Any, List, Dict, Union

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
    control_image: Any = None
    control_alpha: Union[float, List[float]] = None
    prompt_strength: float = 0.8
    preserve_init_image_color_profile = False
    strict_mask_border = False

    sampler_name: str = None  # "ddim", "plms", "heun", "euler", "euler_a", "dpm2", "dpm2_a", "lms"
    hypernetwork_strength: float = 0
    lora_alpha: Union[float, List[float]] = 0
    tiling: str = None  # None, "x", "y", "xy"


class FilterImageRequest(BaseModel):
    image: Any = None
    filter: Union[str, List[str]] = None
    filter_params: dict = {}


class ModelsData(BaseModel):
    """
    Contains the information related to the models involved in a request.

    - To load a model: set the relative path(s) to the model in `model_paths`. No effect if already loaded.
    - To unload a model: set the model to `None` in `model_paths`. No effect if already unloaded.

    Models that aren't present in `model_paths` will not be changed.
    """

    model_paths: Dict[str, Union[str, None, List[str]]] = None
    "model_type to string path, or list of string paths"

    model_params: Dict[str, Dict[str, Any]] = {}
    "model_type to dict of parameters"


class OutputFormatData(BaseModel):
    output_format: str = "jpeg"  # or "png" or "webp"
    output_quality: int = 75
    output_lossless: bool = False


class SaveToDiskData(BaseModel):
    save_to_disk_path: str = None
    metadata_output_format: str = "txt"  # or "json"


class TaskData(BaseModel):
    request_id: str = None
    session_id: str = "session"


class RenderTaskData(TaskData):
    vram_usage_level: str = "balanced"  # or "low" or "medium"

    use_face_correction: Union[str, List[str]] = None  # or "GFPGANv1.3"
    use_upscale: Union[str, List[str]] = None
    upscale_amount: int = 4  # or 2
    latent_upscaler_steps: int = 10
    use_stable_diffusion_model: Union[str, List[str]] = "sd-v1-4"
    use_vae_model: Union[str, List[str]] = None
    use_hypernetwork_model: Union[str, List[str]] = None
    use_lora_model: Union[str, List[str]] = None
    use_controlnet_model: Union[str, List[str]] = None
    use_embeddings_model: Union[str, List[str]] = None
    filters: List[str] = []
    filter_params: Dict[str, Dict[str, Any]] = {}
    control_filter_to_apply: Union[str, List[str]] = None

    show_only_filtered_image: bool = False
    block_nsfw: bool = False
    stream_image_progress: bool = False
    stream_image_progress_interval: int = 5
    clip_skip: bool = False
    codeformer_upscale_faces: bool = False
    codeformer_fidelity: float = 0.5


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


class GenerateImageResponse:
    render_request: GenerateImageRequest
    task_data: TaskData
    models_data: ModelsData
    images: list

    def __init__(
        self,
        render_request: GenerateImageRequest,
        task_data: TaskData,
        models_data: ModelsData,
        output_format: OutputFormatData,
        save_data: SaveToDiskData,
        images: list,
    ):
        self.render_request = render_request
        self.task_data = task_data
        self.models_data = models_data
        self.output_format = output_format
        self.save_data = save_data
        self.images = images

    def json(self):
        del self.render_request.init_image
        del self.render_request.init_image_mask
        del self.render_request.control_image

        task_data = self.task_data.dict()
        task_data.update(self.output_format.dict())
        task_data.update(self.save_data.dict())

        res = {
            "status": "succeeded",
            "render_request": self.render_request.dict(),
            "task_data": task_data,
            # "models_data": self.models_data.dict(), # haven't migrated the UI to the new format (yet)
            "output": [],
        }

        for image in self.images:
            res["output"].append(image.json())

        return res


class FilterImageResponse:
    request: FilterImageRequest
    models_data: ModelsData
    images: list

    def __init__(self, request: FilterImageRequest, models_data: ModelsData, images: list):
        self.request = request
        self.models_data = models_data
        self.images = images

    def json(self):
        del self.request.image

        res = {
            "status": "succeeded",
            "request": self.request.dict(),
            "models_data": self.models_data.dict(),
            "output": [],
        }

        for image in self.images:
            res["output"].append(image)

        return res


class UserInitiatedStop(Exception):
    pass


def convert_legacy_render_req_to_new(old_req: dict):
    new_req = dict(old_req)

    # new keys
    model_paths = new_req["model_paths"] = {}
    model_params = new_req["model_params"] = {}
    filters = new_req["filters"] = []
    filter_params = new_req["filter_params"] = {}

    # move the model info
    model_paths["stable-diffusion"] = old_req.get("use_stable_diffusion_model")
    model_paths["vae"] = old_req.get("use_vae_model")
    model_paths["hypernetwork"] = old_req.get("use_hypernetwork_model")
    model_paths["lora"] = old_req.get("use_lora_model")
    model_paths["controlnet"] = old_req.get("use_controlnet_model")
    model_paths["embeddings"] = old_req.get("use_embeddings_model")

    model_paths["gfpgan"] = old_req.get("use_face_correction", "")
    model_paths["gfpgan"] = model_paths["gfpgan"] if "gfpgan" in model_paths["gfpgan"].lower() else None

    model_paths["codeformer"] = old_req.get("use_face_correction", "")
    model_paths["codeformer"] = model_paths["codeformer"] if "codeformer" in model_paths["codeformer"].lower() else None

    model_paths["realesrgan"] = old_req.get("use_upscale", "")
    model_paths["realesrgan"] = model_paths["realesrgan"] if "realesrgan" in model_paths["realesrgan"].lower() else None

    model_paths["latent_upscaler"] = old_req.get("use_upscale", "")
    model_paths["latent_upscaler"] = (
        model_paths["latent_upscaler"] if "latent_upscaler" in model_paths["latent_upscaler"].lower() else None
    )
    if "control_filter_to_apply" in old_req:
        filter_model = old_req["control_filter_to_apply"]
        model_paths[filter_model] = filter_model

    if old_req.get("block_nsfw"):
        model_paths["nsfw_checker"] = "nsfw_checker"

    # move the model params
    if model_paths["stable-diffusion"]:
        model_params["stable-diffusion"] = {
            "clip_skip": bool(old_req.get("clip_skip", False)),
            "convert_to_tensorrt": bool(old_req.get("convert_to_tensorrt", False)),
            "trt_build_config": old_req.get(
                "trt_build_config", {"batch_size_range": (1, 1), "dimensions_range": [(768, 1024)]}
            ),
        }

    # move the filter params
    if model_paths["realesrgan"]:
        filter_params["realesrgan"] = {"scale": int(old_req.get("upscale_amount", 4))}
    if model_paths["latent_upscaler"]:
        filter_params["latent_upscaler"] = {
            "prompt": old_req["prompt"],
            "negative_prompt": old_req.get("negative_prompt"),
            "seed": int(old_req.get("seed", 42)),
            "num_inference_steps": int(old_req.get("latent_upscaler_steps", 10)),
            "guidance_scale": 0,
        }
    if model_paths["codeformer"]:
        filter_params["codeformer"] = {
            "upscale_faces": bool(old_req.get("codeformer_upscale_faces", True)),
            "codeformer_fidelity": float(old_req.get("codeformer_fidelity", 0.5)),
        }

    # set the filters
    if old_req.get("block_nsfw"):
        filters.append("nsfw_checker")

    if model_paths["codeformer"]:
        filters.append("codeformer")
    elif model_paths["gfpgan"]:
        filters.append("gfpgan")

    if model_paths["realesrgan"]:
        filters.append("realesrgan")
    elif model_paths["latent_upscaler"]:
        filters.append("latent_upscaler")

    return new_req
