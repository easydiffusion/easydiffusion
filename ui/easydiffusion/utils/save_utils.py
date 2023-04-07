import os
import time
import re

from easydiffusion import app
from easydiffusion.types import TaskData, GenerateImageRequest
from functools import reduce
from datetime import datetime

from sdkit.utils import save_images, save_dicts
from numpy import base_repr

filename_regex = re.compile("[^a-zA-Z0-9._-]")
img_number_regex = re.compile("([0-9]{5,})")

# keep in sync with `ui/media/js/dnd.js`
TASK_TEXT_MAPPING = {
    "prompt": "Prompt",
    "width": "Width",
    "height": "Height",
    "seed": "Seed",
    "num_inference_steps": "Steps",
    "guidance_scale": "Guidance Scale",
    "prompt_strength": "Prompt Strength",
    "use_face_correction": "Use Face Correction",
    "use_upscale": "Use Upscaling",
    "upscale_amount": "Upscale By",
    "sampler_name": "Sampler",
    "negative_prompt": "Negative Prompt",
    "use_stable_diffusion_model": "Stable Diffusion model",
    "use_vae_model": "VAE model",
    "use_hypernetwork_model": "Hypernetwork model",
    "hypernetwork_strength": "Hypernetwork Strength",
    "use_lora_model": "LoRA model",
    "lora_alpha": "LoRA Strength",
}

time_placeholders = {
    "$yyyy": "%Y",
    "$MM": "%m",
    "$dd": "%d",
    "$HH": "%H",
    "$mm": "%M",
    "$ss": "%S",
}

other_placeholders = {
    "$id": lambda req, task_data: filename_regex.sub("_", task_data.session_id),
    "$p": lambda req, task_data: filename_regex.sub("_", req.prompt)[:50],
    "$s": lambda req, task_data: str(req.seed),
}

class ImageNumber:
    _factory = None
    _evaluated = False

    def __init__(self, factory):
        self._factory = factory
        self._evaluated = None
    def __call__(self) -> int:
        if self._evaluated is None:
            self._evaluated = self._factory()
        return self._evaluated

def format_placeholders(format: str, req: GenerateImageRequest, task_data: TaskData, now = None):
    if now is None:
        now = time.time()

    for placeholder, time_format in time_placeholders.items():
        if placeholder in format:
            format = format.replace(placeholder, datetime.fromtimestamp(now).strftime(time_format))
    for placeholder, replace_func in other_placeholders.items():
        if placeholder in format:
            format = format.replace(placeholder, replace_func(req, task_data))

    return format

def format_folder_name(format: str, req: GenerateImageRequest, task_data: TaskData):
    format = format_placeholders(format, req, task_data)
    return filename_regex.sub("_", format)

def format_file_name(
    format: str,
    req: GenerateImageRequest,
    task_data: TaskData,
    now: float,
    batch_file_number: int,
    folder_img_number: ImageNumber,
):
    format = format_placeholders(format, req, task_data, now)
    
    if "$n" in format:
        format = format.replace("$n", f"{folder_img_number():05}")
    
    if "$tsb64" in format:
        img_id = base_repr(int(now * 10000), 36)[-7:] + base_repr(int(batch_file_number), 36) # Base 36 conversion, 0-9, A-Z
        format = format.replace("$tsb64", img_id)
    
    if "$ts" in format:
        format = format.replace("$ts", str(int(now * 1000) + batch_file_number))

    return filename_regex.sub("_", format)

def save_images_to_disk(images: list, filtered_images: list, req: GenerateImageRequest, task_data: TaskData):
    now = time.time()
    app_config = app.getConfig()
    folder_format = app_config.get("folder_format", "$id")
    save_dir_path = os.path.join(task_data.save_to_disk_path, format_folder_name(folder_format, req, task_data))
    metadata_entries = get_metadata_entries_for_request(req, task_data)
    file_number = calculate_img_number(save_dir_path, task_data)
    make_filename = make_filename_callback(
        app_config.get("filename_format", "$p_$tsb64"),
        req,
        task_data,
        file_number,
        now=now,
    )

    if task_data.show_only_filtered_image or filtered_images is images:
        save_images(
            filtered_images,
            save_dir_path,
            file_name=make_filename,
            output_format=task_data.output_format,
            output_quality=task_data.output_quality,
            output_lossless=task_data.output_lossless,
        )
        if task_data.metadata_output_format:
            for metadata_output_format in task_data.metadata_output_format.split(','):
                if metadata_output_format.lower() in ["json", "txt", "embed"]:
                    save_dicts(
                        metadata_entries,
                        save_dir_path,
                        file_name=make_filename,
                        output_format=metadata_output_format,
                        file_format=task_data.output_format,
                    )
    else:
        make_filter_filename = make_filename_callback(
            app_config.get("filename_format", "$p_$tsb64"),
            req,
            task_data,
            file_number,
            now=now,
            suffix="filtered")

        save_images(
            images,
            save_dir_path,
            file_name=make_filename,
            output_format=task_data.output_format,
            output_quality=task_data.output_quality,
            output_lossless=task_data.output_lossless,
        )
        save_images(
            filtered_images,
            save_dir_path,
            file_name=make_filter_filename,
            output_format=task_data.output_format,
            output_quality=task_data.output_quality,
            output_lossless=task_data.output_lossless,
        )
        if task_data.metadata_output_format.lower() in ["json", "txt", "embed"]:
            save_dicts(
                metadata_entries,
                save_dir_path,
                file_name=make_filter_filename,
                output_format=task_data.metadata_output_format,
                file_format=task_data.output_format,
            )


def get_metadata_entries_for_request(req: GenerateImageRequest, task_data: TaskData):
    metadata = get_printable_request(req)
    metadata.update(
        {
            "use_stable_diffusion_model": task_data.use_stable_diffusion_model,
            "use_vae_model": task_data.use_vae_model,
            "use_hypernetwork_model": task_data.use_hypernetwork_model,
            "use_lora_model": task_data.use_lora_model,
            "use_face_correction": task_data.use_face_correction,
            "use_upscale": task_data.use_upscale,
        }
    )
    if metadata["use_upscale"] is not None:
        metadata["upscale_amount"] = task_data.upscale_amount
    if task_data.use_hypernetwork_model is None:
        del metadata["hypernetwork_strength"]
    if task_data.use_lora_model is None:
        if "lora_alpha" in metadata:
            del metadata["lora_alpha"]
        app_config = app.getConfig()
        if not app_config.get("test_diffusers", False) and "use_lora_model" in metadata:
            del metadata["use_lora_model"]

    # if text, format it in the text format expected by the UI
    is_txt_format = task_data.metadata_output_format.lower() == "txt"
    if is_txt_format:
        metadata = {TASK_TEXT_MAPPING[key]: val for key, val in metadata.items() if key in TASK_TEXT_MAPPING}

    entries = [metadata.copy() for _ in range(req.num_outputs)]
    for i, entry in enumerate(entries):
        entry["Seed" if is_txt_format else "seed"] = req.seed + i

    return entries


def get_printable_request(req: GenerateImageRequest):
    metadata = req.dict()
    del metadata["init_image"]
    del metadata["init_image_mask"]
    if req.init_image is None:
        del metadata["prompt_strength"]
    return metadata


def make_filename_callback(
    filename_format: str,
    req: GenerateImageRequest,
    task_data: TaskData,
    folder_img_number: int,
    suffix=None,
    now=None,
):
    if now is None:
        now = time.time()

    def make_filename(i):
        name = format_file_name(filename_format, req, task_data, now, i, folder_img_number)
        name = name if suffix is None else f"{name}_{suffix}"

        return name

    return make_filename

def _calculate_img_number(save_dir_path: str, task_data: TaskData):
    def get_highest_img_number(accumulator: int, file: os.DirEntry) -> int:
        if not file.is_file:
            return accumulator
        
        if len(list(filter(lambda e: file.name.endswith(e), app.IMAGE_EXTENSIONS))) == 0:
            return accumulator
        
        get_highest_img_number.number_of_images = get_highest_img_number.number_of_images + 1
        
        number_match = img_number_regex.match(file.name)
        if not number_match:
            return accumulator
        
        file_number = number_match.group().lstrip('0')
        
        # Handle 00000
        return int(file_number) if file_number else 0
    
    get_highest_img_number.number_of_images = 0
    
    highest_file_number = -1

    if os.path.isdir(save_dir_path):
        existing_files = list(os.scandir(save_dir_path))
        highest_file_number = reduce(get_highest_img_number, existing_files, -1)

    calculated_img_number = max(highest_file_number, get_highest_img_number.number_of_images - 1)

    if task_data.session_id in _calculate_img_number.session_img_numbers:
        calculated_img_number = max(
            _calculate_img_number.session_img_numbers[task_data.session_id],
            calculated_img_number,
        )
    
    calculated_img_number = calculated_img_number + 1
    
    _calculate_img_number.session_img_numbers[task_data.session_id] = calculated_img_number
    return calculated_img_number

_calculate_img_number.session_img_numbers = {}

def calculate_img_number(save_dir_path: str, task_data: TaskData):
    return ImageNumber(lambda: _calculate_img_number(save_dir_path, task_data))
