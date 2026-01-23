import os
from glob import glob
from typing import List, Dict, Optional

from easydiffusion.utils.model_identifier import identify_model_type

KNOWN_MODEL_TYPES = [
    "stable-diffusion",
    "vae",
    "hypernetwork",
    "gfpgan",
    "realesrgan",
    "lora",
    "codeformer",
    "embeddings",
    "controlnet",
    "text-encoder",
]

MODEL_EXTENSIONS = {
    "stable-diffusion": [".ckpt", ".safetensors", ".sft", ".gguf"],
    "vae": [".vae.pt", ".ckpt", ".safetensors", ".sft", ".gguf"],
    "hypernetwork": [".pt", ".safetensors", ".sft"],
    "gfpgan": [".pth"],
    "realesrgan": [".pth"],
    "lora": [".ckpt", ".safetensors", ".sft", ".pt"],
    "codeformer": [".pth"],
    "embeddings": [".pt", ".bin", ".safetensors", ".sft"],
    "controlnet": [".pth", ".safetensors", ".sft"],
    "text-encoder": [".safetensors", ".sft", ".gguf"],
}

ALTERNATE_FOLDER_NAMES = {  # for WebUI compatibility
    "stable-diffusion": "Stable-diffusion",
    "vae": "VAE",
    "hypernetwork": "hypernetworks",
    "codeformer": "Codeformer",
    "gfpgan": "GFPGAN",
    "realesrgan": "RealESRGAN",
    "lora": "Lora",
    "controlnet": "ControlNet",
    "text-encoder": "text_encoder",
}


def get_model_dirs(model_type: str, models_dir: str) -> List[str]:
    """Get possible model directory paths for the given model type, including alternates."""
    dirs = [os.path.join(models_dir, model_type)]

    if model_type in ALTERNATE_FOLDER_NAMES:
        alt_dir = ALTERNATE_FOLDER_NAMES[model_type]
        alt_dir_path = os.path.join(models_dir, alt_dir)
        if os.path.exists(alt_dir_path):
            # Check if it's different from the standard dir (to avoid duplicates on case-insensitive filesystems)
            try:
                if not os.path.samefile(dirs[0], alt_dir_path):
                    dirs.append(alt_dir_path)
            except OSError:
                # If samefile fails, assume they are different
                dirs.append(alt_dir_path)

    return dirs


def list_models(model_type: str, models_dir: str) -> List[Dict[str, str]]:
    """List all models of a given type in the model directories, including alternates."""
    model_dirs = get_model_dirs(model_type, models_dir)
    extensions = MODEL_EXTENSIONS.get(model_type, [])
    models = []

    for model_dir in model_dirs:
        if not os.path.exists(model_dir):
            continue

        for ext in extensions:
            pattern = os.path.join(model_dir, f"**/*{ext}")
            files = glob(pattern, recursive=True)
            for file_path in files:
                rel_path = os.path.relpath(file_path, model_dir)
                model_name = os.path.splitext(rel_path)[0]  # Remove extension
                model = {
                    "model": model_name,
                    "name": model_name,
                    "path": file_path,
                    "tags": [model_type],
                }
                if model_type == "stable-diffusion":
                    try:
                        sd_model_class = identify_model_type(file_path)
                        if sd_model_class:
                            model["tags"].append(sd_model_class)
                    except Exception:
                        pass  # Silently ignore if identification fails
                models.append(model)

    return models


def enumerate_all_models(models_dir: str) -> List[Dict[str, str]]:
    """Enumerate all models across all known model types, including alternates."""
    all_models = []
    for model_type in KNOWN_MODEL_TYPES:
        all_models.extend(list_models(model_type, models_dir))
    return all_models


def resolve_model_path(model_name: str, model_type: str, models_dir: str) -> Optional[str]:
    """Resolve a model name to its full file path, checking standard and alternate directories."""
    model_dirs = get_model_dirs(model_type, models_dir)
    extensions = MODEL_EXTENSIONS.get(model_type, [])

    for model_dir in model_dirs:
        if not os.path.exists(model_dir):
            continue

        # First, check if model_name already includes an extension
        for ext in extensions:
            if model_name.endswith(ext):
                full_path = os.path.join(model_dir, model_name)
                if os.path.exists(full_path):
                    return full_path
                break  # If it has an extension but doesn't match, don't add another

        # Otherwise, try adding extensions
        for ext in extensions:
            full_path = os.path.join(model_dir, model_name + ext)
            if os.path.exists(full_path):
                return full_path

    return None
