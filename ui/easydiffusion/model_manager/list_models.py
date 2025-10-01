import os

from sdkit.models.model_loader.embeddings import get_embedding_token
from easydiffusion.utils.model_identifier import identify_model_type

PREFILLED_MODELS = {
    "vae": [
        {"model": "ae", "name": "ae (Flux VAE fp16)", "tags": ["vae"]},
    ],
    "codeformer": [
        {"model": "codeformer", "name": "CodeFormer", "tags": ["codeformer"]},
    ],
    "text-encoder": [
        {"model": "t5xxl_fp16", "name": "T5 XXL fp16", "tags": ["text-encoder"]},
        {"model": "clip_l", "name": "CLIP L", "tags": ["text-encoder"]},
        {"model": "clip_g", "name": "CLIP G", "tags": ["text-encoder"]},
    ],
    "controlnet": [
        # {"model": "control_v11p_sd15_canny", "name": "Canny (*)", "tags": ["controlnet"]},
        # {"model": "control_v11p_sd15_openpose", "name": "OpenPose (*)", "tags": ["controlnet"]},
        # {"model": "control_v11p_sd15_normalbae", "name": "Normal BAE (*)", "tags": ["controlnet"]},
        # {"model": "control_v11f1p_sd15_depth", "name": "Depth (*)", "tags": ["controlnet"]},
        # {"model": "control_v11p_sd15_scribble", "name": "Scribble", "tags": ["controlnet"]},
        # {"model": "control_v11p_sd15_softedge", "name": "Soft Edge", "tags": ["controlnet"]},
        # {"model": "control_v11p_sd15_inpaint", "name": "Inpaint", "tags": ["controlnet"]},
        # {"model": "control_v11p_sd15_lineart", "name": "Line Art", "tags": ["controlnet"]},
        # {"model": "control_v11p_sd15s2_lineart_anime", "name": "Line Art Anime", "tags": ["controlnet"]},
        # {"model": "control_v11p_sd15_mlsd", "name": "Straight Lines", "tags": ["controlnet"]},
        # {"model": "control_v11p_sd15_seg", "name": "Segment", "tags": ["controlnet"]},
        # {"model": "control_v11e_sd15_shuffle", "name": "Shuffle", "tags": ["controlnet"]},
        # {"model": "control_v11f1e_sd15_tile", "name": "Tile", "tags": ["controlnet"]},
    ],
}


def list_files(dir, exts):
    """
    Lists files recursively in a directory filtered by extensions using os.walk().

    Args:
        dir (str): The path to the directory to search.
        exts (list): A list of file extensions (e.g., ['.txt', '.py']).

    Returns:
        list: A list of full file paths matching the criteria.
    """
    found_files = []
    if not os.path.exists(dir):
        return found_files

    for root, _, files in os.walk(dir):
        for file in files:
            if any(file.endswith(ext) for ext in exts):
                found_files.append(os.path.join(root, file))
    return found_files


def list_models_in_dirs(dirs, exts):
    models = []
    for dir in dirs:
        model_paths = list_files(dir, exts)
        model_paths = [p.replace("\\", "/") for p in model_paths]
        model_paths = [{"rel_path": p[len(dir) + 1 :], "abs_path": p} for p in model_paths]

        existing_model_paths = [e["rel_path"] for e in models]
        model_paths = [e for e in model_paths if e["rel_path"] not in existing_model_paths]

        models.extend(model_paths)

    return models


def set_model_metadata(model_type, models):
    for m in models:
        m["model"] = os.path.splitext(m["rel_path"])[0]
        m["name"] = m["model"]

        if model_type == "embeddings":
            dir_name, file_name = os.path.dirname(m["model"]), os.path.basename(m["model"])
            file_name = get_embedding_token(file_name)
            m["model"] = os.path.join(dir_name, file_name).replace("\\", "/")

        if model_type == "gfpgan" and "gfpgan" not in m["model"].lower():
            m["model"] = None  # will get filtered out later

        m["tags"] = [model_type]
        if model_type == "stable-diffusion":
            sd_model_class = identify_model_type(m["abs_path"])
            if sd_model_class:
                m["tags"].append(sd_model_class)


def strip_null_models(models):
    return [m for m in models if m["model"]]


def strip_model_paths(models):
    for m in models:
        del m["abs_path"]
        del m["rel_path"]


def include_prefilled_models(models, prefilled_models):
    model_ids = set(m["model"] for m in models)
    for m in prefilled_models:
        if m["model"] not in model_ids:
            models.append(m)


def list_models():
    from easydiffusion.model_manager import KNOWN_MODEL_TYPES, MODEL_EXTENSIONS, get_model_dirs

    models = []

    for model_type in KNOWN_MODEL_TYPES:
        models_dirs = get_model_dirs(model_type)
        model_extensions = MODEL_EXTENSIONS.get(model_type, [])

        models_in_dirs = list_models_in_dirs(models_dirs, model_extensions)
        set_model_metadata(model_type, models_in_dirs)
        strip_model_paths(models_in_dirs)
        models_in_dirs = strip_null_models(models_in_dirs)
        include_prefilled_models(models_in_dirs, PREFILLED_MODELS.get(model_type, []))

        models.extend(models_in_dirs)

    return models
