import os
import shutil
from glob import glob
import traceback

from easydiffusion import app
from easydiffusion.types import TaskData
from easydiffusion.utils import log
from sdkit import Context
from sdkit.models import load_model, scan_model, unload_model, download_model, get_model_info_from_db
from sdkit.utils import hash_file_quick

KNOWN_MODEL_TYPES = [
    "stable-diffusion",
    "vae",
    "hypernetwork",
    "gfpgan",
    "realesrgan",
    "lora",
    "codeformer",
]
MODEL_EXTENSIONS = {
    "stable-diffusion": [".ckpt", ".safetensors"],
    "vae": [".vae.pt", ".ckpt", ".safetensors"],
    "hypernetwork": [".pt", ".safetensors"],
    "gfpgan": [".pth"],
    "realesrgan": [".pth"],
    "lora": [".ckpt", ".safetensors"],
    "codeformer": [".pth"],
}
DEFAULT_MODELS = {
    "stable-diffusion": [
        {"file_name": "sd-v1-4.ckpt", "model_id": "1.4"},
    ],
    "gfpgan": [
        {"file_name": "GFPGANv1.4.pth", "model_id": "1.4"},
    ],
    "realesrgan": [
        {"file_name": "RealESRGAN_x4plus.pth", "model_id": "x4plus"},
        {"file_name": "RealESRGAN_x4plus_anime_6B.pth", "model_id": "x4plus_anime_6"},
    ],
    "vae": [
        {"file_name": "vae-ft-mse-840000-ema-pruned.ckpt", "model_id": "vae-ft-mse-840000-ema-pruned"},
    ],
}
MODELS_TO_LOAD_ON_START = ["stable-diffusion", "vae", "hypernetwork", "lora"]

known_models = {}


def init():
    make_model_folders()
    migrate_legacy_model_location()  # if necessary
    download_default_models_if_necessary()
    getModels()  # run this once, to cache the picklescan results


def load_default_models(context: Context):
    set_vram_optimizations(context)

    # init default model paths
    for model_type in MODELS_TO_LOAD_ON_START:
        context.model_paths[model_type] = resolve_model_to_use(model_type=model_type)
        try:
            load_model(
                context,
                model_type,
                scan_model=context.model_paths[model_type] != None
                and not context.model_paths[model_type].endswith(".safetensors"),
            )
            if model_type in context.model_load_errors:
                del context.model_load_errors[model_type]
        except Exception as e:
            log.error(f"[red]Error while loading {model_type} model: {context.model_paths[model_type]}[/red]")
            if "DefaultCPUAllocator: not enough memory" in str(e):
                log.error(
                    f"[red]Your PC is low on system RAM. Please add some virtual memory (or swap space) by following the instructions at this link: https://www.ibm.com/docs/en/opw/8.2.0?topic=tuning-optional-increasing-paging-file-size-windows-computers[/red]"
                )
            else:
                log.exception(e)
            del context.model_paths[model_type]

            context.model_load_errors[model_type] = str(e)  # storing the entire Exception can lead to memory leaks


def unload_all(context: Context):
    for model_type in KNOWN_MODEL_TYPES:
        unload_model(context, model_type)
        if model_type in context.model_load_errors:
            del context.model_load_errors[model_type]


def resolve_model_to_use(model_name: str = None, model_type: str = None):
    model_extensions = MODEL_EXTENSIONS.get(model_type, [])
    default_models = DEFAULT_MODELS.get(model_type, [])
    config = app.getConfig()

    model_dir = os.path.join(app.MODELS_DIR, model_type)
    if not model_name:  # When None try user configured model.
        # config = getConfig()
        if "model" in config and model_type in config["model"]:
            model_name = config["model"][model_type]

    if model_name:
        # Check models directory
        model_path = os.path.join(model_dir, model_name)
        if os.path.exists(model_path):
            return model_path
        for model_extension in model_extensions:
            if os.path.exists(model_path + model_extension):
                return model_path + model_extension
            if os.path.exists(model_name + model_extension):
                return os.path.abspath(model_name + model_extension)

    # Can't find requested model, check the default paths.
    if model_type == "stable-diffusion":
        for default_model in default_models:
            default_model_path = os.path.join(model_dir, default_model["file_name"])
            if os.path.exists(default_model_path):
                if model_name is not None:
                    log.warn(
                        f"Could not find the configured custom model {model_name}. Using the default one: {default_model_path}"
                    )
                return default_model_path

    return None


def reload_models_if_necessary(context: Context, task_data: TaskData):
    face_fix_lower = task_data.use_face_correction.lower() if task_data.use_face_correction else ""
    upscale_lower = task_data.use_upscale.lower() if task_data.use_upscale else ""

    model_paths_in_req = {
        "stable-diffusion": task_data.use_stable_diffusion_model,
        "vae": task_data.use_vae_model,
        "hypernetwork": task_data.use_hypernetwork_model,
        "codeformer": task_data.use_face_correction if "codeformer" in face_fix_lower else None,
        "gfpgan": task_data.use_face_correction if "gfpgan" in face_fix_lower else None,
        "realesrgan": task_data.use_upscale if "realesrgan" in upscale_lower else None,
        "latent_upscaler": True if "latent_upscaler" in upscale_lower else None,
        "nsfw_checker": True if task_data.block_nsfw else None,
        "lora": task_data.use_lora_model,
    }
    models_to_reload = {
        model_type: path
        for model_type, path in model_paths_in_req.items()
        if context.model_paths.get(model_type) != path
    }

    if task_data.codeformer_upscale_faces:
        if "realesrgan" not in models_to_reload and "realesrgan" not in context.models:
            default_realesrgan = DEFAULT_MODELS["realesrgan"][0]["file_name"]
            models_to_reload["realesrgan"] = resolve_model_to_use(default_realesrgan, "realesrgan")
        elif "realesrgan" in models_to_reload and models_to_reload["realesrgan"] is None:
            del models_to_reload["realesrgan"]  # don't unload realesrgan

    if set_vram_optimizations(context) or set_clip_skip(context, task_data):  # reload SD
        models_to_reload["stable-diffusion"] = model_paths_in_req["stable-diffusion"]

    for model_type, model_path_in_req in models_to_reload.items():
        context.model_paths[model_type] = model_path_in_req

        action_fn = unload_model if context.model_paths[model_type] is None else load_model
        try:
            action_fn(context, model_type, scan_model=False)  # we've scanned them already
            if model_type in context.model_load_errors:
                del context.model_load_errors[model_type]
        except Exception as e:
            log.exception(e)
            if action_fn == load_model:
                context.model_load_errors[model_type] = str(e)  # storing the entire Exception can lead to memory leaks


def resolve_model_paths(task_data: TaskData):
    task_data.use_stable_diffusion_model = resolve_model_to_use(
        task_data.use_stable_diffusion_model, model_type="stable-diffusion"
    )
    task_data.use_vae_model = resolve_model_to_use(task_data.use_vae_model, model_type="vae")
    task_data.use_hypernetwork_model = resolve_model_to_use(task_data.use_hypernetwork_model, model_type="hypernetwork")
    task_data.use_lora_model = resolve_model_to_use(task_data.use_lora_model, model_type="lora")

    if task_data.use_face_correction:
        if "gfpgan" in task_data.use_face_correction.lower():
            model_type = "gfpgan"
        elif "codeformer" in task_data.use_face_correction.lower():
            model_type = "codeformer"
            download_if_necessary("codeformer", "codeformer.pth", "codeformer-0.1.0")

        task_data.use_face_correction = resolve_model_to_use(task_data.use_face_correction, model_type)
    if task_data.use_upscale and "realesrgan" in task_data.use_upscale.lower():
        task_data.use_upscale = resolve_model_to_use(task_data.use_upscale, "realesrgan")


def fail_if_models_did_not_load(context: Context):
    for model_type in KNOWN_MODEL_TYPES:
        if model_type in context.model_load_errors:
            e = context.model_load_errors[model_type]
            raise Exception(f"Could not load the {model_type} model! Reason: " + e)


def download_default_models_if_necessary():
    for model_type, models in DEFAULT_MODELS.items():
        for model in models:
            try:
                download_if_necessary(model_type, model["file_name"], model["model_id"])
            except:
                traceback.print_exc()
                app.fail_and_die(fail_type="model_download", data=model_type)

        print(model_type, "model(s) found.")


def download_if_necessary(model_type: str, file_name: str, model_id: str):
    model_path = os.path.join(app.MODELS_DIR, model_type, file_name)
    expected_hash = get_model_info_from_db(model_type=model_type, model_id=model_id)["quick_hash"]

    other_models_exist = any_model_exists(model_type)
    known_model_exists = os.path.exists(model_path)
    known_model_is_corrupt = known_model_exists and hash_file_quick(model_path) != expected_hash

    if known_model_is_corrupt or (not other_models_exist and not known_model_exists):
        print("> download", model_type, model_id)
        download_model(model_type, model_id, download_base_dir=app.MODELS_DIR)


def set_vram_optimizations(context: Context):
    config = app.getConfig()
    vram_usage_level = config.get("vram_usage_level", "balanced")

    if vram_usage_level != context.vram_usage_level:
        context.vram_usage_level = vram_usage_level
        return True

    return False


def migrate_legacy_model_location():
    'Move the models inside the legacy "stable-diffusion" folder, to their respective folders'

    for model_type, models in DEFAULT_MODELS.items():
        for model in models:
            file_name = model["file_name"]
            legacy_path = os.path.join(app.SD_DIR, file_name)
            if os.path.exists(legacy_path):
                shutil.move(legacy_path, os.path.join(app.MODELS_DIR, model_type, file_name))


def any_model_exists(model_type: str) -> bool:
    extensions = MODEL_EXTENSIONS.get(model_type, [])
    for ext in extensions:
        if any(glob(f"{app.MODELS_DIR}/{model_type}/**/*{ext}", recursive=True)):
            return True

    return False


def set_clip_skip(context: Context, task_data: TaskData):
    clip_skip = task_data.clip_skip

    if clip_skip != context.clip_skip:
        context.clip_skip = clip_skip
        return True

    return False


def make_model_folders():
    for model_type in KNOWN_MODEL_TYPES:
        model_dir_path = os.path.join(app.MODELS_DIR, model_type)

        os.makedirs(model_dir_path, exist_ok=True)

        help_file_name = f"Place your {model_type} model files here.txt"
        help_file_contents = f'Supported extensions: {" or ".join(MODEL_EXTENSIONS.get(model_type))}'

        with open(os.path.join(model_dir_path, help_file_name), "w", encoding="utf-8") as f:
            f.write(help_file_contents)


def is_malicious_model(file_path):
    try:
        if file_path.endswith(".safetensors"):
            return False
        scan_result = scan_model(file_path)
        if scan_result.issues_count > 0 or scan_result.infected_files > 0:
            log.warn(
                ":warning: [bold red]Scan %s: %d scanned, %d issue, %d infected.[/bold red]"
                % (
                    file_path,
                    scan_result.scanned_files,
                    scan_result.issues_count,
                    scan_result.infected_files,
                )
            )
            return True
        else:
            log.debug(
                "Scan %s: [green]%d scanned, %d issue, %d infected.[/green]"
                % (
                    file_path,
                    scan_result.scanned_files,
                    scan_result.issues_count,
                    scan_result.infected_files,
                )
            )
            return False
    except Exception as e:
        log.error(f"error while scanning: {file_path}, error: {e}")
    return False


def getModels():
    models = {
        "options": {
            "stable-diffusion": ["sd-v1-4"],
            "vae": [],
            "hypernetwork": [],
            "lora": [],
            "codeformer": ["codeformer"],
        },
    }

    models_scanned = 0

    class MaliciousModelException(Exception):
        "Raised when picklescan reports a problem with a model"

    def scan_directory(directory, suffixes, directoriesFirst: bool = True):
        nonlocal models_scanned
        tree = []
        for entry in sorted(
            os.scandir(directory),
            key=lambda entry: (entry.is_file() == directoriesFirst, entry.name.lower()),
        ):
            if entry.is_file():
                matching_suffix = list(filter(lambda s: entry.name.endswith(s), suffixes))
                if len(matching_suffix) == 0:
                    continue
                matching_suffix = matching_suffix[0]

                mtime = entry.stat().st_mtime
                mod_time = known_models[entry.path] if entry.path in known_models else -1
                if mod_time != mtime:
                    models_scanned += 1
                    if is_malicious_model(entry.path):
                        raise MaliciousModelException(entry.path)
                known_models[entry.path] = mtime
                tree.append(entry.name[: -len(matching_suffix)])
            elif entry.is_dir():
                scan = scan_directory(entry.path, suffixes, directoriesFirst=False)

                if len(scan) != 0:
                    tree.append((entry.name, scan))
        return tree

    def listModels(model_type):
        nonlocal models_scanned

        model_extensions = MODEL_EXTENSIONS.get(model_type, [])
        models_dir = os.path.join(app.MODELS_DIR, model_type)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        try:
            models["options"][model_type] = scan_directory(models_dir, model_extensions)
        except MaliciousModelException as e:
            models["scan-error"] = e

    log.info(f"[green]Scanning all model folders for models...[/]")
    # custom models
    listModels(model_type="stable-diffusion")
    listModels(model_type="vae")
    listModels(model_type="hypernetwork")
    listModels(model_type="gfpgan")
    listModels(model_type="lora")

    if models_scanned > 0:
        log.info(f"[green]Scanned {models_scanned} models. Nothing infected[/]")

    return models
