import os
import shutil
from glob import glob
import traceback
from typing import Union

from easydiffusion import app
from easydiffusion.types import ModelsData
from easydiffusion.utils import log
from sdkit import Context
from sdkit.models import load_model, scan_model, unload_model, download_model, get_model_info_from_db
from sdkit.models.model_loader.controlnet_filters import filters as cn_filters
from sdkit.utils import hash_file_quick

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
]
MODEL_EXTENSIONS = {
    "stable-diffusion": [".ckpt", ".safetensors"],
    "vae": [".vae.pt", ".ckpt", ".safetensors"],
    "hypernetwork": [".pt", ".safetensors"],
    "gfpgan": [".pth"],
    "realesrgan": [".pth"],
    "lora": [".ckpt", ".safetensors"],
    "codeformer": [".pth"],
    "embeddings": [".pt", ".bin", ".safetensors"],
    "controlnet": [".pth", ".safetensors"],
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


def load_default_models(context: Context):
    from easydiffusion import runtime

    runtime.set_vram_optimizations(context)

    # init default model paths
    for model_type in MODELS_TO_LOAD_ON_START:
        context.model_paths[model_type] = resolve_model_to_use(model_type=model_type, fail_if_not_found=False)
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


def resolve_model_to_use(model_name: Union[str, list] = None, model_type: str = None, fail_if_not_found: bool = True):
    model_names = model_name if isinstance(model_name, list) else [model_name]
    model_paths = [resolve_model_to_use_single(m, model_type, fail_if_not_found) for m in model_names]

    return model_paths[0] if len(model_paths) == 1 else model_paths


def resolve_model_to_use_single(model_name: str = None, model_type: str = None, fail_if_not_found: bool = True):
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
    if model_type == "stable-diffusion" and not fail_if_not_found:
        for default_model in default_models:
            default_model_path = os.path.join(model_dir, default_model["file_name"])
            if os.path.exists(default_model_path):
                if model_name is not None:
                    log.warn(
                        f"Could not find the configured custom model {model_name}. Using the default one: {default_model_path}"
                    )
                return default_model_path

    if model_name and fail_if_not_found:
        raise Exception(f"Could not find the desired model {model_name}! Is it present in the {model_dir} folder?")


def reload_models_if_necessary(context: Context, models_data: ModelsData, models_to_force_reload: list = []):
    models_to_reload = {
        model_type: path
        for model_type, path in models_data.model_paths.items()
        if context.model_paths.get(model_type) != path or (path is not None and context.models.get(model_type) is None)
    }

    if models_data.model_paths.get("codeformer"):
        if "realesrgan" not in models_to_reload and "realesrgan" not in context.models:
            default_realesrgan = DEFAULT_MODELS["realesrgan"][0]["file_name"]
            models_to_reload["realesrgan"] = resolve_model_to_use(default_realesrgan, "realesrgan")
        elif "realesrgan" in models_to_reload and models_to_reload["realesrgan"] is None:
            del models_to_reload["realesrgan"]  # don't unload realesrgan

    for model_type in models_to_force_reload:
        if model_type not in models_data.model_paths:
            continue
        models_to_reload[model_type] = models_data.model_paths[model_type]

    for model_type, model_path_in_req in models_to_reload.items():
        context.model_paths[model_type] = model_path_in_req

        action_fn = unload_model if context.model_paths[model_type] is None else load_model
        extra_params = models_data.model_params.get(model_type, {})
        try:
            action_fn(context, model_type, scan_model=False, **extra_params)  # we've scanned them already
            if model_type in context.model_load_errors:
                del context.model_load_errors[model_type]
        except Exception as e:
            log.exception(e)
            if action_fn == load_model:
                context.model_load_errors[model_type] = str(e)  # storing the entire Exception can lead to memory leaks


def resolve_model_paths(models_data: ModelsData):
    model_paths = models_data.model_paths
    for model_type in model_paths:
        skip_models = cn_filters + ["latent_upscaler", "nsfw_checker"]
        if model_type in skip_models:  # doesn't use model paths
            continue
        if model_type == "codeformer":
            download_if_necessary("codeformer", "codeformer.pth", "codeformer-0.1.0")
        elif model_type == "controlnet":
            model_id = model_paths[model_type]
            model_info = get_model_info_from_db(model_type=model_type, model_id=model_id)
            if model_info:
                filename = model_info.get("url", "").split("/")[-1]
                download_if_necessary("controlnet", filename, model_id, skip_if_others_exist=False)

        model_paths[model_type] = resolve_model_to_use(model_paths[model_type], model_type=model_type)


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


def download_if_necessary(model_type: str, file_name: str, model_id: str, skip_if_others_exist=True):
    model_path = os.path.join(app.MODELS_DIR, model_type, file_name)
    expected_hash = get_model_info_from_db(model_type=model_type, model_id=model_id)["quick_hash"]

    other_models_exist = any_model_exists(model_type) and skip_if_others_exist
    known_model_exists = os.path.exists(model_path)
    known_model_is_corrupt = known_model_exists and hash_file_quick(model_path) != expected_hash

    if known_model_is_corrupt or (not other_models_exist and not known_model_exists):
        print("> download", model_type, model_id)
        download_model(model_type, model_id, download_base_dir=app.MODELS_DIR, download_config_if_available=False)


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


def getModels(scan_for_malicious: bool = True):
    models = {
        "options": {
            "stable-diffusion": [{"sd-v1-4": "SD 1.4"}],
            "vae": [],
            "hypernetwork": [],
            "lora": [],
            "codeformer": [{"codeformer": "CodeFormer"}],
            "embeddings": [],
            "controlnet": [
                {"control_v11p_sd15_canny": "Canny (*)"},
                {"control_v11p_sd15_openpose": "OpenPose (*)"},
                {"control_v11p_sd15_normalbae": "Normal BAE (*)"},
                {"control_v11f1p_sd15_depth": "Depth (*)"},
                {"control_v11p_sd15_scribble": "Scribble"},
                {"control_v11p_sd15_softedge": "Soft Edge"},
                {"control_v11p_sd15_inpaint": "Inpaint"},
                {"control_v11p_sd15_lineart": "Line Art"},
                {"control_v11p_sd15s2_lineart_anime": "Line Art Anime"},
                {"control_v11p_sd15_mlsd": "Straight Lines"},
                {"control_v11p_sd15_seg": "Segment"},
                {"control_v11e_sd15_shuffle": "Shuffle"},
            ],
        },
    }

    models_scanned = 0

    class MaliciousModelException(Exception):
        "Raised when picklescan reports a problem with a model"

    def scan_directory(directory, suffixes, directoriesFirst: bool = True, default_entries=[]):
        tree = list(default_entries)
        nonlocal models_scanned
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
                    if scan_for_malicious and is_malicious_model(entry.path):
                        raise MaliciousModelException(entry.path)
                if scan_for_malicious:
                    known_models[entry.path] = mtime
                model_id = entry.name[: -len(matching_suffix)]
                model_exists = False
                for m in tree:  # allows default "named" models, like CodeFormer and known ControlNet models
                    if (isinstance(m, str) and model_id == m) or (isinstance(m, dict) and model_id in m):
                        model_exists = True
                        break
                if not model_exists:
                    tree.append(model_id)
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
            default_tree = models["options"].get(model_type, [])
            models["options"][model_type] = scan_directory(models_dir, model_extensions, default_entries=default_tree)
        except MaliciousModelException as e:
            models["scan-error"] = str(e)

    if scan_for_malicious:
        log.info(f"[green]Scanning all model folders for models...[/]")
    # custom models
    listModels(model_type="stable-diffusion")
    listModels(model_type="vae")
    listModels(model_type="hypernetwork")
    listModels(model_type="gfpgan")
    listModels(model_type="lora")
    listModels(model_type="embeddings")
    listModels(model_type="controlnet")

    if scan_for_malicious and models_scanned > 0:
        log.info(f"[green]Scanned {models_scanned} models. Nothing infected[/]")

    return models
