"""Legacy API compatibility helpers for the new EasyDiffusion server."""

from __future__ import annotations

import os
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from easydiffusion.types import FilterTaskRequest, GenerateTaskRequest

LEGACY_MODEL_FIELD_MAP = {
    "use_stable_diffusion_model": "stable-diffusion",
    "use_vae_model": "vae",
    "use_text_encoder_model": "text-encoder",
    "use_hypernetwork_model": "hypernetwork",
    "use_lora_model": "lora",
    "use_controlnet_model": "controlnet",
    "use_embeddings_model": "embeddings",
}
LEGACY_FACE_CORRECTION_MODELS = ("gfpgan", "codeformer")
LEGACY_UPSCALE_MODELS = ("realesrgan", "latent_upscaler", "esrgan_4x", "lanczos", "nearest", "scunet", "swinir")
LEGACY_UPSCALE_FILTER_MODELS = ("realesrgan", "esrgan_4x", "lanczos", "nearest", "scunet", "swinir")
LEGACY_DEFAULT_USERNAME = "easydiffusion"
LEGACY_OUTPUT_DIR_NAME = "Stable Diffusion UI"
_LEGACY_GENERATE_FIELDS = set(GenerateTaskRequest.model_fields)
_LEGACY_FILTER_FIELDS = set(FilterTaskRequest.model_fields)


class NoCacheStaticFiles(StaticFiles):
    def __init__(self, directory: str):
        # follow_symlink is only available on fastapi >= 0.92.0
        if os.path.islink(directory):
            super().__init__(directory=os.path.realpath(directory))
        else:
            super().__init__(directory=directory)

    def is_not_modified(self, response_headers, request_headers) -> bool:
        from easydiffusion.server import NOCACHE_HEADERS

        if "content-type" in response_headers and (
            "javascript" in response_headers["content-type"] or "css" in response_headers["content-type"]
        ):
            response_headers.update(NOCACHE_HEADERS)
            return False

        return super().is_not_modified(response_headers, request_headers)


def support_legacy_paths(
    server_api: FastAPI,
    *,
    enqueue_task: Callable[[GenerateTaskRequest | FilterTaskRequest], JSONResponse],
) -> None:
    from easydiffusion.server import get_models

    async def create_legacy_render_task(req: dict[str, Any]):
        try:
            translated = translate_legacy_render_request(req, get_backend_controlnet_filters(server_api.state))
            request_model = GenerateTaskRequest(**translated)
            return enqueue_task(request_model)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def create_legacy_filter_task(req: dict[str, Any]):
        try:
            translated = translate_legacy_filter_request(req, get_backend_controlnet_filters(server_api.state))
            request_model = FilterTaskRequest(**translated)
            return enqueue_task(request_model)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def _return_json(data_fn):
        from easydiffusion.server import NOCACHE_HEADERS

        try:
            data = data_fn()
            return JSONResponse(data, headers=NOCACHE_HEADERS)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_legacy_system_info():
        def data_fn():
            config_manager = server_api.state.config_manager
            return get_system_info(config_manager)

        return _return_json(data_fn)

    async def get_legacy_modifiers():
        return _return_json(get_image_modifiers)

    async def get_legacy_ui_plugins():
        return _return_json(get_ui_plugins)

    server_api.add_api_route("/render", create_legacy_render_task, methods=["POST"])
    server_api.add_api_route("/filter", create_legacy_filter_task, methods=["POST"])
    server_api.add_api_route("/get/models", get_models, methods=["GET"])
    server_api.add_api_route("/get/system_info", get_legacy_system_info, methods=["GET"])
    server_api.add_api_route("/get/modifiers", get_legacy_modifiers, methods=["GET"])
    server_api.add_api_route("/get/ui_plugins", get_legacy_ui_plugins, methods=["GET"])

    for plugins_dir, dir_prefix in get_ui_plugin_sources():
        if os.path.exists(plugins_dir):
            server_api.mount(
                f"/plugins/{dir_prefix}",
                NoCacheStaticFiles(directory=plugins_dir),
                name=f"plugins-{dir_prefix}",
            )


def get_system_info(config_manager):
    user_config = config_manager.get_user_config(LEGACY_DEFAULT_USERNAME)

    default_save_path = os.path.join(os.path.expanduser("~"), LEGACY_OUTPUT_DIR_NAME)

    force_save_path = config_manager.get("security", {}).get("force_save_path", "")
    output_dir = force_save_path or user_config.get("save", {}).get("save_path") or default_save_path

    system_info = {
        "devices": get_legacy_devices(),
        "hosts": get_legacy_hosts(),
        "default_output_dir": output_dir,
        "enforce_output_dir": (True if force_save_path else False),
        "enforce_output_metadata": config_manager.get("security", {}).get("force_save_metadata", False),
    }
    system_info["devices"]["config"] = config_manager.get("backend", {}).get("devices", "auto")

    return system_info


def get_legacy_devices():
    from easydiffusion.utils import get_devices

    devices = get_devices()
    cpu_device = next((device.model_dump() for device in devices if device.id == "cpu"))
    gpu_devices = [device.model_dump() for device in devices if device.id != "cpu"]

    res = {
        "all": {"cpu": cpu_device},
        "active": {},
    }
    if gpu_devices:
        if len(gpu_devices) > 1:
            gpu_entries = {f"cuda:{i}": device for i, device in enumerate(gpu_devices)}
        else:
            gpu_entries = {"cuda": gpu_devices[0]}

        res["all"].update(gpu_entries)
        res["active"].update(gpu_entries)
    else:
        res["active"] = {"cpu": cpu_device}

    return res


def get_legacy_hosts():
    import socket

    try:
        ips = socket.gethostbyname_ex(socket.gethostname())
        ips[2].append(ips[0])
        return ips[2]
    except Exception:
        return []


def get_image_modifiers():
    import json
    import urllib

    CUSTOM_MODIFIERS_DIR = os.path.abspath("modifiers")
    CUSTOM_MODIFIERS_PORTRAIT_EXTENSIONS = [
        ".portrait",
        "_portrait",
        " portrait",
        "-portrait",
    ]
    CUSTOM_MODIFIERS_LANDSCAPE_EXTENSIONS = [
        ".landscape",
        "_landscape",
        " landscape",
        "-landscape",
    ]
    IMAGE_EXTENSIONS = [
        ".png",
        ".apng",
        ".jpg",
        ".jpeg",
        ".jfif",
        ".pjpeg",
        ".pjp",
        ".jxl",
        ".gif",
        ".webp",
        ".avif",
        ".svg",
    ]

    modifiers_json_path = os.path.join("ui", "modifiers.json")

    modifier_categories = {}
    original_category_order = []
    with open(modifiers_json_path, "r", encoding="utf-8") as f:
        modifiers_file = json.load(f)

        # The trailing slash is needed to support symlinks
        if not os.path.isdir(f"{CUSTOM_MODIFIERS_DIR}/"):
            return modifiers_file

        # convert modifiers from a list of objects to a dict of dicts
        for category_item in modifiers_file:
            category_name = category_item["category"]
            original_category_order.append(category_name)
            category = {}
            for modifier_item in category_item["modifiers"]:
                modifier = {}
                for preview_item in modifier_item["previews"]:
                    modifier[preview_item["name"]] = preview_item["path"]
                category[modifier_item["modifier"]] = modifier
            modifier_categories[category_name] = category

    def scan_directory(directory_path: str, category_name="Modifiers"):
        for entry in os.scandir(directory_path):
            if entry.is_file():
                file_extension = list(filter(lambda e: entry.name.endswith(e), IMAGE_EXTENSIONS))
                if len(file_extension) == 0:
                    continue

                modifier_name = entry.name[: -len(file_extension[0])]
                modifier_path = f"custom/{entry.path[len(CUSTOM_MODIFIERS_DIR) + 1 :]}"
                # URL encode path segments
                modifier_path = "/".join(
                    map(
                        lambda segment: urllib.parse.quote(segment),
                        modifier_path.split("/"),
                    )
                )
                is_portrait = True
                is_landscape = True

                portrait_extension = list(
                    filter(
                        lambda e: modifier_name.lower().endswith(e),
                        CUSTOM_MODIFIERS_PORTRAIT_EXTENSIONS,
                    )
                )
                landscape_extension = list(
                    filter(
                        lambda e: modifier_name.lower().endswith(e),
                        CUSTOM_MODIFIERS_LANDSCAPE_EXTENSIONS,
                    )
                )

                if len(portrait_extension) > 0:
                    is_landscape = False
                    modifier_name = modifier_name[: -len(portrait_extension[0])]
                elif len(landscape_extension) > 0:
                    is_portrait = False
                    modifier_name = modifier_name[: -len(landscape_extension[0])]

                if category_name not in modifier_categories:
                    modifier_categories[category_name] = {}

                category = modifier_categories[category_name]

                if modifier_name not in category:
                    category[modifier_name] = {}

                if is_portrait or "portrait" not in category[modifier_name]:
                    category[modifier_name]["portrait"] = modifier_path

                if is_landscape or "landscape" not in category[modifier_name]:
                    category[modifier_name]["landscape"] = modifier_path
            elif entry.is_dir():
                scan_directory(
                    entry.path,
                    entry.name if directory_path == CUSTOM_MODIFIERS_DIR else f"{category_name}/{entry.name}",
                )

    scan_directory(CUSTOM_MODIFIERS_DIR)

    custom_categories = sorted(
        [cn for cn in modifier_categories.keys() if cn not in original_category_order],
        key=str.casefold,
    )

    # convert the modifiers back into a list of objects
    modifier_categories_list = []
    for category_name in [*original_category_order, *custom_categories]:
        category = {"category": category_name, "modifiers": []}
        for modifier_name in sorted(modifier_categories[category_name].keys(), key=str.casefold):
            modifier = {"modifier": modifier_name, "previews": []}
            for preview_name, preview_path in modifier_categories[category_name][modifier_name].items():
                modifier["previews"].append({"name": preview_name, "path": preview_path})
            category["modifiers"].append(modifier)
        modifier_categories_list.append(category)

    return modifier_categories_list


def get_ui_plugin_sources():
    USER_UI_PLUGINS_DIR = os.path.join("plugins", "ui")
    CORE_UI_PLUGINS_DIR = os.path.join("ui", "plugins", "ui")
    return ((CORE_UI_PLUGINS_DIR, "core"), (USER_UI_PLUGINS_DIR, "user"))


def get_ui_plugins():
    plugins = []

    file_names = set()
    for plugins_dir, dir_prefix in get_ui_plugin_sources():
        if os.path.exists(plugins_dir):
            for file in os.listdir(plugins_dir):
                if file.endswith(".plugin.js") and file not in file_names:
                    plugins.append(f"/plugins/{dir_prefix}/{file}")
                    file_names.add(file)

    return plugins


def get_backend_controlnet_filters(server_state: Any) -> set[str]:
    backend_class = server_state.workers.backend_class

    try:
        return set(backend_class.list_controlnet_filters())
    except Exception:
        return set()


def translate_legacy_render_request(payload: dict[str, Any], controlnet_filters: set[str]) -> dict[str, Any]:
    translated = {key: value for key, value in payload.items() if key in _LEGACY_GENERATE_FIELDS}
    translated["username"] = payload.get("username") or LEGACY_DEFAULT_USERNAME

    request_id = payload.get("task_id") or payload.get("request_id")
    if request_id is not None:
        translated["task_id"] = request_id

    if "mask" in payload and "init_image_mask" not in translated:
        translated["init_image_mask"] = payload["mask"]

    model_paths = dict(payload.get("model_paths") or {})
    model_params = dict(payload.get("model_params") or {})
    filters = list(payload.get("filters") or [])
    filter_params = dict(payload.get("filter_params") or {})

    for legacy_key, model_name in LEGACY_MODEL_FIELD_MAP.items():
        if legacy_key in payload:
            model_paths[model_name] = payload.get(legacy_key)

    face_correction = payload.get("use_face_correction", "") or ""
    for model_name in LEGACY_FACE_CORRECTION_MODELS:
        if model_name in face_correction.lower():
            model_paths[model_name] = face_correction
            if model_name not in filters:
                filters.append(model_name)
            break

    upscale_model = payload.get("use_upscale", "") or ""
    for model_name in LEGACY_UPSCALE_MODELS:
        if model_name in upscale_model.lower():
            model_paths[model_name] = upscale_model
            if model_name in LEGACY_UPSCALE_FILTER_MODELS:
                filter_params[model_name] = {
                    "upscaler": upscale_model,
                    "scale": int(payload.get("upscale_amount", 4)),
                }
            elif model_name == "latent_upscaler":
                filter_params[model_name] = {
                    "prompt": payload["prompt"],
                    "negative_prompt": payload.get("negative_prompt"),
                    "seed": int(payload.get("seed", 42)),
                    "num_inference_steps": int(payload.get("latent_upscaler_steps", 10)),
                    "guidance_scale": 0,
                }
            if model_name not in filters:
                filters.append(model_name)
            break

    if payload.get("block_nsfw"):
        model_paths["nsfw_checker"] = "nsfw_checker"
        if "nsfw_checker" not in filters:
            filters.insert(0, "nsfw_checker")

    if model_paths.get("stable-diffusion"):
        model_params.setdefault(
            "stable-diffusion",
            {
                "clip_skip": bool(payload.get("clip_skip", False)),
            },
        )

    if model_paths.get("codeformer"):
        filter_params["codeformer"] = {
            "upscale_faces": bool(payload.get("codeformer_upscale_faces", True)),
            "codeformer_fidelity": float(payload.get("codeformer_fidelity", 0.5)),
        }

    control_filter = payload.get("control_filter_to_apply")
    if control_filter is not None:
        model_paths[control_filter] = control_filter
        translated["controlnet_filter"] = convert_legacy_controlnet_filter_name(control_filter, controlnet_filters)

    translated["model_paths"] = model_paths
    translated["model_params"] = model_params
    translated["filters"] = filters
    translated["filter_params"] = filter_params
    return translated


def translate_legacy_filter_request(payload: dict[str, Any], controlnet_filters: set[str]) -> dict[str, Any]:
    translated = {key: value for key, value in payload.items() if key in _LEGACY_FILTER_FIELDS}
    translated["username"] = payload.get("username") or LEGACY_DEFAULT_USERNAME

    request_id = payload.get("task_id") or payload.get("request_id")
    if request_id is not None:
        translated["task_id"] = request_id

    translated["filter"] = convert_legacy_controlnet_filter_name(payload.get("filter"), controlnet_filters)

    model_paths = dict(payload.get("model_paths") or {})
    filter_params = dict(payload.get("filter_params") or {})
    for model_name in LEGACY_UPSCALE_FILTER_MODELS:
        upscaler = model_paths.get(model_name)
        if upscaler:
            model_filter_params = dict(filter_params.get(model_name) or {})
            model_filter_params.setdefault("upscaler", upscaler)
            filter_params[model_name] = model_filter_params

    translated["model_paths"] = model_paths
    translated["filter_params"] = filter_params
    return translated


def convert_legacy_controlnet_filter_name(filter_name: Any, controlnet_filters: set[str]) -> Any:
    if filter_name is None:
        return None

    def apply(value: str) -> str:
        return f"controlnet_{value}" if value in controlnet_filters else value

    if isinstance(filter_name, list):
        return [apply(value) for value in filter_name]
    return apply(filter_name)
