"""Legacy API compatibility helpers for the new EasyDiffusion server."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

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

    async def get_legacy_system_info():
        from easydiffusion.server import NOCACHE_HEADERS

        try:
            config_manager = server_api.state.config_manager
            system_info = get_system_info(config_manager)
            return JSONResponse(system_info, headers=NOCACHE_HEADERS)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    server_api.add_api_route("/render", create_legacy_render_task, methods=["POST"])
    server_api.add_api_route("/filter", create_legacy_filter_task, methods=["POST"])
    server_api.add_api_route("/get/models", get_models, methods=["GET"])
    server_api.add_api_route("/get/system_info", get_legacy_system_info, methods=["GET"])


def get_system_info(config_manager):
    import os

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
