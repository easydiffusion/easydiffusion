import os
import requests
from requests.exceptions import ConnectTimeout, ConnectionError, ReadTimeout
from typing import Union, List
from threading import local as Context
from threading import Thread
import uuid
import time
from copy import deepcopy

from sdkit.utils import base64_str_to_img, img_to_base64_str, log

WEBUI_HOST = "localhost"
WEBUI_PORT = "7860"

DEFAULT_WEBUI_OPTIONS = {
    "show_progress_every_n_steps": 3,
    "show_progress_grid": True,
    "live_previews_enable": False,
    "forge_additional_modules": [],
}


webui_opts: dict = None


curr_models = {
    "stable-diffusion": None,
    "vae": None,
    "text-encoder": None,
}


def set_options(context, **kwargs):
    changed_opts = {}

    opts_mapping = {
        "stream_image_progress": ("live_previews_enable", bool),
        "stream_image_progress_interval": ("show_progress_every_n_steps", int),
        "clip_skip": ("CLIP_stop_at_last_layers", int),
        "clip_skip_sdxl": ("sdxl_clip_l_skip", bool),
        "output_format": ("samples_format", str),
    }

    for ed_key, webui_key in opts_mapping.items():
        webui_key, webui_type = webui_key

        if ed_key in kwargs and (webui_opts is None or webui_opts.get(webui_key, False) != webui_type(kwargs[ed_key])):
            changed_opts[webui_key] = webui_type(kwargs[ed_key])

    if changed_opts:
        changed_opts["sd_model_checkpoint"] = curr_models["stable-diffusion"]

        print(f"Got options: {kwargs}. Sending options: {changed_opts}")

        try:
            res = webui_post("/sdapi/v1/options", json=changed_opts)
            if res.status_code != 200:
                raise Exception(res.text)

            webui_opts.update(changed_opts)
        except Exception as e:
            print(f"Error setting options: {e}")


def ping(timeout=1):
    "timeout (in seconds)"

    global webui_opts

    try:
        res = webui_get("/internal/ping", timeout=timeout)

        if res.status_code != 200:
            raise ConnectTimeout(res.text)

        if webui_opts is None:
            try:
                res = webui_post("/sdapi/v1/options", json=DEFAULT_WEBUI_OPTIONS)
                if res.status_code != 200:
                    raise Exception(res.text)
            except Exception as e:
                print(f"Error setting options: {e}")

            try:
                res = webui_get("/sdapi/v1/options")
                if res.status_code != 200:
                    raise Exception(res.text)

                webui_opts = res.json()
            except Exception as e:
                print(f"Error getting options: {e}")

        return True
    except (ConnectTimeout, ConnectionError, ReadTimeout) as e:
        raise TimeoutError(e)


def load_model(context, model_type, **kwargs):
    from easydiffusion.app import ROOT_DIR, getConfig

    config = getConfig()
    models_dir = config.get("models_dir", os.path.join(ROOT_DIR, "models"))

    model_path = context.model_paths[model_type]

    if model_type == "stable-diffusion":
        base_dir = os.path.join(models_dir, model_type)
        model_path = os.path.relpath(model_path, base_dir)

    # print(f"load model: {model_type=} {model_path=} {curr_models=}")
    curr_models[model_type] = model_path


def unload_model(context, model_type, **kwargs):
    # print(f"unload model: {model_type=} {curr_models=}")
    curr_models[model_type] = None


def flush_model_changes(context):
    if webui_opts is None:
        print("Server not ready, can't set the model")
        return

    modules = []
    for model_type in ("vae", "text-encoder"):
        if curr_models[model_type]:
            model_paths = curr_models[model_type]
            model_paths = [model_paths] if not isinstance(model_paths, list) else model_paths
            modules += model_paths

    opts = {"sd_model_checkpoint": curr_models["stable-diffusion"], "forge_additional_modules": modules}

    print("Setting backend models", opts)

    try:
        res = webui_post("/sdapi/v1/options", json=opts)
        print("got res", res.status_code)
        if res.status_code != 200:
            raise Exception(res.text)
    except Exception as e:
        raise RuntimeError(
            f"The engine failed to set the required options. Please check the logs in the command line window for more details."
        )


def generate_images(
    context: Context,
    prompt: str = "",
    negative_prompt: str = "",
    seed: int = 42,
    width: int = 512,
    height: int = 512,
    num_outputs: int = 1,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    distilled_guidance_scale: float = 3.5,
    init_image=None,
    init_image_mask=None,
    control_image=None,
    control_alpha=1.0,
    controlnet_filter=None,
    prompt_strength: float = 0.8,
    preserve_init_image_color_profile=False,
    strict_mask_border=False,
    sampler_name: str = "euler_a",
    scheduler_name: str = "simple",
    hypernetwork_strength: float = 0,
    tiling=None,
    lora_alpha: Union[float, List[float]] = 0,
    sampler_params={},
    callback=None,
    output_type="pil",
):

    task_id = str(uuid.uuid4())

    sampler_name = convert_ED_sampler_names(sampler_name)
    controlnet_filter = convert_ED_controlnet_filter_name(controlnet_filter)

    cmd = {
        "force_task_id": task_id,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "sampler_name": sampler_name,
        "scheduler": scheduler_name,
        "steps": num_inference_steps,
        "seed": seed,
        "cfg_scale": guidance_scale,
        "distilled_cfg_scale": distilled_guidance_scale,
        "batch_size": num_outputs,
        "width": width,
        "height": height,
    }

    if init_image:
        cmd["init_images"] = [init_image]
        cmd["denoising_strength"] = prompt_strength
    if init_image_mask:
        cmd["mask"] = init_image_mask if isinstance(init_image_mask, str) else img_to_base64_str(init_image_mask)
        cmd["include_init_images"] = True
        cmd["inpainting_fill"] = 1
        cmd["initial_noise_multiplier"] = 1
        cmd["inpaint_full_res"] = 0
        cmd["inpaint_full_res_padding"] = 32
        cmd["resize_mode"] = 1
        cmd["mask_blur"] = 4

    if context.model_paths.get("lora"):
        lora_model = context.model_paths["lora"]
        lora_model = lora_model if isinstance(lora_model, list) else [lora_model]
        lora_alpha = lora_alpha if isinstance(lora_alpha, list) else [lora_alpha]

        for lora, alpha in zip(lora_model, lora_alpha):
            lora = os.path.basename(lora)
            lora = os.path.splitext(lora)[0]
            cmd["prompt"] += f" <lora:{lora}:{alpha}>"

    if controlnet_filter and control_image and context.model_paths.get("controlnet"):
        controlnet_model = context.model_paths["controlnet"]

        model_hash = auto1111_hash(controlnet_model)
        controlnet_model = os.path.basename(controlnet_model)
        controlnet_model = os.path.splitext(controlnet_model)[0]
        print(f"setting controlnet model: {controlnet_model}")
        controlnet_model = f"{controlnet_model} [{model_hash}]"

        cmd["alwayson_scripts"] = {
            "controlnet": {
                "args": [
                    {
                        "image": control_image,
                        "weight": control_alpha,
                        "module": controlnet_filter,
                        "model": controlnet_model,
                        "resize_mode": "Crop and Resize",
                        "threshold_a": 50,
                        "threshold_b": 130,
                    }
                ]
            }
        }

    operation_to_apply = "img2img" if init_image else "txt2img"

    stream_image_progress = webui_opts.get("live_previews_enable", False)

    progress_thread = Thread(
        target=image_progress_thread, args=(task_id, callback, stream_image_progress, num_outputs, num_inference_steps)
    )
    progress_thread.start()

    print(f"task id: {task_id}")
    print_request(operation_to_apply, cmd)

    res = webui_post(f"/sdapi/v1/{operation_to_apply}", json=cmd)
    if res.status_code == 200:
        res = res.json()
    else:
        if res.status_code == 500:
            res = res.json()
            log.error(f"Server error: {res}")
            raise Exception(f"{res['message']}. Please check the logs in the command-line window for more details.")

        raise Exception(
            f"HTTP Status {res.status_code}. The engine failed while generating this image. Please check the logs in the command-line window for more details."
        )

    import json

    print(json.loads(res["info"])["infotexts"])

    images = res["images"]
    if output_type == "pil":
        images = [base64_str_to_img(img) for img in images]
    elif output_type == "base64":
        images = [base64_buffer_to_base64_img(img) for img in images]

    return images


def filter_images(context: Context, images, filters, filter_params={}, input_type="pil"):
    """
    * context: Context
    * images: str or PIL.Image or list of str/PIL.Image - image to filter. if a string is passed, it needs to be a base64-encoded image
    * filters: filter_type (string) or list of strings
    * filter_params: dict

    returns: [PIL.Image] - list of filtered images
    """
    images = images if isinstance(images, list) else [images]
    filters = filters if isinstance(filters, list) else [filters]

    if "nsfw_checker" in filters:
        filters.remove("nsfw_checker")  # handled by ED directly

    args = {}
    controlnet_filters = []

    print(filter_params)

    for filter_name in filters:
        params = filter_params.get(filter_name, {})

        if filter_name == "gfpgan":
            args["gfpgan_visibility"] = 1

        if filter_name in ("realesrgan", "esrgan_4x", "lanczos", "nearest", "scunet", "swinir"):
            args["upscaler_1"] = params.get("upscaler", "RealESRGAN_x4plus")
            args["upscaling_resize"] = params.get("scale", 4)

            if args["upscaler_1"] == "RealESRGAN_x4plus":
                args["upscaler_1"] = "R-ESRGAN 4x+"
            elif args["upscaler_1"] == "RealESRGAN_x4plus_anime_6B":
                args["upscaler_1"] = "R-ESRGAN 4x+ Anime6B"

        if filter_name == "codeformer":
            args["codeformer_visibility"] = 1
            args["codeformer_weight"] = params.get("codeformer_fidelity", 0.5)

        if filter_name.startswith("controlnet_"):
            filter_name = convert_ED_controlnet_filter_name(filter_name)
            controlnet_filters.append(filter_name)

    print(f"filtering {len(images)} images with {args}. {controlnet_filters=}")

    if len(filters) > len(controlnet_filters):
        filtered_images = extra_batch_images(images, input_type=input_type, **args)
    else:
        filtered_images = images

    for filter_name in controlnet_filters:
        filtered_images = controlnet_filter(filtered_images, module=filter_name, input_type=input_type)

    return filtered_images


def get_url():
    return f"//{WEBUI_HOST}:{WEBUI_PORT}/?__theme=dark"


def stop_rendering(context):
    try:
        res = webui_post("/sdapi/v1/interrupt")
        if res.status_code != 200:
            raise Exception(res.text)
    except Exception as e:
        print(f"Error interrupting webui: {e}")


def refresh_models():
    def make_refresh_call(type):
        try:
            webui_post(f"/sdapi/v1/refresh-{type}")
        except:
            pass

    try:
        for type in ("checkpoints", "vae-and-text-encoders"):
            t = Thread(target=make_refresh_call, args=(type,))
            t.start()
    except Exception as e:
        print(f"Error refreshing models: {e}")


def list_controlnet_filters():
    return [
        "openpose",
        "openpose_face",
        "openpose_faceonly",
        "openpose_hand",
        "openpose_full",
        "animal_openpose",
        "densepose_parula (black bg & blue torso)",
        "densepose (pruple bg & purple torso)",
        "dw_openpose_full",
        "mediapipe_face",
        "instant_id_face_keypoints",
        "InsightFace+CLIP-H (IPAdapter)",
        "InsightFace (InstantID)",
        "canny",
        "mlsd",
        "scribble_hed",
        "scribble_hedsafe",
        "scribble_pidinet",
        "scribble_pidsafe",
        "scribble_xdog",
        "softedge_hed",
        "softedge_hedsafe",
        "softedge_pidinet",
        "softedge_pidsafe",
        "softedge_teed",
        "normal_bae",
        "depth_midas",
        "normal_midas",
        "depth_zoe",
        "depth_leres",
        "depth_leres++",
        "depth_anything_v2",
        "depth_anything",
        "depth_hand_refiner",
        "depth_marigold",
        "lineart_coarse",
        "lineart_realistic",
        "lineart_anime",
        "lineart_standard (from white bg & black line)",
        "lineart_anime_denoise",
        "reference_adain",
        "reference_only",
        "reference_adain+attn",
        "tile_colorfix",
        "tile_resample",
        "tile_colorfix+sharp",
        "CLIP-ViT-H (IPAdapter)",
        "CLIP-G (Revision)",
        "CLIP-G (Revision ignore prompt)",
        "CLIP-ViT-bigG (IPAdapter)",
        "InsightFace+CLIP-H (IPAdapter)",
        "inpaint_only",
        "inpaint_only+lama",
        "inpaint_global_harmonious",
        "seg_ufade20k",
        "seg_ofade20k",
        "seg_anime_face",
        "seg_ofcoco",
        "shuffle",
        "segment",
        "invert (from white bg & black line)",
        "threshold",
        "t2ia_sketch_pidi",
        "t2ia_color_grid",
        "recolor_intensity",
        "recolor_luminance",
        "blur_gaussian",
    ]


def controlnet_filter(images, module="none", processor_res=512, threshold_a=64, threshold_b=64, input_type="pil"):
    if input_type == "pil":
        images = [img_to_base64_str(x) for x in images]

    payload = {
        "controlnet_module": module,
        "controlnet_input_images": images,
        "controlnet_processor_res": processor_res,
        "controlnet_threshold_a": threshold_a,
        "controlnet_threshold_b": threshold_b,
    }
    res = webui_post("/controlnet/detect", json=payload)
    res = res.json()
    filtered_images = res["images"]

    if input_type == "pil":
        filtered_images = [base64_str_to_img(img) for img in filtered_images]
    elif input_type == "base64":
        filtered_images = [base64_buffer_to_base64_img(img) for img in filtered_images]

    return filtered_images


def image_progress_thread(task_id, callback, stream_image_progress, total_images, total_steps):
    from PIL import Image

    last_preview_id = -1

    EMPTY_IMAGE = Image.new("RGB", (1, 1))

    while True:
        res = webui_post(
            f"/internal/progress",
            json={"id_task": task_id, "live_preview": stream_image_progress, "id_live_preview": last_preview_id},
        )
        if res.status_code == 200:
            res = res.json()
        else:
            raise RuntimeError(f"Unexpected progress response. Status code: {res.status_code}. Res: {res.text}")

        last_preview_id = res["id_live_preview"]

        if res["progress"] is not None:
            step_num = int(res["progress"] * total_steps)

            if res["live_preview"] is not None:
                img = res["live_preview"]
                img = base64_str_to_img(img)
                images = [EMPTY_IMAGE] * total_images
                images[0] = img
            else:
                images = None

            callback(images, step_num)

        if res["completed"] == True:
            print("Complete!")
            break

        time.sleep(0.5)


def webui_get(uri, *args, **kwargs):
    url = f"http://{WEBUI_HOST}:{WEBUI_PORT}{uri}"
    return requests.get(url, *args, **kwargs)


def webui_post(uri, *args, **kwargs):
    url = f"http://{WEBUI_HOST}:{WEBUI_PORT}{uri}"
    return requests.post(url, *args, **kwargs)


def print_request(operation_to_apply, args):
    args = deepcopy(args)
    if "init_images" in args:
        args["init_images"] = ["img" for _ in args["init_images"]]
    if "mask" in args:
        args["mask"] = "mask_img"

    controlnet_args = args.get("alwayson_scripts", {}).get("controlnet", {}).get("args", [])
    if controlnet_args:
        controlnet_args[0]["image"] = "control_image"

    print(f"operation: {operation_to_apply}, args: {args}")


def auto1111_hash(file_path):
    import hashlib

    with open(file_path, "rb") as f:
        f.seek(0x100000)
        b = f.read(0x10000)
        return hashlib.sha256(b).hexdigest()[:8]


def extra_batch_images(
    images,  # list of PIL images
    name_list=None,  # list of image names
    resize_mode=0,
    show_extras_results=True,
    gfpgan_visibility=0,
    codeformer_visibility=0,
    codeformer_weight=0,
    upscaling_resize=2,
    upscaling_resize_w=512,
    upscaling_resize_h=512,
    upscaling_crop=True,
    upscaler_1="None",
    upscaler_2="None",
    extras_upscaler_2_visibility=0,
    upscale_first=False,
    use_async=False,
    input_type="pil",
):
    if name_list is not None:
        if len(name_list) != len(images):
            raise RuntimeError("len(images) != len(name_list)")
    else:
        name_list = [f"image{i + 1:05}" for i in range(len(images))]

    if input_type == "pil":
        images = [img_to_base64_str(x) for x in images]

    image_list = []
    for name, image in zip(name_list, images):
        image_list.append({"data": image, "name": name})

    payload = {
        "resize_mode": resize_mode,
        "show_extras_results": show_extras_results,
        "gfpgan_visibility": gfpgan_visibility,
        "codeformer_visibility": codeformer_visibility,
        "codeformer_weight": codeformer_weight,
        "upscaling_resize": upscaling_resize,
        "upscaling_resize_w": upscaling_resize_w,
        "upscaling_resize_h": upscaling_resize_h,
        "upscaling_crop": upscaling_crop,
        "upscaler_1": upscaler_1,
        "upscaler_2": upscaler_2,
        "extras_upscaler_2_visibility": extras_upscaler_2_visibility,
        "upscale_first": upscale_first,
        "imageList": image_list,
    }

    res = webui_post("/sdapi/v1/extra-batch-images", json=payload)
    if res.status_code == 200:
        res = res.json()
    else:
        raise Exception(
            "The engine failed while filtering this image. Please check the logs in the command-line window for more details."
        )

    images = res["images"]

    if input_type == "pil":
        images = [base64_str_to_img(img) for img in images]
    elif input_type == "base64":
        images = [base64_buffer_to_base64_img(img) for img in images]

    return images


def base64_buffer_to_base64_img(img):
    output_format = webui_opts.get("samples_format", "jpeg")
    mime_type = f"image/{output_format.lower()}"
    return f"data:{mime_type};base64," + img


def convert_ED_sampler_names(sampler_name):
    name_mapping = {
        "dpmpp_2m": "DPM++ 2M",
        "dpmpp_sde": "DPM++ SDE",
        "dpmpp_2m_sde": "DPM++ 2M SDE",
        "dpmpp_2m_sde_heun": "DPM++ 2M SDE Heun",
        "dpmpp_2s_a": "DPM++ 2S a",
        "dpmpp_3m_sde": "DPM++ 3M SDE",
        "euler_a": "Euler a",
        "euler": "Euler",
        "lms": "LMS",
        "heun": "Heun",
        "dpm2": "DPM2",
        "dpm2_a": "DPM2 a",
        "dpm_fast": "DPM fast",
        "dpm_adaptive": "DPM adaptive",
        "restart": "Restart",
        "heun_pp2": "HeunPP2",
        "ipndm": "IPNDM",
        "ipndm_v": "IPNDM_V",
        "deis": "DEIS",
        "ddim": "DDIM",
        "ddim_cfgpp": "DDIM CFG++",
        "plms": "PLMS",
        "unipc": "UniPC",
        "lcm": "LCM",
        "ddpm": "DDPM",
        "forge_flux_realistic": "[Forge] Flux Realistic",
        "forge_flux_realistic_slow": "[Forge] Flux Realistic (Slow)",
        # deprecated samplers in 3.5
        "dpm_solver_stability": None,
        "unipc_snr": None,
        "unipc_tu": None,
        "unipc_snr_2": None,
        "unipc_tu_2": None,
        "unipc_tq": None,
    }
    return name_mapping.get(sampler_name)


def convert_ED_controlnet_filter_name(filter):
    if filter is None:
        return None

    def cn(n):
        if n.startswith("controlnet_"):
            return n[len("controlnet_") :]
        return n

    mapping = {
        "controlnet_scribble_hedsafe": None,
        "controlnet_scribble_pidsafe": None,
        "controlnet_softedge_pidsafe": "controlnet_softedge_pidisafe",
        "controlnet_normal_bae": "controlnet_normalbae",
        "controlnet_segment": None,
    }
    if isinstance(filter, list):
        return [cn(mapping.get(f, f)) for f in filter]
    return cn(mapping.get(filter, filter))
