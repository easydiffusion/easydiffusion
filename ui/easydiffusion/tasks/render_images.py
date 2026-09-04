import json
import pprint
import queue
import time
from PIL import Image

from easydiffusion import model_manager, runtime
from easydiffusion.types import GenerateImageRequest, ModelsData, OutputFormatData, SaveToDiskData
from easydiffusion.types import Image as ResponseImage
from easydiffusion.types import GenerateImageResponse, RenderTaskData
from easydiffusion.utils import get_printable_request, log, save_images_to_disk, filter_nsfw
from sdkit.utils import (
    img_to_base64_str,
    base64_str_to_img,
    img_to_buffer,
    resize_img,
    get_image,
    log,
)

from .task import Task


class RenderTask(Task):
    "For image generation"

    def __init__(
        self,
        req: GenerateImageRequest,
        task_data: RenderTaskData,
        models_data: ModelsData,
        output_format: OutputFormatData,
        save_data: SaveToDiskData,
    ):
        super().__init__(task_data.session_id)

        task_data.request_id = self.id

        self.render_request = req  # Initial Request
        self.task_data = task_data
        self.models_data = models_data
        self.output_format = output_format
        self.save_data = save_data

        self.temp_images: list = [None] * req.num_outputs * (1 if task_data.show_only_filtered_image else 2)

    def run(self):
        "Runs the image generation task on the assigned thread"

        from easydiffusion import task_manager, app
        from easydiffusion.backend_manager import backend

        context = runtime.context
        config = app.getConfig()

        if config.get("block_nsfw", False):  # override if set on the server
            self.task_data.block_nsfw = True

        def step_callback():
            task_manager.keep_task_alive(self)
            task_manager.current_state = task_manager.ServerStates.Rendering

            if isinstance(task_manager.current_state_error, (SystemExit, StopAsyncIteration)) or isinstance(
                self.error, StopAsyncIteration
            ):
                backend.stop_rendering(context)
                if isinstance(task_manager.current_state_error, StopAsyncIteration):
                    self.error = task_manager.current_state_error
                    task_manager.current_state_error = None
                    log.info(f"Session {self.session_id} sent cancel signal for task {self.id}")

        task_manager.current_state = task_manager.ServerStates.LoadingModel
        model_manager.resolve_model_paths(self.models_data)

        models_to_force_reload = []
        if runtime.set_vram_optimizations(context) or self.has_param_changed(context, "clip_skip"):
            models_to_force_reload.append("stable-diffusion")

        model_manager.reload_models_if_necessary(context, self.models_data, models_to_force_reload)
        model_manager.fail_if_models_did_not_load(context)

        task_manager.current_state = task_manager.ServerStates.Rendering
        self.response = make_images(
            context,
            self.render_request,
            self.task_data,
            self.models_data,
            self.output_format,
            self.save_data,
            self.buffer_queue,
            self.temp_images,
            step_callback,
            self,
        )

    def has_param_changed(self, context, param_name):
        if not getattr(context, "test_diffusers", False):
            return False
        if "stable-diffusion" not in context.models or "params" not in context.models["stable-diffusion"]:
            return True

        model = context.models["stable-diffusion"]
        new_val = self.models_data.model_params.get("stable-diffusion", {}).get(param_name, False)
        return model["params"].get(param_name) != new_val


def make_images(
    context,
    req: GenerateImageRequest,
    task_data: RenderTaskData,
    models_data: ModelsData,
    output_format: OutputFormatData,
    save_data: SaveToDiskData,
    data_queue: queue.Queue,
    task_temp_images: list,
    step_callback,
    task,
):
    print_task_info(req, task_data, models_data, output_format, save_data)

    images, seeds = make_images_internal(
        context,
        req,
        task_data,
        models_data,
        output_format,
        save_data,
        data_queue,
        task_temp_images,
        step_callback,
        task,
    )

    res = GenerateImageResponse(
        req, task_data, models_data, output_format, save_data, images=construct_response(images, seeds, output_format)
    )
    res = res.json()
    data_queue.put(json.dumps(res))
    log.info("Task completed")

    return res


def print_task_info(
    req: GenerateImageRequest,
    task_data: RenderTaskData,
    models_data: ModelsData,
    output_format: OutputFormatData,
    save_data: SaveToDiskData,
):
    req_str = pprint.pformat(get_printable_request(req, task_data, models_data, output_format, save_data)).replace(
        "[", "\["
    )
    task_str = pprint.pformat(task_data.dict()).replace("[", "\[")
    models_data = pprint.pformat(models_data.dict()).replace("[", "\[")
    output_format = pprint.pformat(output_format.dict()).replace("[", "\[")
    save_data = pprint.pformat(save_data.dict()).replace("[", "\[")

    log.info(f"request: {req_str}")
    log.info(f"task data: {task_str}")
    log.info(f"models data: {models_data}")
    log.info(f"output format: {output_format}")
    log.info(f"save data: {save_data}")


ONLY_MASKED_PADDING = 32  # px on the original image scale, like A1111's inpaint_full_res_padding


def crop_init_to_mask(req, task_data):
    "Crop init image + mask to the mask bbox ('Only masked area' inpaint). Mutates req. Returns info for paste-back, or None to use the whole picture."
    if not getattr(task_data, "inpaint_only_masked", False):
        return None
    if req.init_image is None or not req.init_image_mask:
        return None
    try:
        init_pil = get_image(req.init_image)
        mask_raw = get_image(req.init_image_mask)
        if init_pil is None or mask_raw is None:
            return None
        init_pil = init_pil.convert("RGB")
        mask_rgba = mask_raw.convert("RGBA")
        if mask_rgba.size != init_pil.size:
            mask_rgba = resize_img(mask_rgba, init_pil.width, init_pil.height)
        alpha = mask_rgba.getchannel("A")
        if alpha.getextrema() == (255, 255):
            # fully opaque (e.g. RGB white-on-black mask): white = masked area
            alpha = mask_rgba.convert("L").point(lambda v: 255 if v >= 16 else 0)
        bbox = alpha.getbbox()
        if bbox is None:
            return None
        x0, y0, x1, y1 = bbox
        x0 = max(0, x0 - ONLY_MASKED_PADDING)
        y0 = max(0, y0 - ONLY_MASKED_PADDING)
        x1 = min(init_pil.width, x1 + ONLY_MASKED_PADDING)
        y1 = min(init_pil.height, y1 + ONLY_MASKED_PADDING)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        crop_init = resize_img(init_pil.crop((x0, y0, x1, y1)), req.width, req.height)
        crop_mask = resize_img(mask_rgba.crop((x0, y0, x1, y1)), req.width, req.height)
        req.init_image = img_to_base64_str(crop_init)
        req.init_image_mask = crop_mask
        log.info(
            f"Only-masked inpaint: cropped {init_pil.width}x{init_pil.height} to bbox "
            f"{x0},{y0},{x1},{y1}, processing at {req.width}x{req.height}"
        )
        return {"bbox": (x0, y0, x1, y1), "orig": init_pil}
    except Exception as e:
        log.error(f"Only-masked crop failed, falling back to whole picture: {e}")
        return None


def paste_back_to_original(images, info):
    "Paste generated crops back into the original image. images: list of base64 strings."
    x0, y0, x1, y1 = info["bbox"]
    orig = info["orig"]
    out = []
    for img_str in images:
        try:
            gen = base64_str_to_img(img_str).convert("RGB")
            gen = resize_img(gen, x1 - x0, y1 - y0)
            full = orig.copy()
            full.paste(gen, (x0, y0))
            out.append(img_to_base64_str(full))
        except Exception as e:
            log.error(f"Only-masked paste-back failed for one image: {e}")
            out.append(img_str)
    return out


def make_images_internal(
    context,
    req: GenerateImageRequest,
    task_data: RenderTaskData,
    models_data: ModelsData,
    output_format: OutputFormatData,
    save_data: SaveToDiskData,
    data_queue: queue.Queue,
    task_temp_images: list,
    step_callback,
    task,
):
    from easydiffusion.backend_manager import backend

    # prep the nsfw_filter
    if task_data.block_nsfw:
        filter_nsfw([Image.new("RGB", (1, 1))])  # hack - ensures that the model is available

    only_masked_info = crop_init_to_mask(req, task_data)
    images = generate_images_internal(
        context,
        req,
        task_data,
        models_data,
        output_format,
        data_queue,
        task_temp_images,
        step_callback,
        task_data.stream_image_progress,
        task_data.stream_image_progress_interval,
    )
    if only_masked_info is not None:
        images = paste_back_to_original(images, only_masked_info)
    user_stopped = isinstance(task.error, StopAsyncIteration)

    filters, filter_params = task_data.filters, task_data.filter_params
    if len(filters) > 0 and not user_stopped:
        filtered_images = backend.filter_images(context, images, filters, filter_params, input_type="base64")
    else:
        filtered_images = images

    if task_data.block_nsfw:
        filtered_images = filter_nsfw(filtered_images)

    if save_data.save_to_disk_path is not None:
        images_pil = [base64_str_to_img(img) for img in images]
        filtered_images_pil = [base64_str_to_img(img) for img in filtered_images]
        save_images_to_disk(images_pil, filtered_images_pil, req, task_data, models_data, output_format, save_data)

    seeds = [*range(req.seed, req.seed + len(images))]
    if task_data.show_only_filtered_image or filtered_images is images:
        return filtered_images, seeds
    else:
        return images + filtered_images, seeds + seeds


def generate_images_internal(
    context,
    req: GenerateImageRequest,
    task_data: RenderTaskData,
    models_data: ModelsData,
    output_format: OutputFormatData,
    data_queue: queue.Queue,
    task_temp_images: list,
    step_callback,
    stream_image_progress: bool,
    stream_image_progress_interval: int,
):
    from easydiffusion.backend_manager import backend

    callback = make_step_callback(context, req, task_data, data_queue, task_temp_images, step_callback)

    req.width, req.height = map(lambda x: x - x % 8, (req.width, req.height))  # clamp to 8

    if req.control_image and task_data.control_filter_to_apply:
        req.controlnet_filter = task_data.control_filter_to_apply

    if req.init_image is not None and int(req.num_inference_steps * req.prompt_strength) == 0:
        req.prompt_strength = 1 / req.num_inference_steps if req.num_inference_steps > 0 else 1

    if req.init_image_mask:
        req.init_image_mask = get_image(req.init_image_mask)
        req.init_image_mask = resize_img(req.init_image_mask.convert("RGB"), req.width, req.height, clamp_to_8=True)

    backend.set_options(
        context,
        output_format=output_format.output_format,
        output_quality=output_format.output_quality,
        output_lossless=output_format.output_lossless,
        vae_tiling=task_data.enable_vae_tiling,
        stream_image_progress=stream_image_progress,
        stream_image_progress_interval=stream_image_progress_interval,
        clip_skip=2 if task_data.clip_skip else 1,
    )

    images = backend.generate_images(context, callback=callback, output_type="base64", **req.dict())

    return images


def construct_response(images: list, seeds: list, output_format: OutputFormatData):
    return [ResponseImage(data=img, seed=seed) for img, seed in zip(images, seeds)]


def make_step_callback(
    context,
    req: GenerateImageRequest,
    task_data: RenderTaskData,
    data_queue: queue.Queue,
    task_temp_images: list,
    step_callback,
):
    from easydiffusion.backend_manager import backend

    n_steps = req.num_inference_steps if req.init_image is None else int(req.num_inference_steps * req.prompt_strength)
    last_callback_time = -1

    def update_temp_img(images, task_temp_images: list):
        partial_images = []

        if images is None:
            return []

        if task_data.block_nsfw:
            images = filter_nsfw(images, print_log=False)

        for i, img in enumerate(images):
            img = img.convert("RGB")
            img = resize_img(img, req.width, req.height)
            buf = img_to_buffer(img, output_format="JPEG")

            task_temp_images[i] = buf
            partial_images.append({"path": f"/image/tmp/{task_data.request_id}/{i}"})
        del images
        return partial_images

    def on_image_step(images, i, *args):
        nonlocal last_callback_time

        step_time = time.time() - last_callback_time if last_callback_time != -1 else -1
        last_callback_time = time.time()

        progress = {"step": i, "step_time": step_time, "total_steps": n_steps}

        if images is not None:
            progress["output"] = update_temp_img(images, task_temp_images)

        data_queue.put(json.dumps(progress))

        step_callback()

    return on_image_step
