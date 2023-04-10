import queue
import time
import json
import pprint

from easydiffusion import device_manager
from easydiffusion.types import TaskData, Response, Image as ResponseImage, UserInitiatedStop, GenerateImageRequest
from easydiffusion.utils import get_printable_request, save_images_to_disk, log

from sdkit import Context
from sdkit.generate import generate_images
from sdkit.filter import apply_filters
from sdkit.utils import (
    img_to_buffer,
    img_to_base64_str,
    latent_samples_to_images,
    diffusers_latent_samples_to_images,
    gc,
)

context = Context()  # thread-local
"""
runtime data (bound locally to this thread), for e.g. device, references to loaded models, optimization flags etc
"""


def init(device):
    """
    Initializes the fields that will be bound to this runtime's context, and sets the current torch device
    """
    context.stop_processing = False
    context.temp_images = {}
    context.partial_x_samples = None

    from easydiffusion import app

    app_config = app.getConfig()
    context.test_diffusers = (
        app_config.get("test_diffusers", False) and app_config.get("update_branch", "main") != "main"
    )

    device_manager.device_init(context, device)


def make_images(
    req: GenerateImageRequest, task_data: TaskData, data_queue: queue.Queue, task_temp_images: list, step_callback
):
    context.stop_processing = False
    print_task_info(req, task_data)

    images, seeds = make_images_internal(req, task_data, data_queue, task_temp_images, step_callback)

    res = Response(req, task_data, images=construct_response(images, seeds, task_data, base_seed=req.seed))
    res = res.json()
    data_queue.put(json.dumps(res))
    log.info("Task completed")

    return res


def print_task_info(req: GenerateImageRequest, task_data: TaskData):
    req_str = pprint.pformat(get_printable_request(req)).replace("[", "\[")
    task_str = pprint.pformat(task_data.dict()).replace("[", "\[")
    log.info(f"request: {req_str}")
    log.info(f"task data: {task_str}")


def make_images_internal(
    req: GenerateImageRequest, task_data: TaskData, data_queue: queue.Queue, task_temp_images: list, step_callback
):
    images, user_stopped = generate_images_internal(
        req,
        task_data,
        data_queue,
        task_temp_images,
        step_callback,
        task_data.stream_image_progress,
        task_data.stream_image_progress_interval,
    )
    gc(context)
    filtered_images = filter_images(task_data, images, user_stopped)

    if task_data.save_to_disk_path is not None:
        save_images_to_disk(images, filtered_images, req, task_data)

    seeds = [*range(req.seed, req.seed + len(images))]
    if task_data.show_only_filtered_image or filtered_images is images:
        return filtered_images, seeds
    else:
        return images + filtered_images, seeds + seeds


def generate_images_internal(
    req: GenerateImageRequest,
    task_data: TaskData,
    data_queue: queue.Queue,
    task_temp_images: list,
    step_callback,
    stream_image_progress: bool,
    stream_image_progress_interval: int,
):
    context.temp_images.clear()

    callback = make_step_callback(
        req,
        task_data,
        data_queue,
        task_temp_images,
        step_callback,
        stream_image_progress,
        stream_image_progress_interval,
    )

    try:
        if req.init_image is not None and not context.test_diffusers:
            req.sampler_name = "ddim"

        images = generate_images(context, callback=callback, **req.dict())
        user_stopped = False
    except UserInitiatedStop:
        images = []
        user_stopped = True
        if context.partial_x_samples is not None:
            if context.test_diffusers:
                images = diffusers_latent_samples_to_images(context, context.partial_x_samples)
            else:
                images = latent_samples_to_images(context, context.partial_x_samples)
    finally:
        if hasattr(context, "partial_x_samples") and context.partial_x_samples is not None:
            if not context.test_diffusers:
                del context.partial_x_samples
            context.partial_x_samples = None

    return images, user_stopped


def filter_images(task_data: TaskData, images: list, user_stopped):
    if user_stopped:
        return images

    filters_to_apply = []
    if task_data.block_nsfw:
        filters_to_apply.append("nsfw_checker")
    if task_data.use_face_correction and "gfpgan" in task_data.use_face_correction.lower():
        filters_to_apply.append("gfpgan")
    if task_data.use_upscale and "realesrgan" in task_data.use_upscale.lower():
        filters_to_apply.append("realesrgan")

    if len(filters_to_apply) == 0:
        return images

    return apply_filters(context, filters_to_apply, images, scale=task_data.upscale_amount)


def construct_response(images: list, seeds: list, task_data: TaskData, base_seed: int):
    return [
        ResponseImage(
            data=img_to_base64_str(img, task_data.output_format, task_data.output_quality, task_data.output_lossless),
            seed=seed,
        )
        for img, seed in zip(images, seeds)
    ]


def make_step_callback(
    req: GenerateImageRequest,
    task_data: TaskData,
    data_queue: queue.Queue,
    task_temp_images: list,
    step_callback,
    stream_image_progress: bool,
    stream_image_progress_interval: int,
):
    n_steps = req.num_inference_steps if req.init_image is None else int(req.num_inference_steps * req.prompt_strength)
    last_callback_time = -1

    def update_temp_img(x_samples, task_temp_images: list):
        partial_images = []

        if context.test_diffusers:
            images = diffusers_latent_samples_to_images(context, x_samples)
        else:
            images = latent_samples_to_images(context, x_samples)

        if task_data.block_nsfw:
            images = apply_filters(context, "nsfw_checker", images)

        for i, img in enumerate(images):
            buf = img_to_buffer(img, output_format="JPEG")

            context.temp_images[f"{task_data.request_id}/{i}"] = buf
            task_temp_images[i] = buf
            partial_images.append({"path": f"/image/tmp/{task_data.request_id}/{i}"})
        del images
        return partial_images

    def on_image_step(x_samples, i, *args):
        nonlocal last_callback_time

        if context.test_diffusers:
            context.partial_x_samples = (x_samples, args[0])
        else:
            context.partial_x_samples = x_samples

        step_time = time.time() - last_callback_time if last_callback_time != -1 else -1
        last_callback_time = time.time()

        progress = {"step": i, "step_time": step_time, "total_steps": n_steps}

        if stream_image_progress and stream_image_progress_interval > 0 and i % stream_image_progress_interval == 0:
            progress["output"] = update_temp_img(context.partial_x_samples, task_temp_images)

        data_queue.put(json.dumps(progress))

        step_callback()

        if context.stop_processing:
            raise UserInitiatedStop("User requested that we stop processing")

    return on_image_step
