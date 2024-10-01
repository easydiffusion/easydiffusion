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
