import json
import pprint
import queue
import time

from easydiffusion import model_manager, runtime
from easydiffusion.types import GenerateImageRequest, ModelsData, OutputFormatData
from easydiffusion.types import Image as ResponseImage
from easydiffusion.types import GenerateImageResponse, TaskData, UserInitiatedStop
from easydiffusion.utils import get_printable_request, log, save_images_to_disk
from sdkit.generate import generate_images
from sdkit.utils import (
    diffusers_latent_samples_to_images,
    gc,
    img_to_base64_str,
    img_to_buffer,
    latent_samples_to_images,
    resize_img,
    get_image,
    log,
)

from .task import Task
from .filter_images import filter_images


class RenderTask(Task):
    "For image generation"

    def __init__(
        self, req: GenerateImageRequest, task_data: TaskData, models_data: ModelsData, output_format: OutputFormatData
    ):
        super().__init__(task_data.session_id)

        task_data.request_id = self.id
        self.render_request: GenerateImageRequest = req  # Initial Request
        self.task_data: TaskData = task_data
        self.models_data = models_data
        self.output_format = output_format
        self.temp_images: list = [None] * req.num_outputs * (1 if task_data.show_only_filtered_image else 2)

    def run(self):
        "Runs the image generation task on the assigned thread"

        from easydiffusion import task_manager

        context = runtime.context

        def step_callback():
            task_manager.keep_task_alive(self)
            task_manager.current_state = task_manager.ServerStates.Rendering

            if isinstance(task_manager.current_state_error, (SystemExit, StopAsyncIteration)) or isinstance(
                self.error, StopAsyncIteration
            ):
                context.stop_processing = True
                if isinstance(task_manager.current_state_error, StopAsyncIteration):
                    self.error = task_manager.current_state_error
                    task_manager.current_state_error = None
                    log.info(f"Session {self.session_id} sent cancel signal for task {self.id}")

        task_manager.current_state = task_manager.ServerStates.LoadingModel
        model_manager.resolve_model_paths(self.models_data)

        models_to_force_reload = []
        if (
            runtime.set_vram_optimizations(context)
            or self.has_param_changed(context, "clip_skip")
            or self.trt_needs_reload(context)
        ):
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
            self.buffer_queue,
            self.temp_images,
            step_callback,
        )

    def has_param_changed(self, context, param_name):
        if not context.test_diffusers:
            return False
        if "stable-diffusion" not in context.models or "params" not in context.models["stable-diffusion"]:
            return True

        model = context.models["stable-diffusion"]
        new_val = self.models_data.model_params.get("stable-diffusion", {}).get(param_name, False)
        return model["params"].get(param_name) != new_val

    def trt_needs_reload(self, context):
        if not context.test_diffusers:
            return False
        if "stable-diffusion" not in context.models or "params" not in context.models["stable-diffusion"]:
            return True

        model = context.models["stable-diffusion"]

        # curr_convert_to_trt = model["params"].get("convert_to_tensorrt")
        new_convert_to_trt = self.models_data.model_params.get("stable-diffusion", {}).get("convert_to_tensorrt", False)

        pipe = model["default"]
        is_trt_loaded = hasattr(pipe.unet, "_allocate_trt_buffers") or hasattr(
            pipe.unet, "_allocate_trt_buffers_backup"
        )
        if new_convert_to_trt and not is_trt_loaded:
            return True

        curr_build_config = model["params"].get("trt_build_config")
        new_build_config = self.models_data.model_params.get("stable-diffusion", {}).get("trt_build_config", {})

        return new_convert_to_trt and curr_build_config != new_build_config


def make_images(
    context,
    req: GenerateImageRequest,
    task_data: TaskData,
    models_data: ModelsData,
    output_format: OutputFormatData,
    data_queue: queue.Queue,
    task_temp_images: list,
    step_callback,
):
    context.stop_processing = False
    print_task_info(req, task_data, models_data, output_format)

    images, seeds = make_images_internal(
        context, req, task_data, models_data, output_format, data_queue, task_temp_images, step_callback
    )

    res = GenerateImageResponse(
        req, task_data, models_data, output_format, images=construct_response(images, seeds, output_format)
    )
    res = res.json()
    data_queue.put(json.dumps(res))
    log.info("Task completed")

    return res


def print_task_info(
    req: GenerateImageRequest, task_data: TaskData, models_data: ModelsData, output_format: OutputFormatData
):
    req_str = pprint.pformat(get_printable_request(req, task_data, output_format)).replace("[", "\[")
    task_str = pprint.pformat(task_data.dict()).replace("[", "\[")
    models_data = pprint.pformat(models_data.dict()).replace("[", "\[")
    output_format = pprint.pformat(output_format.dict()).replace("[", "\[")

    log.info(f"request: {req_str}")
    log.info(f"task data: {task_str}")
    # log.info(f"models data: {models_data}")
    log.info(f"output format: {output_format}")


def make_images_internal(
    context,
    req: GenerateImageRequest,
    task_data: TaskData,
    models_data: ModelsData,
    output_format: OutputFormatData,
    data_queue: queue.Queue,
    task_temp_images: list,
    step_callback,
):
    images, user_stopped = generate_images_internal(
        context,
        req,
        task_data,
        models_data,
        data_queue,
        task_temp_images,
        step_callback,
        task_data.stream_image_progress,
        task_data.stream_image_progress_interval,
    )

    gc(context)

    filters, filter_params = task_data.filters, task_data.filter_params
    filtered_images = filter_images(context, images, filters, filter_params) if not user_stopped else images

    if task_data.save_to_disk_path is not None:
        save_images_to_disk(images, filtered_images, req, task_data, output_format)

    seeds = [*range(req.seed, req.seed + len(images))]
    if task_data.show_only_filtered_image or filtered_images is images:
        return filtered_images, seeds
    else:
        return images + filtered_images, seeds + seeds


def generate_images_internal(
    context,
    req: GenerateImageRequest,
    task_data: TaskData,
    models_data: ModelsData,
    data_queue: queue.Queue,
    task_temp_images: list,
    step_callback,
    stream_image_progress: bool,
    stream_image_progress_interval: int,
):
    context.temp_images.clear()

    callback = make_step_callback(
        context,
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

        req.width, req.height = map(lambda x: x - x % 8, (req.width, req.height))  # clamp to 8

        if req.control_image and task_data.control_filter_to_apply:
            req.control_image = get_image(req.control_image)
            req.control_image = resize_img(req.control_image.convert("RGB"), req.width, req.height, clamp_to_8=True)
            req.control_image = filter_images(context, req.control_image, task_data.control_filter_to_apply)[0]

        if context.test_diffusers:
            pipe = context.models["stable-diffusion"]["default"]
            if hasattr(pipe.unet, "_allocate_trt_buffers_backup"):
                setattr(pipe.unet, "_allocate_trt_buffers", pipe.unet._allocate_trt_buffers_backup)
                delattr(pipe.unet, "_allocate_trt_buffers_backup")

            if hasattr(pipe.unet, "_allocate_trt_buffers"):
                convert_to_trt = models_data.model_params["stable-diffusion"].get("convert_to_tensorrt", False)
                if convert_to_trt:
                    pipe.unet.forward = pipe.unet._trt_forward
                    # pipe.vae.decoder.forward = pipe.vae.decoder._trt_forward
                    log.info(f"Setting unet.forward to TensorRT")
                else:
                    log.info(f"Not using TensorRT for unet.forward")
                    pipe.unet.forward = pipe.unet._non_trt_forward
                    # pipe.vae.decoder.forward = pipe.vae.decoder._non_trt_forward
                    setattr(pipe.unet, "_allocate_trt_buffers_backup", pipe.unet._allocate_trt_buffers)
                    delattr(pipe.unet, "_allocate_trt_buffers")

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


def construct_response(images: list, seeds: list, output_format: OutputFormatData):
    return [
        ResponseImage(
            data=img_to_base64_str(
                img,
                output_format.output_format,
                output_format.output_quality,
                output_format.output_lossless,
            ),
            seed=seed,
        )
        for img, seed in zip(images, seeds)
    ]


def make_step_callback(
    context,
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
            images = filter_images(context, images, "nsfw_checker")

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
