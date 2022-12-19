import queue
import time
import json
import logging

from sd_internal import device_manager, save_utils
from sd_internal import TaskData, Response, Image as ResponseImage, UserInitiatedStop

from sdkit import model_loader, image_generator, image_utils, filters as image_filters
from sdkit.types import Context, GenerateImageRequest, FilterImageRequest

log = logging.getLogger()

context = Context() # thread-local
'''
runtime data (bound locally to this thread), for e.g. device, references to loaded models, optimization flags etc
'''

def init(device):
    '''
    Initializes the fields that will be bound to this runtime's context, and sets the current torch device
    '''
    context.stop_processing = False
    context.temp_images = {}
    context.partial_x_samples = None

    device_manager.device_init(context, device)

def make_images(req: GenerateImageRequest, task_data: TaskData, data_queue: queue.Queue, task_temp_images: list, step_callback):
    context.stop_processing = False
    log.info(f'request: {save_utils.get_printable_request(req)}')
    log.info(f'task data: {task_data.dict()}')

    images = _make_images_internal(req, task_data, data_queue, task_temp_images, step_callback)

    res = Response(req, task_data, images=construct_response(images, task_data, base_seed=req.seed))
    res = res.json()
    data_queue.put(json.dumps(res))
    log.info('Task completed')

    return res

def _make_images_internal(req: GenerateImageRequest, task_data: TaskData, data_queue: queue.Queue, task_temp_images: list, step_callback):
    images, user_stopped = generate_images(req, task_data, data_queue, task_temp_images, step_callback, task_data.stream_image_progress)
    filtered_images = apply_filters(task_data, images, user_stopped)

    if task_data.save_to_disk_path is not None:
        save_utils.save_to_disk(images, filtered_images, req, task_data)

    return filtered_images if task_data.show_only_filtered_image else images + filtered_images

def generate_images(req: GenerateImageRequest, task_data: TaskData, data_queue: queue.Queue, task_temp_images: list, step_callback, stream_image_progress: bool):
    context.temp_images.clear()

    image_generator.on_image_step = make_step_callback(req, task_data, data_queue, task_temp_images, step_callback, stream_image_progress)

    try:
        images = image_generator.make_images(context=context, req=req)
        user_stopped = False
    except UserInitiatedStop:
        images = []
        user_stopped = True
        if context.partial_x_samples is not None:
            images = image_utils.latent_samples_to_images(context, context.partial_x_samples)
            context.partial_x_samples = None
    finally:
        model_loader.gc(context)

    return images, user_stopped

def apply_filters(task_data: TaskData, images: list, user_stopped):
    if user_stopped or (task_data.use_face_correction is None and task_data.use_upscale is None):
        return images

    filters = []
    if 'gfpgan' in task_data.use_face_correction.lower(): filters.append('gfpgan')
    if 'realesrgan' in task_data.use_face_correction.lower(): filters.append('realesrgan')

    filtered_images = []
    for img in images:
        filter_req = FilterImageRequest()
        filter_req.init_image = img

        filtered_image = image_filters.apply(context, filters, filter_req)
        filtered_images.append(filtered_image)

    return filtered_images

def construct_response(images: list, task_data: TaskData, base_seed: int):
    return [
        ResponseImage(
            data=image_utils.img_to_base64_str(img, task_data.output_format, task_data.output_quality),
            seed=base_seed + i
        ) for i, img in enumerate(images)
    ]

def make_step_callback(req: GenerateImageRequest, task_data: TaskData, data_queue: queue.Queue, task_temp_images: list, step_callback, stream_image_progress: bool):
    n_steps = req.num_inference_steps if req.init_image is None else int(req.num_inference_steps * req.prompt_strength)
    last_callback_time = -1

    def update_temp_img(x_samples, task_temp_images: list):
        partial_images = []
        images = image_utils.latent_samples_to_images(context, x_samples)
        for i, img in enumerate(images):
            buf = image_utils.img_to_buffer(img, output_format='JPEG')

            context.temp_images[f"{task_data.request_id}/{i}"] = buf
            task_temp_images[i] = buf
            partial_images.append({'path': f"/image/tmp/{task_data.request_id}/{i}"})
        del images
        return partial_images

    def on_image_step(x_samples, i):
        nonlocal last_callback_time

        context.partial_x_samples = x_samples
        step_time = time.time() - last_callback_time if last_callback_time != -1 else -1
        last_callback_time = time.time()

        progress = {"step": i, "step_time": step_time, "total_steps": n_steps}

        if stream_image_progress and i % 5 == 0:
            progress['output'] = update_temp_img(x_samples, task_temp_images)

        data_queue.put(json.dumps(progress))

        step_callback()

        if context.stop_processing:
            raise UserInitiatedStop("User requested that we stop processing")

    return on_image_step
