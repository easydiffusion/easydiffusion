import queue
import time
import json
import os
import base64
import re
import traceback
import logging

from sd_internal import device_manager
from sd_internal import TaskData, Response, Image as ResponseImage, UserInitiatedStop

from modules import model_loader, image_generator, image_utils, filters as image_filters, data_utils
from modules.types import Context, GenerateImageRequest

log = logging.getLogger()

context = Context() # thread-local
'''
runtime data (bound locally to this thread), for e.g. device, references to loaded models, optimization flags etc
'''

filename_regex = re.compile('[^a-zA-Z0-9]')

def init(device):
    '''
    Initializes the fields that will be bound to this runtime's context, and sets the current torch device
    '''
    context.stop_processing = False
    context.temp_images = {}
    context.partial_x_samples = None

    device_manager.device_init(context, device)

def make_images(req: GenerateImageRequest, task_data: TaskData, data_queue: queue.Queue, task_temp_images: list, step_callback):
    log.info(f'request: {req.dict()}')
    log.info(f'task data: {task_data.dict()}')

    try:
        images = _make_images_internal(req, task_data, data_queue, task_temp_images, step_callback)

        res = Response(req, task_data, images=construct_response(images, task_data, base_seed=req.seed))
        res = res.json()
        data_queue.put(json.dumps(res))
        log.info('Task completed')

        return res
    except Exception as e:
        log.error(traceback.format_exc())

        data_queue.put(json.dumps({
            "status": 'failed',
            "detail": str(e)
        }))
        raise e

def _make_images_internal(req: GenerateImageRequest, task_data: TaskData, data_queue: queue.Queue, task_temp_images: list, step_callback):
    images, user_stopped = generate_images(req, data_queue, task_temp_images, step_callback, task_data.stream_image_progress)
    filtered_images = apply_filters(task_data, images, user_stopped)

    if task_data.save_to_disk_path is not None:
        save_folder_path = os.path.join(task_data.save_to_disk_path, filename_regex.sub('_', task_data.session_id))
        save_to_disk(images, filtered_images, save_folder_path, req, task_data)

    return filtered_images if task_data.show_only_filtered_image else images + filtered_images

def generate_images(req: GenerateImageRequest, data_queue: queue.Queue, task_temp_images: list, step_callback, stream_image_progress: bool):
    context.temp_images.clear()

    image_generator.on_image_step = make_step_callback(req, data_queue, task_temp_images, step_callback, stream_image_progress)

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
    if 'gfpgan' in task_data.use_face_correction.lower(): filters.append(image_filters.apply_gfpgan)
    if 'realesrgan' in task_data.use_face_correction.lower(): filters.append(image_filters.apply_realesrgan)

    filtered_images = []
    for img in images:
        for filter_fn in filters:
            img = filter_fn(context, img)

        filtered_images.append(img)

    return filtered_images

def save_to_disk(images: list, filtered_images: list, save_folder_path, req: GenerateImageRequest, task_data: TaskData):
    metadata_entries = get_metadata_entries(req, task_data)

    if task_data.show_only_filtered_image or filtered_images == images:
        data_utils.save_images(filtered_images, save_folder_path, file_name=make_filename_callback(req), output_format=task_data.output_format, output_quality=task_data.output_quality)
        data_utils.save_metadata(metadata_entries, save_folder_path, file_name=make_filename_callback(req), output_format=task_data.metadata_output_format)
    else:
        data_utils.save_images(images, save_folder_path, file_name=make_filename_callback(req), output_format=task_data.output_format, output_quality=task_data.output_quality)
        data_utils.save_images(filtered_images, save_folder_path, file_name=make_filename_callback(req, suffix='filtered'), output_format=task_data.output_format, output_quality=task_data.output_quality)
        data_utils.save_metadata(metadata_entries, save_folder_path, file_name=make_filename_callback(req, suffix='filtered'), output_format=task_data.metadata_output_format)

def construct_response(images: list, task_data: TaskData, base_seed: int):
    return [
        ResponseImage(
            data=image_utils.img_to_base64_str(img, task_data.output_format, task_data.output_quality),
            seed=base_seed + i
        ) for i, img in enumerate(images)
    ]

def get_metadata_entries(req: GenerateImageRequest, task_data: TaskData):
    metadata = req.dict()
    del metadata['init_image']
    del metadata['init_image_mask']
    metadata.update({
        'use_stable_diffusion_model': task_data.use_stable_diffusion_model,
        'use_vae_model': task_data.use_vae_model,
        'use_hypernetwork_model': task_data.use_hypernetwork_model,
        'use_face_correction': task_data.use_face_correction,
        'use_upscale': task_data.use_upscale,
    })

    return [metadata.copy().update({'seed': req.seed + i}) for i in range(req.num_outputs)]

def make_filename_callback(req: GenerateImageRequest, suffix=None):
    def make_filename(i):
        img_id = base64.b64encode(int(time.time()+i).to_bytes(8, 'big')).decode() # Generate unique ID based on time.
        img_id = img_id.translate({43:None, 47:None, 61:None})[-8:] # Remove + / = and keep last 8 chars.

        prompt_flattened = filename_regex.sub('_', req.prompt)[:50]
        name = f"{prompt_flattened}_{img_id}"
        name = name if suffix is None else f'{name}_{suffix}'
        return name

    return make_filename

def make_step_callback(req: GenerateImageRequest, task_data: TaskData, data_queue: queue.Queue, task_temp_images: list, step_callback, stream_image_progress: bool):
    n_steps = req.num_inference_steps if req.init_image is None else int(req.num_inference_steps * req.prompt_strength)
    last_callback_time = -1

    def update_temp_img(x_samples, task_temp_images: list):
        partial_images = []
        for i in range(req.num_outputs):
            img = image_utils.latent_to_img(context, x_samples[i].unsqueeze(0))
            buf = image_utils.img_to_buffer(img, output_format='JPEG')

            del img

            context.temp_images[f"{task_data.request_id}/{i}"] = buf
            task_temp_images[i] = buf
            partial_images.append({'path': f"/image/tmp/{task_data.request_id}/{i}"})
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
