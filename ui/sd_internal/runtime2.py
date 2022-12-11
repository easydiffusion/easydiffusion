import threading
import queue
import time
import json
import os
import base64
import re
import traceback
import logging

from sd_internal import device_manager, model_manager
from sd_internal import Request, Response, Image as ResponseImage, UserInitiatedStop

from modules import model_loader, image_generator, image_utils, filters as image_filters

log = logging.getLogger()

thread_data = threading.local()
'''
runtime data (bound locally to this thread), for e.g. device, references to loaded models, optimization flags etc
'''

filename_regex = re.compile('[^a-zA-Z0-9]')

def init(device):
    '''
    Initializes the fields that will be bound to this runtime's thread_data, and sets the current torch device
    '''
    thread_data.stop_processing = False
    thread_data.temp_images = {}

    thread_data.models = {}
    thread_data.model_paths = {}

    thread_data.device = None
    thread_data.device_name = None
    thread_data.precision = 'autocast'
    thread_data.vram_optimizations = ('TURBO', 'MOVE_MODELS')

    device_manager.device_init(thread_data, device)

def destroy():
    for model_type in model_manager.KNOWN_MODEL_TYPES:
        model_loader.unload_model(thread_data, model_type)

def load_default_models():
    # init default model paths
    for model_type in model_manager.KNOWN_MODEL_TYPES:
        thread_data.model_paths[model_type] = model_manager.resolve_model_to_use(model_type=model_type)

    # load mandatory models
    model_loader.load_model(thread_data, 'stable-diffusion')

def reload_models_if_necessary(req: Request):
    model_paths_in_req = (
        ('hypernetwork', req.use_hypernetwork_model),
        ('gfpgan', req.use_face_correction),
        ('realesrgan', req.use_upscale),
    )

    if model_manager.is_sd_model_reload_necessary(thread_data, req):
        thread_data.model_paths['stable-diffusion'] = req.use_stable_diffusion_model
        thread_data.model_paths['vae'] = req.use_vae_model

        model_loader.load_model(thread_data, 'stable-diffusion')

    for model_type, model_path_in_req in model_paths_in_req:
        if thread_data.model_paths.get(model_type) != model_path_in_req:
            thread_data.model_paths[model_type] = model_path_in_req

            if thread_data.model_paths[model_type] is not None:
                model_loader.load_model(thread_data, model_type)
            else:
                model_loader.unload_model(thread_data, model_type)

def make_images(req: Request, data_queue: queue.Queue, task_temp_images: list, step_callback):
    try:
        log.info(req)
        return _make_images_internal(req, data_queue, task_temp_images, step_callback)
    except Exception as e:
        log.error(traceback.format_exc())

        data_queue.put(json.dumps({
            "status": 'failed',
            "detail": str(e)
        }))
        raise e

def _make_images_internal(req: Request, data_queue: queue.Queue, task_temp_images: list, step_callback):
    args = req_to_args(req)

    images, user_stopped = generate_images(args, data_queue, task_temp_images, step_callback, req.stream_image_progress)
    images = apply_color_correction(args, images, user_stopped)
    images = apply_filters(args, images, user_stopped, req.show_only_filtered_image)

    if req.save_to_disk_path is not None:
        out_path = os.path.join(req.save_to_disk_path, filename_regex.sub('_', req.session_id))
        save_images(images, out_path, metadata=req.json(), show_only_filtered_image=req.show_only_filtered_image)

    res = Response(req, images=construct_response(req, images))
    res = res.json()
    data_queue.put(json.dumps(res))
    log.info('Task completed')

    return res

def generate_images(args: dict, data_queue: queue.Queue, task_temp_images: list, step_callback, stream_image_progress: bool):
    thread_data.temp_images.clear()

    image_generator.on_image_step = make_step_callback(args, data_queue, task_temp_images, step_callback, stream_image_progress)

    try:
        images = image_generator.make_images(context=thread_data, args=args)
        user_stopped = False
    except UserInitiatedStop:
        images = []
        user_stopped = True
        if not hasattr(thread_data, 'partial_x_samples') or thread_data.partial_x_samples is None:
            return images
        for i in range(args['num_outputs']):
            images[i] = image_utils.latent_to_img(thread_data, thread_data.partial_x_samples[i].unsqueeze(0))
        
        del thread_data.partial_x_samples
    finally:
        model_loader.gc(thread_data)
    
    images = [(image, args['seed'] + i, False) for i, image in enumerate(images)]

    return images, user_stopped

def apply_color_correction(args: dict, images: list, user_stopped):
    if user_stopped or args['init_image'] is None or not args['apply_color_correction']:
        return images

    for i, img_info in enumerate(images):
        img, seed, filtered = img_info
        img = image_utils.apply_color_correction(orig_image=args['init_image'], image_to_correct=img)
        images[i] = (img, seed, filtered)

    return images

def apply_filters(args: dict, images: list, user_stopped, show_only_filtered_image):
    if user_stopped or (args['use_face_correction'] is None and args['use_upscale'] is None):
        return images

    filters = []
    if 'gfpgan' in args['use_face_correction'].lower(): filters.append(image_filters.apply_gfpgan)
    if 'realesrgan' in args['use_face_correction'].lower(): filters.append(image_filters.apply_realesrgan)

    filtered_images = []
    for img, seed, _ in images:
        for filter_fn in filters:
            img = filter_fn(thread_data, img)

        filtered_images.append((img, seed, True))

    if not show_only_filtered_image:
        filtered_images = images + filtered_images

    return filtered_images

def save_images(images: list, save_to_disk_path, metadata: dict, show_only_filtered_image):
    if save_to_disk_path is None:
        return

    def get_image_id(i):
        img_id = base64.b64encode(int(time.time()+i).to_bytes(8, 'big')).decode() # Generate unique ID based on time.
        img_id = img_id.translate({43:None, 47:None, 61:None})[-8:] # Remove + / = and keep last 8 chars.
        return img_id

    def get_image_basepath(i):
        os.makedirs(save_to_disk_path, exist_ok=True)
        prompt_flattened = filename_regex.sub('_', metadata['prompt'])[:50]
        return os.path.join(save_to_disk_path, f"{prompt_flattened}_{get_image_id(i)}")

    for i, img_data in enumerate(images):
        img, seed, filtered = img_data
        img_path = get_image_basepath(i)

        if not filtered or show_only_filtered_image:
            img_metadata_path = img_path + '.txt'
            m = metadata.copy()
            m['seed'] = seed
            with open(img_metadata_path, 'w', encoding='utf-8') as f:
                f.write(m)

        img_path += '_filtered' if filtered else ''
        img_path += '.' + metadata['output_format']
        img.save(img_path, quality=metadata['output_quality'])

def construct_response(req: Request, images: list):
    return [
        ResponseImage(
            data=image_utils.img_to_base64_str(img, req.output_format, req.output_quality),
            seed=seed
        ) for img, seed, _ in images
    ]

def req_to_args(req: Request):
    args = req.json()

    args['init_image'] = image_utils.base64_str_to_img(req.init_image) if req.init_image is not None else None
    args['mask'] = image_utils.base64_str_to_img(req.mask) if req.mask is not None else None

    return args

def make_step_callback(args: dict, data_queue: queue.Queue, task_temp_images: list, step_callback, stream_image_progress: bool):
    n_steps = args['num_inference_steps'] if args['init_image'] is None else int(args['num_inference_steps'] * args['prompt_strength'])
    last_callback_time = -1

    def update_temp_img(x_samples, task_temp_images: list):
        partial_images = []
        for i in range(args['num_outputs']):
            img = image_utils.latent_to_img(thread_data, x_samples[i].unsqueeze(0))
            buf = image_utils.img_to_buffer(img, output_format='JPEG')

            del img

            thread_data.temp_images[f"{args['request_id']}/{i}"] = buf
            task_temp_images[i] = buf
            partial_images.append({'path': f"/image/tmp/{args['request_id']}/{i}"})
        return partial_images

    def on_image_step(x_samples, i):
        nonlocal last_callback_time

        thread_data.partial_x_samples = x_samples
        step_time = time.time() - last_callback_time if last_callback_time != -1 else -1
        last_callback_time = time.time()

        progress = {"step": i, "step_time": step_time, "total_steps": n_steps}

        if stream_image_progress and i % 5 == 0:
            progress['output'] = update_temp_img(x_samples, task_temp_images)

        data_queue.put(json.dumps(progress))

        step_callback()

        if thread_data.stop_processing:
            raise UserInitiatedStop("User requested that we stop processing")

    return on_image_step
