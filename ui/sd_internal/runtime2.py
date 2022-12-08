import threading
import queue
import time
import json
import os
import base64
import re

from sd_internal import device_manager, model_manager
from sd_internal import Request, Response, Image as ResponseImage, UserInitiatedStop

from modules import model_loader, image_generator, image_utils, image_filters

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

    init_and_load_default_models()

def destroy():
    model_loader.unload_sd_model(thread_data)
    model_loader.unload_gfpgan_model(thread_data)
    model_loader.unload_realesrgan_model(thread_data)

def init_and_load_default_models():
    # init default model paths
    thread_data.model_paths['stable-diffusion'] = model_manager.resolve_sd_model_to_use()
    thread_data.model_paths['vae'] = model_manager.resolve_vae_model_to_use()
    thread_data.model_paths['hypernetwork'] = model_manager.resolve_hypernetwork_model_to_use()
    thread_data.model_paths['gfpgan'] = model_manager.resolve_gfpgan_model_to_use()
    thread_data.model_paths['realesrgan'] = model_manager.resolve_realesrgan_model_to_use()

    # load mandatory models
    model_loader.load_sd_model(thread_data)

def reload_models_if_necessary(req: Request):
    if model_manager.is_sd_model_reload_necessary(thread_data, req):
        thread_data.model_paths['stable-diffusion'] = req.use_stable_diffusion_model
        thread_data.model_paths['vae'] = req.use_vae_model

        model_loader.load_sd_model(thread_data)

    # if is_hypernetwork_reload_necessary(task.request):
    #     current_state = ServerStates.LoadingModel
    #     runtime.reload_hypernetwork()

def make_images(req: Request, data_queue: queue.Queue, task_temp_images: list, step_callback):
    images, user_stopped = generate_images(req, data_queue, task_temp_images, step_callback)
    images = apply_filters(req, images, user_stopped)

    save_images(req, images)

    return Response(req, images=construct_response(req, images))

def generate_images(req: Request, data_queue: queue.Queue, task_temp_images: list, step_callback):
    thread_data.temp_images.clear()

    image_generator.on_image_step = make_step_callback(req, data_queue, task_temp_images, step_callback)

    try:
        images = image_generator.make_image(context=thread_data, args=get_mk_img_args(req))
        user_stopped = False
    except UserInitiatedStop:
        images = []
        user_stopped = True
        if not hasattr(thread_data, 'partial_x_samples') or thread_data.partial_x_samples is None:
            return images
        for i in range(req.num_outputs):
            images[i] = image_utils.latent_to_img(thread_data, thread_data.partial_x_samples[i].unsqueeze(0))
        
        del thread_data.partial_x_samples
    finally:
        model_loader.gc(thread_data)
    
    images = [(image, req.seed + i, False) for i, image in enumerate(images)]

    return images, user_stopped

def apply_filters(req: Request, images: list, user_stopped):
    if user_stopped or (req.use_face_correction is None and req.use_upscale is None):
        return images

    filters = []
    if req.use_face_correction.startswith('GFPGAN'): filters.append((image_filters.apply_gfpgan, model_manager.resolve_gfpgan_model_to_use(req.use_face_correction)))
    if req.use_upscale.startswith('RealESRGAN'): filters.append((image_filters.apply_realesrgan, model_manager.resolve_realesrgan_model_to_use(req.use_upscale)))

    filtered_images = []
    for img, seed, _ in images:
        for filter_fn, filter_model_path in filters:
            img = filter_fn(thread_data, img, filter_model_path)

        filtered_images.append((img, seed, True))

    if not req.show_only_filtered_image:
        filtered_images = images + filtered_images

    return filtered_images

def save_images(req: Request, images: list):
    if req.save_to_disk_path is None:
        return

    def get_image_id(i):
        img_id = base64.b64encode(int(time.time()+i).to_bytes(8, 'big')).decode() # Generate unique ID based on time.
        img_id = img_id.translate({43:None, 47:None, 61:None})[-8:] # Remove + / = and keep last 8 chars.
        return img_id

    def get_image_basepath(i):
        session_out_path = os.path.join(req.save_to_disk_path, filename_regex.sub('_', req.session_id))
        os.makedirs(session_out_path, exist_ok=True)
        prompt_flattened = filename_regex.sub('_', req.prompt)[:50]
        return os.path.join(session_out_path, f"{prompt_flattened}_{get_image_id(i)}")

    for i, img_data in enumerate(images):
        img, seed, filtered = img_data
        img_path = get_image_basepath(i)

        if not filtered or req.show_only_filtered_image:
            img_metadata_path = img_path + '.txt'
            metadata = req.json()
            metadata['seed'] = seed
            with open(img_metadata_path, 'w', encoding='utf-8') as f:
                f.write(metadata)

        img_path += '_filtered' if filtered else ''
        img_path += '.' + req.output_format
        img.save(img_path, quality=req.output_quality)

def construct_response(req: Request, images: list):
    return [
        ResponseImage(
            data=image_utils.img_to_base64_str(img, req.output_format, req.output_quality),
            seed=seed
        ) for img, seed, _ in images
    ]

def get_mk_img_args(req: Request):
    args = req.json()

    args['init_image'] = image_utils.base64_str_to_img(req.init_image) if req.init_image is not None else None
    args['mask'] = image_utils.base64_str_to_img(req.mask) if req.mask is not None else None

    return args

def make_step_callback(req: Request, data_queue: queue.Queue, task_temp_images: list, step_callback):
    n_steps = req.num_inference_steps if req.init_image is None else int(req.num_inference_steps * req.prompt_strength)
    last_callback_time = -1

    def update_temp_img(req, x_samples, task_temp_images: list):
        partial_images = []
        for i in range(req.num_outputs):
            img = image_utils.latent_to_img(thread_data, x_samples[i].unsqueeze(0))
            buf = image_utils.img_to_buffer(img, output_format='JPEG')

            del img

            thread_data.temp_images[f'{req.request_id}/{i}'] = buf
            task_temp_images[i] = buf
            partial_images.append({'path': f'/image/tmp/{req.request_id}/{i}'})
        return partial_images

    def on_image_step(x_samples, i):
        nonlocal last_callback_time

        thread_data.partial_x_samples = x_samples
        step_time = time.time() - last_callback_time if last_callback_time != -1 else -1
        last_callback_time = time.time()

        progress = {"step": i, "step_time": step_time, "total_steps": n_steps}

        if req.stream_image_progress and i % 5 == 0:
            progress['output'] = update_temp_img(req, x_samples, task_temp_images)

        data_queue.put(json.dumps(progress))

        step_callback()

        if thread_data.stop_processing:
            raise UserInitiatedStop("User requested that we stop processing")

    return on_image_step
