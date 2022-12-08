import threading
import queue

from sd_internal import device_manager, model_manager
from sd_internal import Request, Response, Image as ResponseImage

from modules import model_loader, image_generator, image_utils

thread_data = threading.local()
'''
runtime data (bound locally to this thread), for e.g. device, references to loaded models, optimization flags etc
'''

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

    load_default_models()

def destroy():
    model_loader.unload_sd_model(thread_data)
    model_loader.unload_gfpgan_model(thread_data)
    model_loader.unload_realesrgan_model(thread_data)

def load_default_models():
    thread_data.model_paths['stable-diffusion'] = model_manager.default_model_to_load
    thread_data.model_paths['vae'] = model_manager.default_vae_to_load

    model_loader.load_sd_model(thread_data)

def reload_models_if_necessary(req: Request=None):
    needs_model_reload = False
    if 'stable-diffusion' not in thread_data.models or thread_data.ckpt_file != req.use_stable_diffusion_model or thread_data.vae_file != req.use_vae_model:
        thread_data.ckpt_file = req.use_stable_diffusion_model
        thread_data.vae_file = req.use_vae_model
        needs_model_reload = True

    if thread_data.device != 'cpu':
        if (thread_data.precision == 'autocast' and (req.use_full_precision or not thread_data.model_is_half)) or \
            (thread_data.precision == 'full' and not req.use_full_precision and not thread_data.force_full_precision):
            thread_data.precision = 'full' if req.use_full_precision else 'autocast'
            needs_model_reload = True

    return needs_model_reload

    if is_hypernetwork_reload_necessary(task.request):
        current_state = ServerStates.LoadingModel
        runtime.reload_hypernetwork()

    if is_model_reload_necessary(task.request):
        current_state = ServerStates.LoadingModel
        runtime.reload_model()

def load_models():
    if ckpt_file_path == None:
        ckpt_file_path = default_model_to_load
    if vae_file_path == None:
        vae_file_path = default_vae_to_load
    if hypernetwork_file_path == None:
        hypernetwork_file_path = default_hypernetwork_to_load
    if ckpt_file_path == current_model_path and vae_file_path == current_vae_path:
        return
    current_state = ServerStates.LoadingModel
    try:
        from sd_internal import runtime2
        runtime.thread_data.hypernetwork_file = hypernetwork_file_path
        runtime.thread_data.ckpt_file = ckpt_file_path
        runtime.thread_data.vae_file = vae_file_path
        runtime.load_model_ckpt()
        runtime.load_hypernetwork()
        current_model_path = ckpt_file_path
        current_vae_path = vae_file_path
        current_hypernetwork_path = hypernetwork_file_path
        current_state_error = None
        current_state = ServerStates.Online
    except Exception as e:
        current_model_path = None
        current_vae_path = None
        current_state_error = e
        current_state = ServerStates.Unavailable
        print(traceback.format_exc())

def make_image(req: Request, data_queue: queue.Queue, task_temp_images: list, step_callback):
    try:
        images = image_generator.make_image(context=thread_data, args=get_mk_img_args(req))
    except UserInitiatedStop:
        pass

def get_mk_img_args(req: Request):
    args = req.json()

    if req.init_image is not None:
        args['init_image'] = image_utils.base64_str_to_img(req.init_image)

    if req.mask is not None:
        args['mask'] = image_utils.base64_str_to_img(req.mask)

    return args

def on_image_step(x_samples, i):
    pass

image_generator.on_image_step = on_image_step
