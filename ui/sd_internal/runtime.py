"""runtime.py: torch device owned by a thread.
Notes:
    Avoid device switching, transfering all models will get too complex.
    To use a diffrent device signal the current render device to exit
    And then start a new clean thread for the new device.
"""
import json
import os, re
import traceback
import queue
import torch
import numpy as np
from gc import collect as gc_collect
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from einops import rearrange, repeat
from ldm.util import instantiate_from_config
from transformers import logging

from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from server import HYPERNETWORK_MODEL_EXTENSIONS# , STABLE_DIFFUSION_MODEL_EXTENSIONS, VAE_MODEL_EXTENSIONS

from threading import Lock
from safetensors.torch import load_file

import uuid

logging.set_verbosity_error()

# consts
config_yaml = "optimizedSD/v1-inference.yaml"
filename_regex = re.compile('[^a-zA-Z0-9]')
gfpgan_temp_device_lock = Lock() # workaround: gfpgan currently can only start on one device at a time.

# api stuff
from sd_internal import device_manager
from . import Request, Response, Image as ResponseImage
import base64
from io import BytesIO
#from colorama import Fore

from threading import local as LocalThreadVars
thread_data = LocalThreadVars()

def thread_init(device):
    # Thread bound properties
    thread_data.stop_processing = False
    thread_data.temp_images = {}

    thread_data.ckpt_file = None
    thread_data.vae_file = None
    thread_data.hypernetwork_file = None
    thread_data.gfpgan_file = None
    thread_data.real_esrgan_file = None

    thread_data.model = None
    thread_data.modelCS = None
    thread_data.modelFS = None
    thread_data.hypernetwork = None
    thread_data.hypernetwork_strength = 1
    thread_data.model_gfpgan = None
    thread_data.model_real_esrgan = None

    thread_data.model_is_half = False
    thread_data.model_fs_is_half = False
    thread_data.device = None
    thread_data.device_name = None
    thread_data.unet_bs = 1
    thread_data.precision = 'autocast'
    thread_data.sampler_plms = None
    thread_data.sampler_ddim = None

    thread_data.turbo = False
    thread_data.force_full_precision = False
    thread_data.reduced_memory = True

    thread_data.test_sd2 = isSD2()

    device_manager.device_init(thread_data, device)

# temp hack, will remove soon
def isSD2():
    try:
        SD_UI_DIR = os.getenv('SD_UI_PATH', None)
        CONFIG_DIR = os.path.abspath(os.path.join(SD_UI_DIR, '..', 'scripts'))
        config_json_path = os.path.join(CONFIG_DIR, 'config.json')
        if not os.path.exists(config_json_path):
            return False
        with open(config_json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get('test_sd2', False)
    except Exception as e:
        return False

def load_model_ckpt():
    if not thread_data.ckpt_file: raise ValueError(f'Thread ckpt_file is undefined.')
    if os.path.exists(thread_data.ckpt_file + '.ckpt'):
        thread_data.ckpt_file += '.ckpt'
    elif os.path.exists(thread_data.ckpt_file + '.safetensors'):
        thread_data.ckpt_file += '.safetensors'
    elif not os.path.exists(thread_data.ckpt_file):
        raise FileNotFoundError(f'Cannot find {thread_data.ckpt_file}.ckpt or .safetensors')

    if not thread_data.precision:
        thread_data.precision = 'full' if thread_data.force_full_precision else 'autocast'

    if not thread_data.unet_bs:
        thread_data.unet_bs = 1

    if thread_data.device == 'cpu':
        thread_data.precision = 'full'

    print('loading', thread_data.ckpt_file, 'to device', thread_data.device, 'using precision', thread_data.precision)

    if thread_data.test_sd2:
        load_model_ckpt_sd2()
    else:
        load_model_ckpt_sd1()

def load_model_ckpt_sd1():
    sd, model_ver = load_model_from_config(thread_data.ckpt_file)
    li, lo = [], []
    for key, value in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    config = OmegaConf.load(f"{config_yaml}")

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()
    model.cdevice = torch.device(thread_data.device)
    model.unet_bs = thread_data.unet_bs
    model.turbo = thread_data.turbo
    # if thread_data.device != 'cpu':
    #     model.to(thread_data.device)
    #if thread_data.reduced_memory:
        #model.model1.to("cpu")
        #model.model2.to("cpu")
    thread_data.model = model

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()
    modelCS.cond_stage_model.device = torch.device(thread_data.device)
    # if thread_data.device != 'cpu':
    #     if thread_data.reduced_memory:
    #         modelCS.to('cpu')
    #     else:
    #         modelCS.to(thread_data.device) # Preload on device if not already there.
    thread_data.modelCS = modelCS

    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)

    if thread_data.vae_file is not None:
        try:
            loaded = False
            for model_extension in ['.ckpt', '.vae.pt']:
                if os.path.exists(thread_data.vae_file + model_extension):
                    print(f"Loading VAE weights from: {thread_data.vae_file}{model_extension}")
                    vae_ckpt = torch.load(thread_data.vae_file + model_extension, map_location="cpu")
                    vae_dict = {k: v for k, v in vae_ckpt["state_dict"].items() if k[0:4] != "loss"}
                    modelFS.first_stage_model.load_state_dict(vae_dict, strict=False)
                    loaded = True
                    break

            if not loaded:
                print(f'Cannot find VAE: {thread_data.vae_file}')
                thread_data.vae_file = None
        except:
            print(traceback.format_exc())
            print(f'Could not load VAE: {thread_data.vae_file}')
            thread_data.vae_file = None

    modelFS.eval()
    # if thread_data.device != 'cpu':
    #     if thread_data.reduced_memory:
    #         modelFS.to('cpu')
    #     else:
    #         modelFS.to(thread_data.device) # Preload on device if not already there.
    thread_data.modelFS = modelFS
    del sd

    if thread_data.device != "cpu" and thread_data.precision == "autocast":
        thread_data.model.half()
        thread_data.modelCS.half()
        thread_data.modelFS.half()
        thread_data.model_is_half = True
        thread_data.model_fs_is_half = True
    else:
        thread_data.model_is_half = False
        thread_data.model_fs_is_half = False

    print(f'''loaded model
 model file: {thread_data.ckpt_file}
 model.device: {model.device}
 modelCS.device: {modelCS.cond_stage_model.device}
 modelFS.device: {thread_data.modelFS.device}
 using precision: {thread_data.precision}''')

def load_model_ckpt_sd2():
    sd, model_ver = load_model_from_config(thread_data.ckpt_file)

    config_file = 'configs/stable-diffusion/v2-inference-v.yaml' if model_ver == 'sd2' else "configs/stable-diffusion/v1-inference.yaml"
    config = OmegaConf.load(config_file)
    verbose = False

    thread_data.model = instantiate_from_config(config.model)
    m, u = thread_data.model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    thread_data.model.to(thread_data.device)
    thread_data.model.eval()
    del sd

    thread_data.model.cond_stage_model.device = torch.device(thread_data.device)

    if thread_data.device != "cpu" and thread_data.precision == "autocast":
        thread_data.model.half()
        thread_data.model_is_half = True
        thread_data.model_fs_is_half = True
    else:
        thread_data.model_is_half = False
        thread_data.model_fs_is_half = False

    print(f'''loaded model
 model file: {thread_data.ckpt_file}
 using precision: {thread_data.precision}''')

def unload_filters():
    if thread_data.model_gfpgan is not None:
        if thread_data.device != 'cpu': thread_data.model_gfpgan.gfpgan.to('cpu')

        del thread_data.model_gfpgan
    thread_data.model_gfpgan = None

    if thread_data.model_real_esrgan is not None:
        if thread_data.device != 'cpu': thread_data.model_real_esrgan.model.to('cpu')

        del thread_data.model_real_esrgan
    thread_data.model_real_esrgan = None

    gc()

def unload_models():
    if thread_data.model is not None:
        print('Unloading models...')
        if thread_data.device != 'cpu':
            if not thread_data.test_sd2:
                thread_data.modelFS.to('cpu')
                thread_data.modelCS.to('cpu')
                thread_data.model.model1.to("cpu")
                thread_data.model.model2.to("cpu")

        del thread_data.model
        del thread_data.modelCS
        del thread_data.modelFS

    thread_data.model = None
    thread_data.modelCS = None
    thread_data.modelFS = None

    gc()

# def wait_model_move_to(model, target_device): # Send to target_device and wait until complete.
#     if thread_data.device == target_device: return
#     start_mem = torch.cuda.memory_allocated(thread_data.device) / 1e6
#     if start_mem <= 0: return
#     model_name = model.__class__.__name__
#     print(f'Device {thread_data.device} - Sending model {model_name} to {target_device} | Memory transfer starting. Memory Used: {round(start_mem)}Mb')
#     start_time = time.time()
#     model.to(target_device)
#     time_step = start_time
#     WARNING_TIMEOUT = 1.5 # seconds - Show activity in console after timeout.
#     last_mem = start_mem
#     is_transfering = True
#     while is_transfering:
#         time.sleep(0.5) # 500ms
#         mem = torch.cuda.memory_allocated(thread_data.device) / 1e6
#         is_transfering = bool(mem > 0 and mem < last_mem) # still stuff loaded, but less than last time.
#         last_mem = mem
#         if not is_transfering:
#             break;
#         if time.time() - time_step > WARNING_TIMEOUT: # Long delay, print to console to show activity.
#             print(f'Device {thread_data.device} - Waiting for Memory transfer. Memory Used: {round(mem)}Mb, Transfered: {round(start_mem - mem)}Mb')
#             time_step = time.time()
#     print(f'Device {thread_data.device} - {model_name} Moved: {round(start_mem - last_mem)}Mb in {round(time.time() - start_time, 3)} seconds to {target_device}')

def move_to_cpu(model):
    if thread_data.device != "cpu":
        d = torch.device(thread_data.device)
        mem = torch.cuda.memory_allocated(d) / 1e6
        model.to("cpu")
        while torch.cuda.memory_allocated(d) / 1e6 >= mem:
            time.sleep(1)

def load_model_gfpgan():
    if thread_data.gfpgan_file is None: raise ValueError(f'Thread gfpgan_file is undefined.')
    model_path = thread_data.gfpgan_file + ".pth"
    thread_data.model_gfpgan = GFPGANer(device=torch.device(thread_data.device), model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
    print('loaded', thread_data.gfpgan_file, 'to', thread_data.model_gfpgan.device, 'precision', thread_data.precision)

def load_model_real_esrgan():
    if thread_data.real_esrgan_file is None: raise ValueError(f'Thread real_esrgan_file is undefined.')
    model_path = thread_data.real_esrgan_file + ".pth"

    RealESRGAN_models = {
        'RealESRGAN_x4plus': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        'RealESRGAN_x4plus_anime_6B': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    }

    model_to_use = RealESRGAN_models[thread_data.real_esrgan_file]

    if thread_data.device == 'cpu':
        thread_data.model_real_esrgan = RealESRGANer(device=torch.device(thread_data.device), scale=2, model_path=model_path, model=model_to_use, pre_pad=0, half=False) # cpu does not support half
        #thread_data.model_real_esrgan.device = torch.device(thread_data.device)
        thread_data.model_real_esrgan.model.to('cpu')
    else:
        thread_data.model_real_esrgan = RealESRGANer(device=torch.device(thread_data.device), scale=2, model_path=model_path, model=model_to_use, pre_pad=0, half=thread_data.model_is_half)

    thread_data.model_real_esrgan.model.name = thread_data.real_esrgan_file
    print('loaded ', thread_data.real_esrgan_file, 'to', thread_data.model_real_esrgan.device, 'precision', thread_data.precision)


def get_session_out_path(disk_path, session_id):
    if disk_path is None: return None
    if session_id is None: return None

    session_out_path = os.path.join(disk_path, filename_regex.sub('_',session_id))
    os.makedirs(session_out_path, exist_ok=True)
    return session_out_path

def get_base_path(disk_path, session_id, prompt, img_id, ext, suffix=None):
    if disk_path is None: return None
    if session_id is None: return None
    if ext is None: raise Exception('Missing ext')

    session_out_path = get_session_out_path(disk_path, session_id)

    prompt_flattened = filename_regex.sub('_', prompt)[:50]

    if suffix is not None:
        return os.path.join(session_out_path, f"{prompt_flattened}_{img_id}_{suffix}.{ext}")
    return os.path.join(session_out_path, f"{prompt_flattened}_{img_id}.{ext}")

def apply_filters(filter_name, image_data, model_path=None):
    print(f'Applying filter {filter_name}...')
    gc() # Free space before loading new data.

    if isinstance(image_data, torch.Tensor):
        image_data.to(thread_data.device)

    if filter_name == 'gfpgan':
        # This lock is only ever used here. No need to use timeout for the request. Should never deadlock.
        with gfpgan_temp_device_lock: # Wait for any other devices to complete before starting.
            # hack for a bug in facexlib: https://github.com/xinntao/facexlib/pull/19/files
            from facexlib.detection import retinaface
            retinaface.device = torch.device(thread_data.device)
            print('forced retinaface.device to', thread_data.device)

            if model_path is not None and model_path != thread_data.gfpgan_file:
                thread_data.gfpgan_file = model_path
                load_model_gfpgan()
            elif not thread_data.model_gfpgan:
                load_model_gfpgan()
            if thread_data.model_gfpgan is None: raise Exception('Model "gfpgan" not loaded.')

            print('enhance with', thread_data.gfpgan_file, 'on', thread_data.model_gfpgan.device, 'precision', thread_data.precision)
            _, _, output = thread_data.model_gfpgan.enhance(image_data[:,:,::-1], has_aligned=False, only_center_face=False, paste_back=True)
            image_data = output[:,:,::-1]

    if filter_name == 'real_esrgan':
        if model_path is not None and model_path != thread_data.real_esrgan_file:
            thread_data.real_esrgan_file = model_path
            load_model_real_esrgan()
        elif not thread_data.model_real_esrgan:
            load_model_real_esrgan()
        if thread_data.model_real_esrgan is None: raise Exception('Model "gfpgan" not loaded.')
        print('enhance with', thread_data.real_esrgan_file, 'on', thread_data.model_real_esrgan.device, 'precision', thread_data.precision)
        output, _ = thread_data.model_real_esrgan.enhance(image_data[:,:,::-1])
        image_data = output[:,:,::-1]

    return image_data

def is_model_reload_necessary(req: Request):
    # custom model support:
    #  the req.use_stable_diffusion_model needs to be a valid path
    #  to the ckpt file (without the extension).
    if os.path.exists(req.use_stable_diffusion_model + '.ckpt'):
        req.use_stable_diffusion_model += '.ckpt'
    elif os.path.exists(req.use_stable_diffusion_model + '.safetensors'):
        req.use_stable_diffusion_model += '.safetensors'
    elif not os.path.exists(req.use_stable_diffusion_model): 
        raise FileNotFoundError(f'Cannot find {req.use_stable_diffusion_model}.ckpt or .safetensors')

    needs_model_reload = False
    if not thread_data.model or thread_data.ckpt_file != req.use_stable_diffusion_model or thread_data.vae_file != req.use_vae_model:
        thread_data.ckpt_file = req.use_stable_diffusion_model
        thread_data.vae_file = req.use_vae_model
        needs_model_reload = True

    if thread_data.device != 'cpu':
        if (thread_data.precision == 'autocast' and (req.use_full_precision or not thread_data.model_is_half)) or \
            (thread_data.precision == 'full' and not req.use_full_precision and not thread_data.force_full_precision):
            thread_data.precision = 'full' if req.use_full_precision else 'autocast'
            needs_model_reload = True

    return needs_model_reload

def reload_model():
    unload_models()
    unload_filters()
    load_model_ckpt()

def is_hypernetwork_reload_necessary(req: Request):
    needs_model_reload = False
    if thread_data.hypernetwork_file != req.use_hypernetwork_model:
        thread_data.hypernetwork_file = req.use_hypernetwork_model
        needs_model_reload = True

    return needs_model_reload

def load_hypernetwork():
    if thread_data.test_sd2:
        # Not yet supported in SD2
        return

    from . import hypernetwork
    if thread_data.hypernetwork_file is not None:
        try:
            loaded = False
            for model_extension in HYPERNETWORK_MODEL_EXTENSIONS:
                if os.path.exists(thread_data.hypernetwork_file + model_extension):
                    print(f"Loading hypernetwork weights from: {thread_data.hypernetwork_file}{model_extension}")
                    thread_data.hypernetwork = hypernetwork.load_hypernetwork(thread_data.hypernetwork_file + model_extension)
                    loaded = True
                    break

            if not loaded:
                print(f'Cannot find hypernetwork: {thread_data.hypernetwork_file}')
                thread_data.hypernetwork_file = None
        except:
            print(traceback.format_exc())
            print(f'Could not load hypernetwork: {thread_data.hypernetwork_file}')
            thread_data.hypernetwork_file = None

def unload_hypernetwork():
    if thread_data.hypernetwork is not None:
        print('Unloading hypernetwork...')
        if thread_data.device != 'cpu':
            for i in thread_data.hypernetwork:
                thread_data.hypernetwork[i][0].to('cpu')
                thread_data.hypernetwork[i][1].to('cpu')
        del thread_data.hypernetwork
    thread_data.hypernetwork = None

    gc()

def reload_hypernetwork():
    unload_hypernetwork()
    load_hypernetwork()

def mk_img(req: Request, data_queue: queue.Queue, task_temp_images: list, step_callback):
    try:
        return do_mk_img(req, data_queue, task_temp_images, step_callback)
    except Exception as e:
        print(traceback.format_exc())

        if thread_data.device != 'cpu' and not thread_data.test_sd2:
            thread_data.modelFS.to('cpu')
            thread_data.modelCS.to('cpu')
            thread_data.model.model1.to("cpu")
            thread_data.model.model2.to("cpu")

        gc() # Release from memory.
        data_queue.put(json.dumps({
            "status": 'failed',
            "detail": str(e)
        }))
        raise e

def update_temp_img(req, x_samples, task_temp_images: list):
    partial_images = []
    for i in range(req.num_outputs):
        if thread_data.test_sd2:
            x_sample_ddim = thread_data.model.decode_first_stage(x_samples[i].unsqueeze(0))
        else:
            x_sample_ddim = thread_data.modelFS.decode_first_stage(x_samples[i].unsqueeze(0))
        x_sample = torch.clamp((x_sample_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
        x_sample = x_sample.astype(np.uint8)
        img = Image.fromarray(x_sample)
        buf = img_to_buffer(img, output_format='JPEG')

        del img, x_sample, x_sample_ddim
        # don't delete x_samples, it is used in the code that called this callback

        thread_data.temp_images[f'{req.request_id}/{i}'] = buf
        task_temp_images[i] = buf
        partial_images.append({'path': f'/image/tmp/{req.request_id}/{i}'})
    return partial_images

# Build and return the apropriate generator for do_mk_img
def get_image_progress_generator(req, data_queue: queue.Queue, task_temp_images: list, step_callback, extra_props=None):
    if not req.stream_progress_updates:
        def empty_callback(x_samples, i):
            step_callback()
        return empty_callback

    thread_data.partial_x_samples = None
    last_callback_time = -1
    def img_callback(x_samples, i):
        nonlocal last_callback_time

        thread_data.partial_x_samples = x_samples
        step_time = time.time() - last_callback_time if last_callback_time != -1 else -1
        last_callback_time = time.time()

        progress = {"step": i, "step_time": step_time}
        if extra_props is not None:
            progress.update(extra_props)

        if req.stream_image_progress and i % 5 == 0:
            progress['output'] = update_temp_img(req, x_samples, task_temp_images)

        data_queue.put(json.dumps(progress))

        step_callback()

        if thread_data.stop_processing:
            raise UserInitiatedStop("User requested that we stop processing")
    return img_callback

def do_mk_img(req: Request, data_queue: queue.Queue, task_temp_images: list, step_callback):
    thread_data.stop_processing = False

    res = Response()
    res.request = req
    res.images = []
    thread_data.hypernetwork_strength = req.hypernetwork_strength

    thread_data.temp_images.clear()

    if thread_data.turbo != req.turbo and not thread_data.test_sd2:
        thread_data.turbo = req.turbo
        thread_data.model.turbo = req.turbo

    # Start by cleaning memory, loading and unloading things can leave memory allocated.
    gc()

    opt_prompt = req.prompt
    opt_seed = req.seed
    opt_n_iter = 1
    opt_C = 4
    opt_f = 8
    opt_ddim_eta = 0.0

    print(req, '\n    device', torch.device(thread_data.device), "as", thread_data.device_name)
    print('\n\n    Using precision:', thread_data.precision)

    seed_everything(opt_seed)

    batch_size = req.num_outputs
    prompt = opt_prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

    if thread_data.precision == "autocast" and thread_data.device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    mask = None

    if req.init_image is None:
        handler = _txt2img

        init_latent = None
        t_enc = None
    else:
        handler = _img2img

        init_image = load_img(req.init_image, req.width, req.height)
        init_image = init_image.to(thread_data.device)

        if thread_data.device != "cpu" and thread_data.precision == "autocast":
            init_image = init_image.half()

        if not thread_data.test_sd2:
            thread_data.modelFS.to(thread_data.device)

        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        if thread_data.test_sd2:
            init_latent = thread_data.model.get_first_stage_encoding(thread_data.model.encode_first_stage(init_image))  # move to latent space
        else:
            init_latent = thread_data.modelFS.get_first_stage_encoding(thread_data.modelFS.encode_first_stage(init_image))  # move to latent space

        if req.mask is not None:
            mask = load_mask(req.mask, req.width, req.height, init_latent.shape[2], init_latent.shape[3], True).to(thread_data.device)
            mask = mask[0][0].unsqueeze(0).repeat(4, 1, 1).unsqueeze(0)
            mask = repeat(mask, '1 ... -> b ...', b=batch_size)

            if thread_data.device != "cpu" and thread_data.precision == "autocast":
                mask = mask.half()

        # Send to CPU and wait until complete.
        # wait_model_move_to(thread_data.modelFS, 'cpu')
        if not thread_data.test_sd2:
            move_to_cpu(thread_data.modelFS)

        assert 0. <= req.prompt_strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(req.prompt_strength * req.num_inference_steps)
        print(f"target t_enc is {t_enc} steps")

    with torch.no_grad():
        for n in trange(opt_n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):

                with precision_scope("cuda"):
                    if thread_data.reduced_memory and not thread_data.test_sd2:
                        thread_data.modelCS.to(thread_data.device)
                    uc = None
                    if req.guidance_scale != 1.0:
                        if thread_data.test_sd2:
                            uc = thread_data.model.get_learned_conditioning(batch_size * [req.negative_prompt])
                        else:
                            uc = thread_data.modelCS.get_learned_conditioning(batch_size * [req.negative_prompt])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        c = torch.zeros_like(uc)
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(len(subprompts)):
                            weight = weights[i]
                            # if not skip_normalize:
                            weight = weight / totalWeight
                            if thread_data.test_sd2:
                                c = torch.add(c, thread_data.model.get_learned_conditioning(subprompts[i]), alpha=weight)
                            else:
                                c = torch.add(c, thread_data.modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                    else:
                        if thread_data.test_sd2:
                            c = thread_data.model.get_learned_conditioning(prompts)
                        else:
                            c = thread_data.modelCS.get_learned_conditioning(prompts)

                    if thread_data.reduced_memory and not thread_data.test_sd2:
                        thread_data.modelFS.to(thread_data.device)

                    n_steps = req.num_inference_steps if req.init_image is None else t_enc
                    img_callback = get_image_progress_generator(req, data_queue, task_temp_images, step_callback, {"total_steps": n_steps})

                    # run the handler
                    try:
                        print('Running handler...')
                        if handler == _txt2img:
                            x_samples = _txt2img(req.width, req.height, req.num_outputs, req.num_inference_steps, req.guidance_scale, None, opt_C, opt_f, opt_ddim_eta, c, uc, opt_seed, img_callback, mask, req.sampler)
                        else:
                            x_samples = _img2img(init_latent, t_enc, batch_size, req.guidance_scale, c, uc, req.num_inference_steps, opt_ddim_eta, opt_seed, img_callback, mask, opt_C, req.height, req.width, opt_f)
                    except UserInitiatedStop:
                        if not hasattr(thread_data, 'partial_x_samples'):
                            continue
                        if thread_data.partial_x_samples is None:
                            del thread_data.partial_x_samples
                            continue
                        x_samples = thread_data.partial_x_samples
                        del thread_data.partial_x_samples

                    print("decoding images")
                    img_data = [None] * batch_size
                    for i in range(batch_size):
                        if thread_data.test_sd2:
                            x_samples_ddim = thread_data.model.decode_first_stage(x_samples[i].unsqueeze(0))
                        else:
                            x_samples_ddim = thread_data.modelFS.decode_first_stage(x_samples[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                        x_sample = x_sample.astype(np.uint8)
                        img_data[i] = x_sample
                    del x_samples, x_samples_ddim, x_sample

                    print("saving images")
                    for i in range(batch_size):
                        img = Image.fromarray(img_data[i])
                        img_id = base64.b64encode(int(time.time()+i).to_bytes(8, 'big')).decode() # Generate unique ID based on time.
                        img_id = img_id.translate({43:None, 47:None, 61:None})[-8:] # Remove + / = and keep last 8 chars.

                        has_filters =   (req.use_face_correction is not None and req.use_face_correction.startswith('GFPGAN')) or \
                                        (req.use_upscale is not None and req.use_upscale.startswith('RealESRGAN'))

                        return_orig_img = not has_filters or not req.show_only_filtered_image

                        if thread_data.stop_processing:
                            return_orig_img = True

                        if req.save_to_disk_path is not None:
                            if return_orig_img:
                                img_out_path = get_base_path(req.save_to_disk_path, req.session_id, prompts[0], img_id, req.output_format)
                                save_image(img, img_out_path, req.output_format, req.output_quality)
                            meta_out_path = get_base_path(req.save_to_disk_path, req.session_id, prompts[0], img_id, 'txt')
                            save_metadata(meta_out_path, req, prompts[0], opt_seed)

                        if return_orig_img:
                            img_buffer = img_to_buffer(img, req.output_format, req.output_quality)
                            img_str = buffer_to_base64_str(img_buffer, req.output_format)
                            res_image_orig = ResponseImage(data=img_str, seed=opt_seed)
                            res.images.append(res_image_orig)
                            task_temp_images[i] = img_buffer

                            if req.save_to_disk_path is not None:
                                res_image_orig.path_abs = img_out_path
                        del img

                        if has_filters and not thread_data.stop_processing:
                            filters_applied = []
                            if req.use_face_correction:
                                img_data[i] = apply_filters('gfpgan', img_data[i], req.use_face_correction)
                                filters_applied.append(req.use_face_correction)
                            if req.use_upscale:
                                img_data[i] = apply_filters('real_esrgan', img_data[i], req.use_upscale)
                                filters_applied.append(req.use_upscale)
                            if (len(filters_applied) > 0):
                                filtered_image = Image.fromarray(img_data[i])
                                filtered_buffer = img_to_buffer(filtered_image, req.output_format, req.output_quality)
                                filtered_img_data = buffer_to_base64_str(filtered_buffer, req.output_format)
                                response_image = ResponseImage(data=filtered_img_data, seed=opt_seed)
                                res.images.append(response_image)
                                task_temp_images[i] = filtered_buffer
                                if req.save_to_disk_path is not None:
                                    filtered_img_out_path = get_base_path(req.save_to_disk_path, req.session_id, prompts[0], img_id, req.output_format, "_".join(filters_applied))
                                    save_image(filtered_image, filtered_img_out_path, req.output_format, req.output_quality)
                                    response_image.path_abs = filtered_img_out_path
                                del filtered_image
                        # Filter Applied, move to next seed
                        opt_seed += 1

                    # if thread_data.reduced_memory:
                    #     unload_filters()
                    if not thread_data.test_sd2:
                        move_to_cpu(thread_data.modelFS)
                    del img_data
                    gc()
                    if thread_data.device != 'cpu':
                        print(f'memory_final = {round(torch.cuda.memory_allocated(thread_data.device) / 1e6, 2)}Mb')

    print('Task completed')
    res = res.json()
    data_queue.put(json.dumps(res))

    return res

def save_image(img, img_out_path, output_format="", output_quality=75):
    try:
        if output_format.upper() == "JPEG":
            img.save(img_out_path, quality=output_quality)
        else:
            img.save(img_out_path)
    except:
        print('could not save the file', traceback.format_exc())

def save_metadata(meta_out_path, req, prompt, opt_seed):
    metadata = f'''{prompt}
Width: {req.width}
Height: {req.height}
Seed: {opt_seed}
Steps: {req.num_inference_steps}
Guidance Scale: {req.guidance_scale}
Prompt Strength: {req.prompt_strength}
Use Face Correction: {req.use_face_correction}
Use Upscaling: {req.use_upscale}
Sampler: {req.sampler}
Negative Prompt: {req.negative_prompt}
Stable Diffusion model: {req.use_stable_diffusion_model + '.ckpt'}
VAE model: {req.use_vae_model}
Hypernetwork Model: {req.use_hypernetwork_model}
Hypernetwork Strength: {req.hypernetwork_strength}
'''
    try:
        with open(meta_out_path, 'w', encoding='utf-8') as f:
            f.write(metadata)
    except:
        print('could not save the file', traceback.format_exc())

def _txt2img(opt_W, opt_H, opt_n_samples, opt_ddim_steps, opt_scale, start_code, opt_C, opt_f, opt_ddim_eta, c, uc, opt_seed, img_callback, mask, sampler_name):
    shape = [opt_n_samples, opt_C, opt_H // opt_f, opt_W // opt_f]

    # Send to CPU and wait until complete.
    # wait_model_move_to(thread_data.modelCS, 'cpu')

    if not thread_data.test_sd2:
        move_to_cpu(thread_data.modelCS)

    if thread_data.test_sd2 and sampler_name not in ('plms', 'ddim', 'dpm2'):
        raise Exception('Only plms and ddim samplers are supported right now, in SD 2.0')


    # samples, _ = sampler.sample(S=opt.steps,
    #                                                  conditioning=c,
    #                                                  batch_size=opt.n_samples,
    #                                                  shape=shape,
    #                                                  verbose=False,
    #                                                  unconditional_guidance_scale=opt.scale,
    #                                                  unconditional_conditioning=uc,
    #                                                  eta=opt.ddim_eta,
    #                                                  x_T=start_code)

    if thread_data.test_sd2:
        if sampler_name == 'plms':
            from ldm.models.diffusion.plms import PLMSSampler
            sampler = PLMSSampler(thread_data.model)
        elif sampler_name == 'ddim':
            from ldm.models.diffusion.ddim import DDIMSampler
            sampler = DDIMSampler(thread_data.model)
            sampler.make_schedule(ddim_num_steps=opt_ddim_steps, ddim_eta=opt_ddim_eta, verbose=False)
        elif sampler_name == 'dpm2':
            from ldm.models.diffusion.dpm_solver import DPMSolverSampler
            sampler = DPMSolverSampler(thread_data.model)

        shape = [opt_C, opt_H // opt_f, opt_W // opt_f]

        samples_ddim, intermediates = sampler.sample(
            S=opt_ddim_steps,
            conditioning=c,
            batch_size=opt_n_samples,
            seed=opt_seed,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=opt_scale,
            unconditional_conditioning=uc,
            eta=opt_ddim_eta,
            x_T=start_code,
            img_callback=img_callback,
            mask=mask,
            sampler = sampler_name,
        )
    else:
        if sampler_name == 'ddim':
            thread_data.model.make_schedule(ddim_num_steps=opt_ddim_steps, ddim_eta=opt_ddim_eta, verbose=False)

        samples_ddim = thread_data.model.sample(
            S=opt_ddim_steps,
            conditioning=c,
            seed=opt_seed,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=opt_scale,
            unconditional_conditioning=uc,
            eta=opt_ddim_eta,
            x_T=start_code,
            img_callback=img_callback,
            mask=mask,
            sampler = sampler_name,
        )
    return samples_ddim

def _img2img(init_latent, t_enc, batch_size, opt_scale, c, uc, opt_ddim_steps, opt_ddim_eta, opt_seed, img_callback, mask, opt_C=1, opt_H=1, opt_W=1, opt_f=1):
    # encode (scaled latent)
    x_T = None if mask is None else init_latent

    if thread_data.test_sd2:
        from ldm.models.diffusion.ddim import DDIMSampler

        sampler = DDIMSampler(thread_data.model)

        sampler.make_schedule(ddim_num_steps=opt_ddim_steps, ddim_eta=opt_ddim_eta, verbose=False)

        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(thread_data.device))

        samples_ddim = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt_scale,unconditional_conditioning=uc, img_callback=img_callback)

    else:
        z_enc = thread_data.model.stochastic_encode(
            init_latent,
            torch.tensor([t_enc] * batch_size).to(thread_data.device),
            opt_seed,
            opt_ddim_eta,
            opt_ddim_steps,
        )

        # decode it
        samples_ddim = thread_data.model.sample(
            t_enc,
            c,
            z_enc,
            unconditional_guidance_scale=opt_scale,
            unconditional_conditioning=uc,
            img_callback=img_callback,
            mask=mask,
            x_T=x_T,
            sampler = 'ddim'
        )
    return samples_ddim

def gc():
    gc_collect()
    if thread_data.device == 'cpu':
        return
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# internal

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    model_ver = 'sd1'

    if ckpt.endswith(".safetensors"):
        print("Loading from safetensors")
        pl_sd = load_file(ckpt, device="cpu")
    else:
        pl_sd = torch.load(ckpt, map_location="cpu")

    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    if "state_dict" in pl_sd:
        # check for a key that only seems to be present in SD2 models
        if 'cond_stage_model.model.ln_final.bias' in pl_sd['state_dict'].keys():
            model_ver = 'sd2'

        return pl_sd["state_dict"], model_ver
    else:
        return pl_sd, model_ver

class UserInitiatedStop(Exception):
    pass

def load_img(img_str, w0, h0):
    image = base64_str_to_img(img_str).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from base64")
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def load_mask(mask_str, h0, w0, newH, newW, invert=False):
    image = base64_str_to_img(mask_str).convert("RGB")
    w, h = image.size
    print(f"loaded input mask of size ({w}, {h})")

    if invert:
        print("inverted")
        image = ImageOps.invert(image)
        # where_0, where_1 = np.where(image == 0), np.where(image == 255)
        # image[where_0], image[where_1] = 255, 0

    if h0 is not None and w0 is not None:
        h, w = h0, w0

    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64

    print(f"New mask size ({w}, {h})")
    image = image.resize((newW, newH), resample=Image.Resampling.LANCZOS)
    image = np.array(image)

    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

# https://stackoverflow.com/a/61114178
def img_to_base64_str(img, output_format="PNG", output_quality=75):
    buffered = img_to_buffer(img, output_format, quality=output_quality)
    return buffer_to_base64_str(buffered, output_format)

def img_to_buffer(img, output_format="PNG", output_quality=75):
    buffered = BytesIO()
    if ( output_format.upper() == "JPEG" ):
        img.save(buffered, format=output_format, quality=output_quality)
    else:
        img.save(buffered, format=output_format)
    buffered.seek(0)
    return buffered

def buffer_to_base64_str(buffered, output_format="PNG"):
    buffered.seek(0)
    img_byte = buffered.getvalue()
    mime_type = "image/png" if output_format.lower() == "png" else "image/jpeg"
    img_str = f"data:{mime_type};base64," + base64.b64encode(img_byte).decode()
    return img_str

def base64_str_to_buffer(img_str):
    mime_type = "image/png" if img_str.startswith("data:image/png;") else "image/jpeg"
    img_str = img_str[len(f"data:{mime_type};base64,"):]
    data = base64.b64decode(img_str)
    buffered = BytesIO(data)
    return buffered

def base64_str_to_img(img_str):
    buffered = base64_str_to_buffer(img_str)
    img = Image.open(buffered)
    return img

def split_weighted_subprompts(text):
    """
    grabs all text up to the first occurrence of ':' 
    uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
    if ':' has no value defined, defaults to 1.0
    repeats until no text remaining
    """
    remaining = len(text)
    prompts = []
    weights = []
    while remaining > 0:
        if ":" in text:
            idx = text.index(":") # first occurrence from start
            # grab up to index as sub-prompt
            prompt = text[:idx]
            remaining -= idx
            # remove from main text
            text = text[idx+1:]
            # find value for weight 
            if " " in text:
                idx = text.index(" ") # first occurence
            else: # no space, read to end
                idx = len(text)
            if idx != 0:
                try:
                    weight = float(text[:idx])
                except: # couldn't treat as float
                    print(f"Warning: '{text[:idx]}' is not a value, are you missing a space?")
                    weight = 1.0
            else: # no value found
                weight = 1.0
            # remove from main text
            remaining -= idx
            text = text[idx+1:]
            # append the sub-prompt and its weight
            prompts.append(prompt)
            weights.append(weight)
        else: # no : found
            if len(text) > 0: # there is still text though
                # take remainder as weight 1
                prompts.append(text)
                weights.append(1.0)
            remaining = 0
    return prompts, weights
