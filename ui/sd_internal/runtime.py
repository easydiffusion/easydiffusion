import json
import os, re
import traceback
import torch
import numpy as np
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
from optimizedSD.optimUtils import split_weighted_subprompts
from transformers import logging

from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

import uuid

logging.set_verbosity_error()

# consts
config_yaml = "optimizedSD/v1-inference.yaml"
filename_regex = re.compile('[^a-zA-Z0-9]')

# api stuff
from . import Request, Response, Image as ResponseImage
import base64
from io import BytesIO
#from colorama import Fore

# local
stop_processing = False
temp_images = {}

ckpt_file = None
gfpgan_file = None
real_esrgan_file = None

model = None
modelCS = None
modelFS = None
model_gfpgan = None
model_real_esrgan = None

model_is_half = False
model_fs_is_half = False
device = None
unet_bs = 1
precision = 'autocast'
sampler_plms = None
sampler_ddim = None

has_valid_gpu = False
force_full_precision = False
try:
    gpu = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu)
    print('GPU detected: ', gpu_name)

    force_full_precision = ('nvidia' in gpu_name.lower() or 'geforce' in gpu_name.lower()) and (' 1660' in gpu_name or ' 1650' in gpu_name) # otherwise these NVIDIA cards create green images
    if force_full_precision:
        print('forcing full precision on NVIDIA 16xx cards, to avoid green images. GPU detected: ', gpu_name)

    mem_free, mem_total = torch.cuda.mem_get_info(gpu)
    mem_total /= float(10**9)
    if mem_total < 3.0:
        print("GPUs with less than 3 GB of VRAM are not compatible with Stable Diffusion")
        raise Exception()

    has_valid_gpu = True
except:
    print('WARNING: No compatible GPU found. Using the CPU, but this will be very slow!')
    pass

def load_model_ckpt(ckpt_to_use, device_to_use='cuda', turbo=False, unet_bs_to_use=1, precision_to_use='autocast'):
    global ckpt_file, model, modelCS, modelFS, model_is_half, device, unet_bs, precision, model_fs_is_half

    device = device_to_use if has_valid_gpu else 'cpu'
    precision = precision_to_use if not force_full_precision else 'full'
    unet_bs = unet_bs_to_use

    unload_model()

    if device == 'cpu':
        precision = 'full'

    sd = load_model_from_config(f"{ckpt_to_use}.ckpt")
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
    model.cdevice = device
    model.unet_bs = unet_bs
    model.turbo = turbo

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()
    modelCS.cond_stage_model.device = device

    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()
    del sd

    if device != "cpu" and precision == "autocast":
        model.half()
        modelCS.half()
        modelFS.half()
        model_is_half = True
        model_fs_is_half = True
    else:
        model_is_half = False
        model_fs_is_half = False

    ckpt_file = ckpt_to_use

    print('loaded ', ckpt_file, 'to', device, 'precision', precision)

def unload_model():
    global model, modelCS, modelFS

    if model is not None:
        del model
        del modelCS
        del modelFS

    model = None
    modelCS = None
    modelFS = None

def load_model_gfpgan(gfpgan_to_use):
    global gfpgan_file, model_gfpgan

    if gfpgan_to_use is None:
        return

    gfpgan_file = gfpgan_to_use
    model_path = gfpgan_to_use + ".pth"

    if device == 'cpu':
        model_gfpgan = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None, device=torch.device('cpu'))
    else:
        model_gfpgan = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None, device=torch.device('cuda'))

    print('loaded ', gfpgan_to_use, 'to', device, 'precision', precision)

def load_model_real_esrgan(real_esrgan_to_use):
    global real_esrgan_file, model_real_esrgan

    if real_esrgan_to_use is None:
        return

    real_esrgan_file = real_esrgan_to_use
    model_path = real_esrgan_to_use + ".pth"

    RealESRGAN_models = {
        'RealESRGAN_x4plus': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        'RealESRGAN_x4plus_anime_6B': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    }

    model_to_use = RealESRGAN_models[real_esrgan_to_use]

    if device == 'cpu':
        model_real_esrgan = RealESRGANer(scale=2, model_path=model_path, model=model_to_use, pre_pad=0, half=False) # cpu does not support half
        model_real_esrgan.device = torch.device('cpu')
        model_real_esrgan.model.to('cpu')
    else:
        model_real_esrgan = RealESRGANer(scale=2, model_path=model_path, model=model_to_use, pre_pad=0, half=model_is_half)

    model_real_esrgan.model.name = real_esrgan_to_use

    print('loaded ', real_esrgan_to_use, 'to', device, 'precision', precision)

def get_base_path(disk_path, session_id, prompt, img_id, ext, suffix=None):
    if disk_path is None: return None
    if session_id is None: return None
    if ext is None: raise Exception('Missing ext')

    session_out_path = os.path.join(disk_path, session_id)
    os.makedirs(session_out_path, exist_ok=True)

    prompt_flattened = filename_regex.sub('_', prompt)[:50]
    

    if suffix is not None:
        return os.path.join(session_out_path, f"{prompt_flattened}_{img_id}_{suffix}.{ext}")
    return os.path.join(session_out_path, f"{prompt_flattened}_{img_id}.{ext}")

def apply_filters(filter_name, image_data):
    print(f'Applying filter {filter_name}...')
    gc()

    if filter_name == 'gfpgan':
        _, _, output = model_gfpgan.enhance(image_data[:,:,::-1], has_aligned=False, only_center_face=False, paste_back=True)
        image_data = output[:,:,::-1]

    if filter_name == 'real_esrgan':
        output, _ = model_real_esrgan.enhance(image_data[:,:,::-1])
        image_data = output[:,:,::-1]

    return image_data

def mk_img(req: Request):
    try:
        yield from do_mk_img(req)
    except Exception as e:
        print(traceback.format_exc())

        gc()

        if device != "cpu":
            modelFS.to("cpu")
            modelCS.to("cpu")

            model.model1.to("cpu")
            model.model2.to("cpu")

        gc()

        yield json.dumps({
            "status": 'failed',
            "detail": str(e)
        })

def do_mk_img(req: Request):
    global ckpt_file
    global model, modelCS, modelFS, device
    global model_gfpgan, model_real_esrgan
    global stop_processing

    stop_processing = False

    res = Response()
    res.request = req
    res.images = []

    temp_images.clear()

    # custom model support:
    #  the req.use_stable_diffusion_model needs to be a valid path
    #  to the ckpt file (without the extension).

    needs_model_reload = False
    ckpt_to_use = ckpt_file
    if ckpt_to_use != req.use_stable_diffusion_model:
        ckpt_to_use = req.use_stable_diffusion_model
        needs_model_reload = True

    model.turbo = req.turbo
    if req.use_cpu:
        if device != 'cpu':
            device = 'cpu'

            if model_is_half:
                load_model_ckpt(ckpt_to_use, device)
                needs_model_reload = False

            load_model_gfpgan(gfpgan_file)
            load_model_real_esrgan(real_esrgan_file)
    else:
        if has_valid_gpu:
            prev_device = device
            device = 'cuda'

            if (precision == 'autocast' and (req.use_full_precision or not model_is_half)) or \
                (precision == 'full' and not req.use_full_precision and not force_full_precision):

                load_model_ckpt(ckpt_to_use, device, req.turbo, unet_bs, ('full' if req.use_full_precision else 'autocast'))
                needs_model_reload = False

                if prev_device != device:
                    load_model_gfpgan(gfpgan_file)
                    load_model_real_esrgan(real_esrgan_file)

    if needs_model_reload:
        load_model_ckpt(ckpt_to_use, device, req.turbo, unet_bs, precision)

    if req.use_face_correction != gfpgan_file:
        load_model_gfpgan(req.use_face_correction)

    if req.use_upscale != real_esrgan_file:
        load_model_real_esrgan(req.use_upscale)

    model.cdevice = device
    modelCS.cond_stage_model.device = device

    opt_prompt = req.prompt
    opt_seed = req.seed
    opt_n_iter = 1
    opt_C = 4
    opt_f = 8
    opt_ddim_eta = 0.0
    opt_init_img = req.init_image
    

    print(req.to_string(), '\n    device', device)

    print('\n\n    Using precision:', precision)

    seed_everything(opt_seed)

    batch_size = req.num_outputs
    prompt = opt_prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

    if precision == "autocast" and device != "cpu":
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
        init_image = init_image.to(device)

        if device != "cpu" and precision == "autocast":
            init_image = init_image.half()

        modelFS.to(device)

        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = modelFS.get_first_stage_encoding(modelFS.encode_first_stage(init_image))  # move to latent space

        if req.mask is not None:
            mask = load_mask(req.mask, req.width, req.height, init_latent.shape[2], init_latent.shape[3], True).to(device)
            mask = mask[0][0].unsqueeze(0).repeat(4, 1, 1).unsqueeze(0)
            mask = repeat(mask, '1 ... -> b ...', b=batch_size)

            if device != "cpu" and precision == "autocast":
                mask = mask.half()

        move_fs_to_cpu()

        assert 0. <= req.prompt_strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(req.prompt_strength * req.num_inference_steps)
        print(f"target t_enc is {t_enc} steps")

    if req.save_to_disk_path is not None:
        session_out_path = os.path.join(req.save_to_disk_path, req.session_id)
        os.makedirs(session_out_path, exist_ok=True)
    else:
        session_out_path = None

    seeds = ""
    with torch.no_grad():
        for n in trange(opt_n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):

                with precision_scope("cuda"):
                    modelCS.to(device)
                    uc = None
                    if req.guidance_scale != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [req.negative_prompt])
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
                            c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                    else:
                        c = modelCS.get_learned_conditioning(prompts)

                    modelFS.to(device)

                    partial_x_samples = None
                    last_callback_time = -1
                    def img_callback(x_samples, i):
                        nonlocal partial_x_samples, last_callback_time

                        partial_x_samples = x_samples

                        if req.stream_progress_updates:
                            n_steps = req.num_inference_steps if req.init_image is None else t_enc
                            step_time = time.time() - last_callback_time if last_callback_time != -1 else -1
                            last_callback_time = time.time()

                            progress = {"step": i, "total_steps": n_steps, "step_time": step_time}

                            if req.stream_image_progress and i % 5 == 0:
                                partial_images = []

                                for i in range(batch_size):
                                    x_samples_ddim = modelFS.decode_first_stage(x_samples[i].unsqueeze(0))
                                    x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                                    x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                                    x_sample = x_sample.astype(np.uint8)
                                    img = Image.fromarray(x_sample)
                                    buf = BytesIO()
                                    img.save(buf, format='JPEG')
                                    buf.seek(0)

                                    del img, x_sample, x_samples_ddim
                                    # don't delete x_samples, it is used in the code that called this callback

                                    temp_images[str(req.session_id) + '/' + str(i)] = buf
                                    partial_images.append({'path': f'/image/tmp/{req.session_id}/{i}'})

                                progress['output'] = partial_images

                            yield json.dumps(progress)

                        if stop_processing:
                            raise UserInitiatedStop("User requested that we stop processing")

                    # run the handler
                    try:
                        if handler == _txt2img:
                            x_samples = _txt2img(req.width, req.height, req.num_outputs, req.num_inference_steps, req.guidance_scale, None, opt_C, opt_f, opt_ddim_eta, c, uc, opt_seed, img_callback, mask, req.sampler)
                        else:
                            x_samples = _img2img(init_latent, t_enc, batch_size, req.guidance_scale, c, uc, req.num_inference_steps, opt_ddim_eta, opt_seed, img_callback, mask)

                        yield from x_samples

                        x_samples = partial_x_samples
                    except UserInitiatedStop:
                        if partial_x_samples is None:
                            continue

                        x_samples = partial_x_samples

                    print("saving images")
                    for i in range(batch_size):
                        img_id = base64.b64encode(int(time.time()+i).to_bytes(8, 'big')).decode() # Generate unique ID based on time.
                        img_id = img_id.translate({43:None, 47:None, 61:None})[-8:] # Remove + / = and keep last 8 chars.

                        x_samples_ddim = modelFS.decode_first_stage(x_samples[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                        x_sample = x_sample.astype(np.uint8)
                        img = Image.fromarray(x_sample)

                        has_filters =   (req.use_face_correction is not None and req.use_face_correction.startswith('GFPGAN')) or \
                                        (req.use_upscale is not None and req.use_upscale.startswith('RealESRGAN'))

                        return_orig_img = not has_filters or not req.show_only_filtered_image

                        if stop_processing:
                            return_orig_img = True

                        if req.save_to_disk_path is not None:
                            if return_orig_img:
                                img_out_path = get_base_path(req.save_to_disk_path, req.session_id, prompts[0], img_id, req.output_format)
                                save_image(img, img_out_path)
                            meta_out_path = get_base_path(req.save_to_disk_path, req.session_id, prompts[0], img_id, 'txt')
                            save_metadata(meta_out_path, req, prompts[0], opt_seed)

                        if return_orig_img:
                            img_data = img_to_base64_str(img, req.output_format)
                            res_image_orig = ResponseImage(data=img_data, seed=opt_seed)
                            res.images.append(res_image_orig)

                            if req.save_to_disk_path is not None:
                                res_image_orig.path_abs = img_out_path

                        del img

                        if has_filters and not stop_processing:
                            filters_applied = []
                            if req.use_face_correction:
                                x_sample = apply_filters('gfpgan', x_sample)
                                filters_applied.append(req.use_face_correction)
                            if req.use_upscale:
                                x_sample = apply_filters('real_esrgan', x_sample)
                                filters_applied.append(req.use_upscale)
                            if (len(filters_applied) > 0):
                                filtered_image = Image.fromarray(x_sample)
                                filtered_img_data = img_to_base64_str(filtered_image, req.output_format)
                                response_image = ResponseImage(data=filtered_img_data, seed=req.seed)
                                res.images.append(response_image)
                                if req.save_to_disk_path is not None:
                                    filtered_img_out_path = get_base_path(req.save_to_disk_path, req.session_id, prompts[0], img_id, req.output_format, "_".join(filters_applied))
                                    save_image(filtered_image, filtered_img_out_path)
                                    response_image.path_abs = filtered_img_out_path
                                del filtered_image

                        seeds += str(opt_seed) + ","
                        opt_seed += 1

                    move_fs_to_cpu()
                    gc()
                    del x_samples, x_samples_ddim, x_sample
                    print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

    print('Task completed')

    yield json.dumps(res.json())

def save_image(img, img_out_path):
    try:
        img.save(img_out_path)
    except:
        print('could not save the file', traceback.format_exc())

def save_metadata(meta_out_path, req, prompt, opt_seed):
    metadata = f"""{prompt}
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
Stable Diffusion Model: {req.use_stable_diffusion_model + '.ckpt'}
"""
    try:
        with open(meta_out_path, 'w', encoding='utf-8') as f:
            f.write(metadata)
    except:
        print('could not save the file', traceback.format_exc())

def _txt2img(opt_W, opt_H, opt_n_samples, opt_ddim_steps, opt_scale, start_code, opt_C, opt_f, opt_ddim_eta, c, uc, opt_seed, img_callback, mask, sampler_name):
    shape = [opt_n_samples, opt_C, opt_H // opt_f, opt_W // opt_f]

    if device != "cpu":
        mem = torch.cuda.memory_allocated() / 1e6
        modelCS.to("cpu")
        while torch.cuda.memory_allocated() / 1e6 >= mem:
            time.sleep(1)

    if sampler_name == 'ddim':
        model.make_schedule(ddim_num_steps=opt_ddim_steps, ddim_eta=opt_ddim_eta, verbose=False)

    samples_ddim = model.sample(
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

    yield from samples_ddim

def _img2img(init_latent, t_enc, batch_size, opt_scale, c, uc, opt_ddim_steps, opt_ddim_eta, opt_seed, img_callback, mask):
    # encode (scaled latent)
    z_enc = model.stochastic_encode(
        init_latent,
        torch.tensor([t_enc] * batch_size).to(device),
        opt_seed,
        opt_ddim_eta,
        opt_ddim_steps,
    )
    x_T = None if mask is None else init_latent

    # decode it
    samples_ddim = model.sample(
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

    yield from samples_ddim

def move_fs_to_cpu():
    if device != "cpu":
        mem = torch.cuda.memory_allocated() / 1e6
        modelFS.to("cpu")
        while torch.cuda.memory_allocated() / 1e6 >= mem:
            time.sleep(1)

def gc():
    if device == 'cpu':
        return

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# internal

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd

# utils
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
def img_to_base64_str(img, output_format="PNG"):
    buffered = BytesIO()
    img.save(buffered, format=output_format)
    buffered.seek(0)
    img_byte = buffered.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    return img_str

def base64_str_to_img(img_str):
    img_str = img_str[len("data:image/png;base64,"):]
    data = base64.b64decode(img_str)
    buffered = BytesIO(data)
    img = Image.open(buffered)
    return img
