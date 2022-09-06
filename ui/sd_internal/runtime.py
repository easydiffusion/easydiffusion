import os, re
import traceback
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
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

import uuid

logging.set_verbosity_error()

# consts
config_yaml = "optimizedSD/v1-inference.yaml"

# api stuff
from . import Request, Response, Image as ResponseImage
import base64
from io import BytesIO

# local
session_id = str(uuid.uuid4())[-8:]

ckpt = None
model = None
modelCS = None
modelFS = None
model_is_half = False
model_fs_is_half = False
device = None
unet_bs = 1
precision = 'autocast'
sampler_plms = None
sampler_ddim = None

# api
def load_model(ckpt_to_use, device_to_use='cuda', turbo=False, unet_bs_to_use=1, precision_to_use='autocast', half_model_fs=False):
    global ckpt, model, modelCS, modelFS, model_is_half, device, unet_bs, precision, model_fs_is_half

    ckpt = ckpt_to_use
    device = device_to_use
    precision = precision_to_use
    unet_bs = unet_bs_to_use

    sd = load_model_from_config(f"{ckpt}")
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
        model_is_half = True
    else:
        model_is_half = False

    if half_model_fs:
        modelFS.half()
        model_fs_is_half = True
    else:
        model_fs_is_half = False

def mk_img(req: Request):
    global modelFS, device

    res = Response()
    res.images = []

    model.turbo = req.turbo
    if req.use_cpu:
        device = 'cpu'

        if model_is_half:
            print('reloading model for cpu')
            load_model(ckpt, device)
    else:
        device = 'cuda'

        if (precision == 'autocast' and (req.use_full_precision or not model_is_half)) or \
            (precision == 'full' and not req.use_full_precision) or \
            (req.init_image is None and model_fs_is_half) or \
            (req.init_image is not None and not model_fs_is_half):

            print('reloading model for cuda')
            load_model(ckpt, device, model.turbo, unet_bs, ('full' if req.use_full_precision else 'autocast'), half_model_fs=(req.init_image is not None and not req.use_full_precision))

    model.cdevice = device
    modelCS.cond_stage_model.device = device

    opt_prompt = req.prompt
    opt_seed = req.seed
    opt_n_samples = req.num_outputs
    opt_n_iter = 1
    opt_scale = req.guidance_scale
    opt_C = 4
    opt_H = req.height
    opt_W = req.width
    opt_f = 8
    opt_ddim_steps = req.num_inference_steps
    opt_ddim_eta = 0.0
    opt_strength = req.prompt_strength
    opt_save_to_disk_path = req.save_to_disk_path
    opt_init_img = req.init_image
    opt_format = 'png'

    print(req.to_string(), '\n    device', device)

    seed_everything(opt_seed)

    batch_size = opt_n_samples
    prompt = opt_prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

    if precision == "autocast" and device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    if req.init_image is None:
        handler = _txt2img

        init_latent = None
        t_enc = None
    else:
        handler = _img2img

        init_image = load_img(req.init_image)
        init_image = init_image.to(device)

        if device != "cpu" and precision == "autocast":
            init_image = init_image.half()

        modelFS.to(device)

        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = modelFS.get_first_stage_encoding(modelFS.encode_first_stage(init_image))  # move to latent space

        if device != "cpu":
            mem = torch.cuda.memory_allocated() / 1e6
            modelFS.to("cpu")
            while torch.cuda.memory_allocated() / 1e6 >= mem:
                time.sleep(1)

        assert 0. <= opt_strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(opt_strength * opt_ddim_steps)
        print(f"target t_enc is {t_enc} steps")

    if opt_save_to_disk_path is not None:
        session_out_dir = session_id
        session_out_path = os.path.join(opt_save_to_disk_path, session_out_dir)
        os.makedirs(session_out_path, exist_ok=True)
    else:
        session_out_dir = None
        session_out_path = None

    seeds = ""
    with torch.no_grad():
        for n in trange(opt_n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):

                with precision_scope("cuda"):
                    modelCS.to(device)
                    uc = None
                    if opt_scale != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [""])
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

                    # run the handler
                    if handler == _txt2img:
                        x_samples = _txt2img(opt_W, opt_H, opt_n_samples, opt_ddim_steps, opt_scale, None, opt_C, opt_f, opt_ddim_eta, c, uc, opt_seed)
                    else:
                        x_samples = _img2img(init_latent, t_enc, batch_size, opt_scale, c, uc, opt_ddim_steps, opt_ddim_eta, opt_seed)

                    modelFS.to(device)

                    print("saving images")
                    for i in range(batch_size):

                        x_samples_ddim = modelFS.decode_first_stage(x_samples[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                        img = Image.fromarray(x_sample.astype(np.uint8))

                        img_data = img_to_base64_str(img)
                        response_image = ResponseImage(data=img_data, seed=opt_seed)

                        if opt_save_to_disk_path is not None:
                            try:
                                prompt_flattened = "_".join(re.split(":| ", prompts[0]))
                                prompt_flattened = prompt_flattened.replace(',', '')
                                prompt_flattened = prompt_flattened[:50]

                                img_id = str(uuid.uuid4())[-8:]

                                file_path = f"{prompt_flattened}_{img_id}"
                                file_name = f"{file_path}.{opt_format}"
                                img_out_path = os.path.join(session_out_path, file_name)
                                meta_out_path = os.path.join(session_out_path, f"{file_path}.txt")

                                metadata = f"{prompts[0]}\nWidth: {opt_W}\nHeight: {opt_H}\nSeed: {opt_seed}\nSteps: {opt_ddim_steps}\nGuidance Scale: {opt_scale}"
                                response_image.url = f"{session_out_dir}/{file_name}"
                                img.save(img_out_path)
                                with open(meta_out_path, 'w') as f:
                                    f.write(metadata)
                            except:
                                print('could not save the file', traceback.format_exc())

                        res.images.append(response_image)
                        seeds += str(opt_seed) + ","
                        opt_seed += 1

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)
                    del x_samples
                    print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

    return res

def _txt2img(opt_W, opt_H, opt_n_samples, opt_ddim_steps, opt_scale, start_code, opt_C, opt_f, opt_ddim_eta, c, uc, opt_seed):
    shape = [opt_n_samples, opt_C, opt_H // opt_f, opt_W // opt_f]

    if device != "cpu":
        mem = torch.cuda.memory_allocated() / 1e6
        modelCS.to("cpu")
        while torch.cuda.memory_allocated() / 1e6 >= mem:
            time.sleep(1)

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
        sampler = 'plms',
    )

    return samples_ddim

def _img2img(init_latent, t_enc, batch_size, opt_scale, c, uc, opt_ddim_steps, opt_ddim_eta, opt_seed):
    # encode (scaled latent)
    z_enc = model.stochastic_encode(
        init_latent,
        torch.tensor([t_enc] * batch_size).to(device),
        opt_seed,
        opt_ddim_eta,
        opt_ddim_steps,
    )
    # decode it
    samples_ddim = model.sample(
        t_enc,
        c,
        z_enc,
        unconditional_guidance_scale=opt_scale,
        unconditional_conditioning=uc,
        sampler = 'ddim'
    )

    return samples_ddim

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

def load_img(img_str):
    image = base64_str_to_img(img_str).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from base64")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

# https://stackoverflow.com/a/61114178
def img_to_base64_str(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
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
