import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

# my stuff
from . import Request, Response, Image as ResponseImage
import base64
from io import BytesIO


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

# local
model = None
device = None
sampler_plms = None
sampler_ddim = None

# api
def load_model(config="configs/stable-diffusion/v1-inference.yaml", ckpt="models/ldm/stable-diffusion-v1/model.ckpt"):
    global model, device
    global sampler_plms, sampler_ddim

    config = OmegaConf.load(f"{config}")
    model = load_model_from_config(config, f"{ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler_plms = PLMSSampler(model)
    sampler_ddim = DDIMSampler(model)

def mk_img(req: Request):
    res = Response()
    res.images = []

    sampler = sampler_ddim

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
    opt_precision = "autocast"
    opt_strength = req.prompt_strength
    opt_skip_save = False
    opt_init_img = req.init_image

    print(opt_prompt, ': seed', opt_seed, 'num_inference_steps', opt_ddim_steps, 'guidance_scale', opt_scale, 'w', opt_W, 'h', opt_H, 'allow_nsfw', req.allow_nsfw)

    seed_everything(opt_seed)

    batch_size = opt_n_samples
    prompt = opt_prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

    if req.init_image is None:
        handler = _txt2img

        init_latent = None
        t_enc = None
    else:
        handler = _img2img

        init_image = load_img(req.init_image)
        init_image = init_image.to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        assert 0. <= opt_strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(opt_strength * opt_ddim_steps)
        print(f"target t_enc is {t_enc} steps")

    sampler.make_schedule(ddim_num_steps=opt_ddim_steps, ddim_eta=opt_ddim_eta, verbose=False)

    precision_scope = autocast if opt_precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for n in trange(opt_n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt_scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        # run the handler
                        if handler == _txt2img:
                            x_samples = _txt2img(opt_W, opt_H, opt_n_samples, opt_ddim_steps, opt_scale, None, opt_C, opt_f, opt_ddim_eta, c, uc)
                        else:
                            x_samples = _img2img(init_latent, t_enc, batch_size, opt_scale, c, uc)

                        # process the generated images
                        if req.allow_nsfw:
                            x_checked_image = x_samples
                        else:
                            x_checked_image, has_nsfw_concept = check_safety(x_samples)

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        if not opt_skip_save:
                            for i, x_sample in enumerate(x_checked_image_torch):
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img_data = img_to_base64_str(img)

                                res.images.append(ResponseImage(data=img_data))

    return res

def _txt2img(opt_W, opt_H, opt_n_samples, opt_ddim_steps, opt_scale, start_code, opt_C, opt_f, opt_ddim_eta, c, uc):
    sampler = sampler_plms

    shape = [opt_C, opt_H // opt_f, opt_W // opt_f]
    samples_ddim, _ = sampler.sample(S=opt_ddim_steps,
                                        conditioning=c,
                                        batch_size=opt_n_samples,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=opt_scale,
                                        unconditional_conditioning=uc,
                                        eta=opt_ddim_eta,
                                        x_T=start_code)

    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

    return x_samples_ddim

def _img2img(init_latent, t_enc, batch_size, opt_scale, c, uc):
    sampler = sampler_ddim

    # this part is specific to img2img
    # encode (scaled latent)
    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
    # decode it
    samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt_scale,
                                unconditional_conditioning=uc,)

    x_samples = model.decode_first_stage(samples)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy() # extra

    return x_samples

# internal

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

def load_img(img_str):
    image = base64_str_to_img(img_str).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from base64")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
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
