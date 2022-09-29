import sys
import os
import uuid
import re
import torch
import traceback
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import logging
from einops import rearrange
from PIL import Image, ImageOps, ImageChops
from ldm.generate import Generate
import transformers

from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

transformers.logging.set_verbosity_error()

from . import Request, Response, Image as ResponseImage
import base64
import json
from io import BytesIO

filename_regex = re.compile('[^a-zA-Z0-9]')

generator = None

gfpgan_file = None
real_esrgan_file = None
model_gfpgan = None
model_real_esrgan = None

device = None
precision = 'autocast'

has_valid_gpu = False
force_full_precision = False

# local
stop_processing = False
temp_images = {}

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

def load_model_ckpt(ckpt_to_use, device_to_use='cuda', precision_to_use='autocast'):
    global generator

    device = device_to_use if has_valid_gpu else 'cpu'
    precision = precision_to_use if not force_full_precision else 'full'

    try:
        config = 'configs/models.yaml'
        model = 'stable-diffusion-1.4'

        models = OmegaConf.load(config)
        width = models[model].width
        height = models[model].height
        config = models[model].config
        weights = ckpt_to_use + '.ckpt'
    except (FileNotFoundError, IOError, KeyError) as e:
        print(f'{e}. Aborting.')
        sys.exit(-1)

    generator = Generate(
        width=width,
        height=height,
        sampler_name='ddim',
        weights=weights,
        full_precision=(precision == 'full'),
        config=config,
        grid=False,
        # this is solely for recreating the prompt
        seamless=False,
        embedding_path=None,
        device_type=device,
        ignore_ctrl_c=True,
    )

    # gets rid of annoying messages about random seed
    logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

    # preload the model
    generator.load_model()

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
        model_real_esrgan = RealESRGANer(scale=2, model_path=model_path, model=model_to_use, pre_pad=0, half=(precision != 'full'))

    model_real_esrgan.model.name = real_esrgan_to_use

    print('loaded ', real_esrgan_to_use, 'to', device, 'precision', precision)

def mk_img(req: Request):
    try:
        yield from do_mk_img(req)
    except Exception as e:
        print(traceback.format_exc())

        gc()

        # if device != "cpu":
        #     modelFS.to("cpu")
        #     modelCS.to("cpu")

        #     model.model1.to("cpu")
        #     model.model2.to("cpu")

        # gc()

        yield json.dumps({
            "status": 'failed',
            "detail": str(e)
        })

def do_mk_img(req: Request):
    stop_processing = False

    if req.use_face_correction != gfpgan_file:
        load_model_gfpgan(req.use_face_correction)

    if req.use_upscale != real_esrgan_file:
        load_model_real_esrgan(req.use_upscale)

    init_image = None
    init_mask = None

    if req.init_image is not None:
        image = base64_str_to_img(req.init_image)

        w, h = image.size
        print(f"loaded input image of size ({w}, {h}) from base64")
        if req.width is not None and req.height is not None:
            h, w = req.height, req.width

        w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
        image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
        init_image = generator._create_init_image(image)

        if generator._has_transparency(image) and req.mask is None:      # if image has a transparent area and no mask was provided, then try to generate mask
            print('>> Initial image has transparent areas. Will inpaint in these regions.')
            if generator._check_for_erasure(image):
                print(
                    '>> WARNING: Colors underneath the transparent region seem to have been erased.\n',
                    '>>          Inpainting will be suboptimal. Please preserve the colors when making\n',
                    '>>          a transparency mask, or provide mask explicitly using --init_mask (-M).'
                )
            init_mask = generator._create_init_mask(image)                   # this returns a torch tensor

        if device != "cpu" and precision != "full":
            init_image = init_image.half()

        if req.mask is not None:
            image = base64_str_to_img(req.mask)

            image = ImageChops.invert(image)

            w, h = image.size
            print(f"loaded input image of size ({w}, {h}) from base64")
            if req.width is not None and req.height is not None:
                h, w = req.height, req.width

            w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
            image = image.resize((w, h), resample=Image.Resampling.LANCZOS)

            init_mask = generator._create_init_mask(image)

        if init_mask is not None:
            req.sampler = 'plms' # hack to force the underlying implementation to initialize DDIM properly

    result = generator.prompt2image(
        req.prompt,
        iterations     =    req.num_outputs,
        steps          =    req.num_inference_steps,
        seed           =    req.seed,
        cfg_scale      =    req.guidance_scale,
        ddim_eta       =    0.0,
        skip_normalize =    False,
        image_callback =    None,
        step_callback  =    None,
        width          =    req.width,
        height         =    req.height,
        sampler_name   =    req.sampler,
        seamless       =    False,
        log_tokenization=  False,
        with_variations =   None,
        variation_amount =  0.0,
        # these are specific to img2img and inpaint
        init_img       =    init_image,
        init_mask      =    init_mask,
        fit            =    False,
        strength       =    req.prompt_strength,
        init_img_is_path = False,
        # these are specific to GFPGAN/ESRGAN
        gfpgan_strength=    0,
        save_original  =    False,
        upscale        =    None,
        negative_prompt=    req.negative_prompt,
    )

    has_filters =   (req.use_face_correction is not None and req.use_face_correction.startswith('GFPGAN')) or \
                    (req.use_upscale is not None and req.use_upscale.startswith('RealESRGAN'))

    print('has filter', has_filters)

    return_orig_img = not has_filters or not req.show_only_filtered_image

    res = Response()
    res.request = req
    res.images = []

    if req.save_to_disk_path is not None:
        session_out_path = os.path.join(req.save_to_disk_path, req.session_id)
        os.makedirs(session_out_path, exist_ok=True)
    else:
        session_out_path = None

    for img, seed in result:
        if req.save_to_disk_path is not None:
            prompt_flattened = filename_regex.sub('_', req.prompt)
            prompt_flattened = prompt_flattened[:50]

            img_id = str(uuid.uuid4())[-8:]

            file_path = f"{prompt_flattened}_{img_id}"
            img_out_path = os.path.join(session_out_path, f"{file_path}.{req.output_format}")
            meta_out_path = os.path.join(session_out_path, f"{file_path}.txt")

            if return_orig_img:
                save_image(img, img_out_path)

            save_metadata(meta_out_path, req.prompt, seed, req.width, req.height, req.num_inference_steps, req.guidance_scale, req.prompt_strength, req.use_face_correction, req.use_upscale, req.sampler, req.negative_prompt)

        if return_orig_img:
            img_data = img_to_base64_str(img)
            res_image_orig = ResponseImage(data=img_data, seed=seed)
            res.images.append(res_image_orig)

            if req.save_to_disk_path is not None:
                res_image_orig.path_abs = img_out_path

        if has_filters and not stop_processing:
            print('Applying filters..')

            gc()
            filters_applied = []

            np_img = img.convert('RGB')
            np_img = np.array(np_img, dtype=np.uint8)

            if req.use_face_correction:
                _, _, np_img = model_gfpgan.enhance(np_img, has_aligned=False, only_center_face=False, paste_back=True)
                filters_applied.append(req.use_face_correction)

            if req.use_upscale:
                np_img, _ = model_real_esrgan.enhance(np_img)
                filters_applied.append(req.use_upscale)

            filtered_image = Image.fromarray(np_img)

            filtered_img_data = img_to_base64_str(filtered_image)
            res_image_filtered = ResponseImage(data=filtered_img_data, seed=seed)
            res.images.append(res_image_filtered)

            filters_applied = "_".join(filters_applied)

            if req.save_to_disk_path is not None:
                filtered_img_out_path = os.path.join(session_out_path, f"{file_path}_{filters_applied}.{req.output_format}")
                save_image(filtered_image, filtered_img_out_path)
                res_image_filtered.path_abs = filtered_img_out_path

            del filtered_image
        
        del img

    print('Task completed')

    yield json.dumps(res.json())

def save_image(img, img_out_path):
    try:
        img.save(img_out_path)
    except:
        print('could not save the file', traceback.format_exc())

def save_metadata(meta_out_path, prompt, seed, width, height, num_inference_steps, guidance_scale, prompt_strength, use_correct_face, use_upscale, sampler_name, negative_prompt):
    metadata = f"{prompt}\nWidth: {width}\nHeight: {height}\nSeed: {seed}\nSteps: {num_inference_steps}\nGuidance Scale: {guidance_scale}\nPrompt Strength: {prompt_strength}\nUse Face Correction: {use_correct_face}\nUse Upscaling: {use_upscale}\nSampler: {sampler_name}\nNegative Prompt: {negative_prompt}"

    try:
        with open(meta_out_path, 'w') as f:
            f.write(metadata)
    except:
        print('could not save the file', traceback.format_exc())

def gc():
    if device == 'cpu':
        return

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

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
