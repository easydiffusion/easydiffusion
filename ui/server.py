import json
import traceback

import sys
import os

SD_DIR = os.getcwd()
print('started in ', SD_DIR)

SD_UI_DIR = os.getenv('SD_UI_PATH', None)
sys.path.append(os.path.dirname(SD_UI_DIR))

CONFIG_DIR = os.path.abspath(os.path.join(SD_UI_DIR, '..', 'scripts'))
MODELS_DIR = os.path.abspath(os.path.join(SD_DIR, '..', 'models'))

OUTPUT_DIRNAME = "Stable Diffusion UI" # in the user's home folder

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import logging

from sd_internal import Request, Response

app = FastAPI()

model_loaded = False
model_is_loading = False

modifiers_cache = None
outpath = os.path.join(os.path.expanduser("~"), OUTPUT_DIRNAME)

# don't show access log entries for URLs that start with the given prefix
ACCESS_LOG_SUPPRESS_PATH_PREFIXES = ['/ping', '/modifier-thumbnails']

app.mount('/media', StaticFiles(directory=os.path.join(SD_UI_DIR, 'media/')), name="media")

# defaults from https://huggingface.co/blog/stable_diffusion
class ImageRequest(BaseModel):
    session_id: str = "session"
    prompt: str = ""
    negative_prompt: str = ""
    init_image: str = None # base64
    mask: str = None # base64
    num_outputs: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: int = 42
    prompt_strength: float = 0.8
    sampler: str = None # "ddim", "plms", "heun", "euler", "euler_a", "dpm2", "dpm2_a", "lms"
    # allow_nsfw: bool = False
    save_to_disk_path: str = None
    turbo: bool = True
    use_cpu: bool = False
    use_full_precision: bool = False
    use_face_correction: str = None # or "GFPGANv1.3"
    use_upscale: str = None # or "RealESRGAN_x4plus" or "RealESRGAN_x4plus_anime_6B"
    use_stable_diffusion_model: str = "sd-v1-4"
    show_only_filtered_image: bool = False
    output_format: str = "jpeg" # or "png"

    stream_progress_updates: bool = False
    stream_image_progress: bool = False

class SetAppConfigRequest(BaseModel):
    update_branch: str = "main"

@app.get('/')
def read_root():
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    return FileResponse(os.path.join(SD_UI_DIR, 'index.html'), headers=headers)

@app.get('/ping')
async def ping():
    global model_loaded, model_is_loading

    try:
        if model_loaded:
            return {'OK'}

        if model_is_loading:
            return {'ERROR'}

        model_is_loading = True

        from sd_internal import runtime

        runtime.load_model_ckpt(ckpt_to_use=get_initial_model_to_load())

        model_loaded = True
        model_is_loading = False

        return {'OK'}
    except Exception as e:
        print(traceback.format_exc())
        return HTTPException(status_code=500, detail=str(e))

# needs to support the legacy installations
def get_initial_model_to_load():
    custom_weight_path = os.path.join(SD_DIR, 'custom-model.ckpt')
    ckpt_to_use = "sd-v1-4" if not os.path.exists(custom_weight_path) else "custom-model"

    ckpt_to_use = os.path.join(SD_DIR, ckpt_to_use)

    config = getConfig()
    if 'model' in config and 'stable-diffusion' in config['model']:
        model_name = config['model']['stable-diffusion']
        model_path = resolve_model_to_use(model_name)

        if os.path.exists(model_path + '.ckpt'):
            ckpt_to_use = model_path
        else:
            print('Could not find the configured custom model at:', model_path + '.ckpt', '. Using the default one:', ckpt_to_use + '.ckpt')

    return ckpt_to_use

def resolve_model_to_use(model_name):
    if model_name in ('sd-v1-4', 'custom-model'):
        model_path = os.path.join(MODELS_DIR, 'stable-diffusion', model_name)

        legacy_model_path = os.path.join(SD_DIR, model_name)
        if not os.path.exists(model_path + '.ckpt') and os.path.exists(legacy_model_path + '.ckpt'):
            model_path = legacy_model_path
    else:
        model_path = os.path.join(MODELS_DIR, 'stable-diffusion', model_name)

    return model_path

def save_model_to_config(model_name):
    config = getConfig()
    if 'model' not in config:
        config['model'] = {}

    config['model']['stable-diffusion'] = model_name

    setConfig(config)

@app.post('/image')
def image(req : ImageRequest):
    from sd_internal import runtime

    r = Request()
    r.session_id = req.session_id
    r.prompt = req.prompt
    r.negative_prompt = req.negative_prompt
    r.init_image = req.init_image
    r.mask = req.mask
    r.num_outputs = req.num_outputs
    r.num_inference_steps = req.num_inference_steps
    r.guidance_scale = req.guidance_scale
    r.width = req.width
    r.height = req.height
    r.seed = req.seed
    r.prompt_strength = req.prompt_strength
    r.sampler = req.sampler
    # r.allow_nsfw = req.allow_nsfw
    r.turbo = req.turbo
    r.use_cpu = req.use_cpu
    r.use_full_precision = req.use_full_precision
    r.save_to_disk_path = req.save_to_disk_path
    r.use_upscale: str = req.use_upscale
    r.use_face_correction = req.use_face_correction
    r.show_only_filtered_image = req.show_only_filtered_image
    r.output_format = req.output_format

    r.stream_progress_updates = True # the underlying implementation only supports streaming
    r.stream_image_progress = req.stream_image_progress

    r.use_stable_diffusion_model = resolve_model_to_use(req.use_stable_diffusion_model)

    save_model_to_config(req.use_stable_diffusion_model)

    try:
        if not req.stream_progress_updates:
            r.stream_image_progress = False

        res = runtime.mk_img(r)

        if req.stream_progress_updates:
            return StreamingResponse(res, media_type='application/json')
        else: # compatibility mode: buffer the streaming responses, and return the last one
            last_result = None

            for result in res:
                last_result = result

            return json.loads(last_result)
    except Exception as e:
        print(traceback.format_exc())
        return HTTPException(status_code=500, detail=str(e))

@app.get('/image/stop')
def stop():
    try:
        if model_is_loading:
            return {'ERROR'}

        from sd_internal import runtime
        runtime.stop_processing = True

        return {'OK'}
    except Exception as e:
        print(traceback.format_exc())
        return HTTPException(status_code=500, detail=str(e))

@app.get('/image/tmp/{session_id}/{img_id}')
def get_image(session_id, img_id):
    from sd_internal import runtime
    buf = runtime.temp_images[session_id + '/' + img_id]
    buf.seek(0)
    return StreamingResponse(buf, media_type='image/jpeg')

@app.post('/app_config')
async def setAppConfig(req : SetAppConfigRequest):
    try:
        config = {
            'update_branch': req.update_branch
        }

        config_json_str = json.dumps(config)
        config_bat_str = f'@set update_branch={req.update_branch}'
        config_sh_str = f'export update_branch={req.update_branch}'

        config_json_path = os.path.join(CONFIG_DIR, 'config.json')
        config_bat_path = os.path.join(CONFIG_DIR, 'config.bat')
        config_sh_path = os.path.join(CONFIG_DIR, 'config.sh')

        with open(config_json_path, 'w') as f:
            f.write(config_json_str)

        with open(config_bat_path, 'w') as f:
            f.write(config_bat_str)

        with open(config_sh_path, 'w') as f:
            f.write(config_sh_str)

        return {'OK'}
    except Exception as e:
        print(traceback.format_exc())
        return HTTPException(status_code=500, detail=str(e))

def getConfig(default_val={}):
    try:
        config_json_path = os.path.join(CONFIG_DIR, 'config.json')
        if not os.path.exists(config_json_path):
            return default_val
        with open(config_json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        return default_val

def setConfig(config):
    try:
        config_json_path = os.path.join(CONFIG_DIR, 'config.json')

        with open(config_json_path, 'w') as f:
            return json.dump(config, f)
    except:
        print(str(e))
        print(traceback.format_exc())

def getModels():
    models = {
        'active': {
            'stable-diffusion': 'sd-v1-4',
        },
        'options': {
            'stable-diffusion': ['sd-v1-4'],
        },
    }

    # custom models
    sd_models_dir = os.path.join(MODELS_DIR, 'stable-diffusion')
    for file in os.listdir(sd_models_dir):
        if file.endswith('.ckpt'):
            model_name = os.path.splitext(file)[0]
            models['options']['stable-diffusion'].append(model_name)

    # legacy
    custom_weight_path = os.path.join(SD_DIR, 'custom-model.ckpt')
    if os.path.exists(custom_weight_path):
        models['active']['stable-diffusion'] = 'custom-model'
        models['options']['stable-diffusion'].append('custom-model')

    config = getConfig()
    if 'model' in config and 'stable-diffusion' in config['model']:
        models['active']['stable-diffusion'] = config['model']['stable-diffusion']

    return models

@app.get('/get')
def read_web_data(key:str=None):
    if key is None: # /get without parameters, stable-diffusion easter egg.
        return HTTPException(status_code=418, detail="StableDiffusion is drawing a teapot!") # HTTP418 I'm a teapot
    elif key == 'app_config':
        config = getConfig(default_val=None)
        if config is None:
            return HTTPException(status_code=500, detail="Config file is missing or unreadable")
        return config
    elif key == 'models':
        return getModels()
    elif key == 'modifiers': return FileResponse(os.path.join(SD_UI_DIR, 'modifiers.json'), headers=NOCACHE_HEADERS)
    elif key == 'output_dir': return {outpath}
    else:
        return HTTPException(status_code=404, detail=f'Request for unknown {key}') # HTTP404 Not Found

# don't log certain requests
class LogSuppressFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        path = record.getMessage()
        for prefix in ACCESS_LOG_SUPPRESS_PATH_PREFIXES:
            if path.find(prefix) != -1:
                return False

        return True

logging.getLogger('uvicorn.access').addFilter(LogSuppressFilter())

# start the browser ui
import webbrowser; webbrowser.open('http://localhost:9000')