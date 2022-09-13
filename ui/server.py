import json
import traceback

import sys
import os

SCRIPT_DIR = os.getcwd()
print('started in ', SCRIPT_DIR)

SD_UI_DIR = os.getenv('SD_UI_PATH', None)
sys.path.append(os.path.dirname(SD_UI_DIR))

CONFIG_DIR = os.path.join(SD_UI_DIR, '..', 'scripts')

OUTPUT_DIRNAME = "Stable Diffusion UI" # in the user's home folder

from fastapi import FastAPI, HTTPException
from starlette.responses import FileResponse
from pydantic import BaseModel
import logging

from sd_internal import Request, Response

app = FastAPI()

model_loaded = False
model_is_loading = False

modifiers_cache = None
outpath = os.path.join(os.path.expanduser("~"), OUTPUT_DIRNAME)

# defaults from https://huggingface.co/blog/stable_diffusion
class ImageRequest(BaseModel):
    prompt: str = ""
    init_image: str = None # base64
    mask: str = None # base64
    num_outputs: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: int = 42
    prompt_strength: float = 0.8
    # allow_nsfw: bool = False
    save_to_disk_path: str = None
    turbo: bool = True
    use_cpu: bool = False
    use_full_precision: bool = False
    use_face_correction: str = None # or "GFPGANv1.3"
    use_upscale: str = None # or "RealESRGAN_x4plus" or "RealESRGAN_x4plus_anime_6B"
    show_only_filtered_image: bool = False

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
        runtime.load_model_ckpt(ckpt_to_use="sd-v1-4")

        model_loaded = True
        model_is_loading = False

        return {'OK'}
    except Exception as e:
        print(traceback.format_exc())
        return HTTPException(status_code=500, detail=str(e))

@app.post('/image')
def image(req : ImageRequest):
    from sd_internal import runtime

    r = Request()
    r.prompt = req.prompt
    r.init_image = req.init_image
    r.mask = req.mask
    r.num_outputs = req.num_outputs
    r.num_inference_steps = req.num_inference_steps
    r.guidance_scale = req.guidance_scale
    r.width = req.width
    r.height = req.height
    r.seed = req.seed
    r.prompt_strength = req.prompt_strength
    # r.allow_nsfw = req.allow_nsfw
    r.turbo = req.turbo
    r.use_cpu = req.use_cpu
    r.use_full_precision = req.use_full_precision
    r.save_to_disk_path = req.save_to_disk_path
    r.use_upscale: str = req.use_upscale
    r.use_face_correction = req.use_face_correction
    r.show_only_filtered_image = req.show_only_filtered_image

    try:
        res: Response = runtime.mk_img(r)

        return res.json()
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

@app.get('/app_config')
def getAppConfig():
    try:
        config_json_path = os.path.join(CONFIG_DIR, 'config.json')

        if not os.path.exists(config_json_path):
            return HTTPException(status_code=500, detail="No config file")

        with open(config_json_path, 'r') as f:
            config_json_str = f.read()
            config = json.loads(config_json_str)
            return config
    except Exception as e:
        print(traceback.format_exc())
        return HTTPException(status_code=500, detail=str(e))

@app.get('/media/ding.mp3')
def read_ding():
    return FileResponse(os.path.join(SD_UI_DIR, 'media/ding.mp3'))

@app.get('/media/kofi.png')
def read_modifiers():
    return FileResponse(os.path.join(SD_UI_DIR, 'media/kofi.png'))

@app.get('/modifiers.json')
def read_modifiers():
    return FileResponse(os.path.join(SD_UI_DIR, 'modifiers.json'))

@app.get('/output_dir')
def read_home_dir():
    return {outpath}

# don't log /ping requests
class HealthCheckLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find('/ping') == -1

logging.getLogger('uvicorn.access').addFilter(HealthCheckLogFilter())

# start the browser ui
import webbrowser; webbrowser.open('http://localhost:9000')
