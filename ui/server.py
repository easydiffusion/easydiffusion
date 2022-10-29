"""server.py: FastAPI SD-UI Web Host.
Notes:
    async endpoints always run on the main thread. Without they run on the thread pool.
"""
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
UI_PLUGINS_DIR = os.path.abspath(os.path.join(SD_DIR, '..', 'plugins', 'ui'))

OUTPUT_DIRNAME = "Stable Diffusion UI" # in the user's home folder
TASK_TTL = 15 * 60 # Discard last session's task timeout
APP_CONFIG_DEFAULTS = {
    # auto: selects the cuda device with the most free memory, cuda: use the currently active cuda device.
    'render_devices': ['auto'], # ['cuda'] or ['CPU', 'GPU:0', 'GPU:1', ...] or ['cpu']
    'update_branch': 'main',
}
APP_CONFIG_DEFAULT_MODELS = [
    # needed to support the legacy installations
    'custom-model', # Check if user has a custom model, use it first.
    'sd-v1-4', # Default fallback.
]
APP_CONFIG_DEFAULT_VAE = [
    'vae-ft-mse-840000-ema-pruned', # Default fallback.
]

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import logging
#import queue, threading, time
from typing import Any, Generator, Hashable, List, Optional, Union

from sd_internal import Request, Response, task_manager

app = FastAPI()

modifiers_cache = None
outpath = os.path.join(os.path.expanduser("~"), OUTPUT_DIRNAME)

os.makedirs(UI_PLUGINS_DIR, exist_ok=True)

# don't show access log entries for URLs that start with the given prefix
ACCESS_LOG_SUPPRESS_PATH_PREFIXES = ['/ping', '/image', '/modifier-thumbnails']

NOCACHE_HEADERS={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}

app.mount('/media', StaticFiles(directory=os.path.join(SD_UI_DIR, 'media')), name="media")
app.mount('/plugins', StaticFiles(directory=UI_PLUGINS_DIR), name="plugins")

config_cached = None
config_last_mod_time = 0
def getConfig(default_val=APP_CONFIG_DEFAULTS):
    global config_cached, config_last_mod_time
    try:
        config_json_path = os.path.join(CONFIG_DIR, 'config.json')
        if not os.path.exists(config_json_path):
            return default_val
        if config_last_mod_time > 0 and config_cached is not None:
            # Don't read if file was not modified
            mtime = os.path.getmtime(config_json_path)
            if mtime <= config_last_mod_time:
                return config_cached
        with open(config_json_path, 'r', encoding='utf-8') as f:
            config_cached = json.load(f)
        config_last_mod_time = os.path.getmtime(config_json_path)
        return config_cached
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        return default_val

def setConfig(config):
    try: # config.json
        config_json_path = os.path.join(CONFIG_DIR, 'config.json')
        with open(config_json_path, 'w', encoding='utf-8') as f:
            json.dump(config, f)
    except:
        print(traceback.format_exc())

    if 'render_devices' in config:
        gpu_devices = list(filter(lambda dev: dev.lower().startswith('gpu') or dev.lower().startswith('cuda'), config['render_devices']))
    else:
        gpu_devices = []

    has_first_cuda_device = False
    for device in gpu_devices:
        if not task_manager.is_first_cuda_device(device): continue
        has_first_cuda_device = True
        break
    if len(gpu_devices) > 0 and not has_first_cuda_device:
        print('WARNING: GFPGANer only works on GPU:0, use CUDA_VISIBLE_DEVICES if GFPGANer is needed on a specific GPU.')
        print('Using CUDA_VISIBLE_DEVICES will remap the selected devices starting at GPU:0 fixing GFPGANer')

    try: # config.bat
        config_bat = [
            f"@set update_branch={config['update_branch']}"
        ]
        if len(gpu_devices) > 0 and not has_first_cuda_device:
            config_bat.append('::Set the devices visible inside SD-UI here')
            config_bat.append(f"::@set CUDA_VISIBLE_DEVICES={','.join(gpu_devices)}") # Needs better detection for edge cases, add as a comment for now.
            print('Add the line "@set CUDA_VISIBLE_DEVICES=N" where N is the GPUs to use to config.bat')
        config_bat_path = os.path.join(CONFIG_DIR, 'config.bat')
        with open(config_bat_path, 'w', encoding='utf-8') as f:
            f.write('\r\n'.join(config_bat))
    except Exception as e:
        print(traceback.format_exc())

    try: # config.sh
        config_sh = [
            '#!/bin/bash',
            f"export update_branch={config['update_branch']}"
        ]
        if len(gpu_devices) > 0 and not has_first_cuda_device:
            config_sh.append('#Set the devices visible inside SD-UI here')
            config_sh.append(f"#CUDA_VISIBLE_DEVICES={','.join(gpu_devices)}") # Needs better detection for edge cases, add as a comment for now.
            print('Add the line "CUDA_VISIBLE_DEVICES=N" where N is the GPUs to use to config.sh')
        config_sh_path = os.path.join(CONFIG_DIR, 'config.sh')
        with open(config_sh_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(config_sh))
    except Exception as e:
        print(traceback.format_exc())

def resolve_model_to_use(model_name:str, model_type:str, model_dir:str, model_extension:str, default_models=[]):
    model_dirs = [os.path.join(MODELS_DIR, model_dir), SD_DIR]
    if not model_name: # When None try user configured model.
        config = getConfig()
        if 'model' in config and model_type in config['model']:
            model_name = config['model'][model_type]
    if model_name:
        # Check models directory
        models_dir_path = os.path.join(MODELS_DIR, model_dir, model_name)
        if os.path.exists(models_dir_path + model_extension):
            return models_dir_path
        if os.path.exists(model_name + model_extension):
            # Direct Path to file
            model_name = os.path.abspath(model_name)
            return model_name
    # Default locations
    if model_name in default_models:
        default_model_path = os.path.join(SD_DIR, model_name)
        if os.path.exists(default_model_path + model_extension):
            return default_model_path
    # Can't find requested model, check the default paths.
    for default_model in default_models:
        for model_dir in model_dirs:
            default_model_path = os.path.join(model_dir, default_model)
            if os.path.exists(default_model_path + model_extension):
                if model_name is not None:
                    print(f'Could not find the configured custom model {model_name}{model_extension}. Using the default one: {default_model_path}{model_extension}')
                return default_model_path
    raise Exception('No valid models found.')

def resolve_ckpt_to_use(model_name:str=None):
    return resolve_model_to_use(model_name, model_type='stable-diffusion', model_dir='stable-diffusion', model_extension='.ckpt', default_models=APP_CONFIG_DEFAULT_MODELS)

def resolve_vae_to_use(ckpt_model_path:str=None):
    if ckpt_model_path is not None:
        if os.path.exists(ckpt_model_path + '.vae.pt'):
            return ckpt_model_path

        ckpt_model_name = os.path.basename(ckpt_model_path)
        model_dirs = [os.path.join(MODELS_DIR, 'stable-diffusion'), SD_DIR]
        for model_dir in model_dirs:
            default_model_path = os.path.join(model_dir, ckpt_model_name)
            if os.path.exists(default_model_path + '.vae.pt'):
                return default_model_path

    return resolve_model_to_use(model_name=None, model_type='vae', model_dir='stable-diffusion', model_extension='.vae.pt', default_models=APP_CONFIG_DEFAULT_VAE)

class SetAppConfigRequest(BaseModel):
    update_branch: str = None
    render_devices: Union[List[str], List[int], str, int] = None
    model_vae: str = None

@app.post('/app_config')
async def setAppConfig(req : SetAppConfigRequest):
    config = getConfig()
    if req.update_branch:
        config['update_branch'] = req.update_branch
    if req.render_devices and hasattr(req.render_devices, "__len__"): # strings, array of strings or numbers.
        render_devices = []
        if isinstance(req.render_devices, str):
            req.render_devices = req.render_devices.split(',')
        if isinstance(req.render_devices, list):
            for gpu in req.render_devices:
                if isinstance(req.render_devices, int):
                    render_devices.append('GPU:' + gpu)
                else:
                    render_devices.append(gpu)
        if isinstance(req.render_devices, int):
            render_devices.append('GPU:' + req.render_devices)
        if len(render_devices) > 0:
            config['render_devices'] = render_devices
    if req.model_vae:
        if 'model' not in config:
            config['model'] = {}
        config['model']['vae'] = req.model_vae
    try:
        setConfig(config)
        return JSONResponse({'status': 'OK'}, headers=NOCACHE_HEADERS)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def getModels():
    models = {
        'active': {
            'stable-diffusion': 'sd-v1-4',
            'vae': '',
        },
        'options': {
            'stable-diffusion': ['sd-v1-4'],
            'vae': [],
        },
    }

    # custom models
    sd_models_dir = os.path.join(MODELS_DIR, 'stable-diffusion')
    for model_type, model_extension in [('stable-diffusion', '.ckpt'), ('vae', '.vae.pt')]:
        for file in os.listdir(sd_models_dir):
            if file.endswith(model_extension):
                model_name = file[:-len(model_extension)]
                models['options'][model_type].append(model_name)

    # legacy
    custom_weight_path = os.path.join(SD_DIR, 'custom-model.ckpt')
    if os.path.exists(custom_weight_path):
        models['options']['stable-diffusion'].append('custom-model')

    models['active']['vae'] = os.path.basename(task_manager.default_vae_to_load)

    return models

def getUIPlugins():
    plugins = []

    for file in os.listdir(UI_PLUGINS_DIR):
        if file.endswith('.plugin.js'):
            plugins.append(f'/plugins/{file}')

    return plugins

@app.get('/get/{key:path}')
def read_web_data(key:str=None):
    if not key: # /get without parameters, stable-diffusion easter egg.
        raise HTTPException(status_code=418, detail="StableDiffusion is drawing a teapot!") # HTTP418 I'm a teapot
    elif key == 'app_config':
        config = getConfig(default_val=None)
        if config is None:
            raise HTTPException(status_code=500, detail="Config file is missing or unreadable")
        return JSONResponse(config, headers=NOCACHE_HEADERS)
    elif key == 'models':
        return JSONResponse(getModels(), headers=NOCACHE_HEADERS)
    elif key == 'modifiers': return FileResponse(os.path.join(SD_UI_DIR, 'modifiers.json'), headers=NOCACHE_HEADERS)
    elif key == 'output_dir': return JSONResponse({ 'output_dir': outpath }, headers=NOCACHE_HEADERS)
    elif key == 'ui_plugins': return JSONResponse(getUIPlugins(), headers=NOCACHE_HEADERS)
    else:
        raise HTTPException(status_code=404, detail=f'Request for unknown {key}') # HTTP404 Not Found

@app.get('/ping') # Get server and optionally session status.
def ping(session_id:str=None):
    if task_manager.is_alive() <= 0: # Check that render threads are alive.
        if task_manager.current_state_error: raise HTTPException(status_code=500, detail=str(task_manager.current_state_error))
        raise HTTPException(status_code=500, detail='Render thread is dead.')
    if task_manager.current_state_error and not isinstance(task_manager.current_state_error, StopAsyncIteration): raise HTTPException(status_code=500, detail=str(task_manager.current_state_error))
    # Alive
    response = {'status': str(task_manager.current_state)}
    if session_id:
        task = task_manager.get_cached_task(session_id, update_ttl=True)
        if task:
            response['task'] = id(task)
            if task.lock.locked():
                response['session'] = 'running'
            elif isinstance(task.error, StopAsyncIteration):
                response['session'] = 'stopped'
            elif task.error:
                response['session'] = 'error'
            elif not task.buffer_queue.empty():
                response['session'] = 'buffer'
            elif task.response:
                response['session'] = 'completed'
            else:
                response['session'] = 'pending'
    return JSONResponse(response, headers=NOCACHE_HEADERS)

def save_model_to_config(model_name):
    config = getConfig()
    if 'model' not in config:
        config['model'] = {}

    config['model']['stable-diffusion'] = model_name
    setConfig(config)

@app.post('/render')
def render(req : task_manager.ImageRequest):
    if req.use_cpu and task_manager.is_alive('cpu') <= 0: raise HTTPException(status_code=403, detail=f'CPU rendering is not enabled in config.json or the thread has died...') # HTTP403 Forbidden
    if req.use_face_correction and task_manager.is_alive(0) <= 0: #TODO Remove when GFPGANer is fixed upstream.
        raise HTTPException(status_code=412, detail=f'GFPGANer only works GPU:0, use CUDA_VISIBLE_DEVICES if GFPGANer is needed on a specific GPU.') # HTTP412 Precondition Failed
    try:
        save_model_to_config(req.use_stable_diffusion_model)
        req.use_stable_diffusion_model = resolve_ckpt_to_use(req.use_stable_diffusion_model)
        req.use_vae_model = resolve_vae_to_use(ckpt_model_path=req.use_stable_diffusion_model)
        new_task = task_manager.render(req)
        response = {
            'status': str(task_manager.current_state), 
            'queue': len(task_manager.tasks_queue),
            'stream': f'/image/stream/{req.session_id}/{id(new_task)}',
            'task': id(new_task)
        }
        return JSONResponse(response, headers=NOCACHE_HEADERS)
    except ChildProcessError as e: # Render thread is dead
        raise HTTPException(status_code=500, detail=f'Rendering thread has died.') # HTTP500 Internal Server Error
    except ConnectionRefusedError as e: # Unstarted task pending, deny queueing more than one.
        raise HTTPException(status_code=503, detail=f'Session {req.session_id} has an already pending task.') # HTTP503 Service Unavailable
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/image/stream/{session_id:str}/{task_id:int}')
def stream(session_id:str, task_id:int):
    #TODO Move to WebSockets ??
    task = task_manager.get_cached_task(session_id, update_ttl=True)
    if not task: raise HTTPException(status_code=410, detail='No request received.') # HTTP410 Gone
    if (id(task) != task_id): raise HTTPException(status_code=409, detail=f'Wrong task id received. Expected:{id(task)}, Received:{task_id}') # HTTP409 Conflict
    if task.buffer_queue.empty() and not task.lock.locked():
        if task.response:
            #print(f'Session {session_id} sending cached response')
            return JSONResponse(task.response, headers=NOCACHE_HEADERS)
        raise HTTPException(status_code=425, detail='Too Early, task not started yet.') # HTTP425 Too Early
    #print(f'Session {session_id} opened live render stream {id(task.buffer_queue)}')
    return StreamingResponse(task.read_buffer_generator(), media_type='application/json')

@app.get('/image/stop')
def stop(session_id:str=None):
    if not session_id:
        if task_manager.current_state == task_manager.ServerStates.Online or task_manager.current_state == task_manager.ServerStates.Unavailable:
            raise HTTPException(status_code=409, detail='Not currently running any tasks.') # HTTP409 Conflict
        task_manager.current_state_error = StopAsyncIteration('')
        return {'OK'}
    task = task_manager.get_cached_task(session_id, update_ttl=False)
    if not task: raise HTTPException(status_code=404, detail=f'Session {session_id} has no active task.') # HTTP404 Not Found
    if isinstance(task.error, StopAsyncIteration): raise HTTPException(status_code=409, detail=f'Session {session_id} task is already stopped.') # HTTP409 Conflict
    task.error = StopAsyncIteration('')
    return {'OK'}

@app.get('/image/tmp/{session_id}/{img_id:int}')
def get_image(session_id, img_id):
    task = task_manager.get_cached_task(session_id, update_ttl=True)
    if not task: raise HTTPException(status_code=410, detail=f'Session {session_id} has not submitted a task.') # HTTP410 Gone
    if not task.temp_images[img_id]: raise HTTPException(status_code=425, detail='Too Early, task data is not available yet.') # HTTP425 Too Early
    try:
        img_data = task.temp_images[img_id]
        if isinstance(img_data, str):
            return img_data
        img_data.seek(0)
        return StreamingResponse(img_data, media_type='image/jpeg')
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/')
def read_root():
    return FileResponse(os.path.join(SD_UI_DIR, 'index.html'), headers=NOCACHE_HEADERS)

@app.on_event("shutdown")
def shutdown_event(): # Signal render thread to close on shutdown
    task_manager.current_state_error = SystemExit('Application shutting down.')

# don't log certain requests
class LogSuppressFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        path = record.getMessage()
        for prefix in ACCESS_LOG_SUPPRESS_PATH_PREFIXES:
            if path.find(prefix) != -1:
                return False
        return True
logging.getLogger('uvicorn.access').addFilter(LogSuppressFilter())

config = getConfig()

# Start the task_manager
task_manager.default_model_to_load = resolve_ckpt_to_use()
task_manager.default_vae_to_load = resolve_vae_to_use(ckpt_model_path=task_manager.default_model_to_load)
if 'render_devices' in config:  # Start a new thread for each device.
    if isinstance(config['render_devices'], str):
        config['render_devices'] = config['render_devices'].split(',')
    if not isinstance(config['render_devices'], list):
        raise Exception('Invalid render_devices value in config.')
    for device in config['render_devices']:
        if not task_manager.start_render_thread(device):
            print(device, 'failed to start.')
    if task_manager.is_alive() <= 0: # No running devices, probably invalid user config.
        print('WARNING: No active render devices after loading config. Validate "render_devices" in config.json')
        print('Loading default render devices to replace invalid render_devices field from config', config['render_devices'])

display_warning = False
if task_manager.is_alive() <= 0: # Either no defauls or no devices after loading config.
    # Select best GPU device using free memory, if more than one device.
    if task_manager.start_render_thread('auto'): # Detect best device for renders
        if task_manager.is_alive(0) <= 0:  # has no cuda:0
            if task_manager.is_alive('cpu') >= 1:  # auto used CPU.
                pass # Maybe add warning here about CPU mode...
            elif task_manager.start_render_thread('cuda'): # An other cuda device is better and cuda:0 is missing, try to start it...
                display_warning = True # And warn user to update settings...
            else:
                print('Failed to start GPU:0...')
    else:
        print('Failed to start gpu device.')
    if task_manager.is_alive('cpu') <= 0:
        # Allow CPU to be used for renders
        if not task_manager.start_render_thread('cpu'):
            print('Failed to start CPU render device...')

if display_warning or task_manager.is_alive(0) <= 0:
    print('WARNING: GFPGANer only works on GPU:0, use CUDA_VISIBLE_DEVICES if GFPGANer is needed on a specific GPU.')
    print('Using CUDA_VISIBLE_DEVICES will remap the selected devices starting at GPU:0 fixing GFPGANer')
    print('Add the line "@set CUDA_VISIBLE_DEVICES=N" where N is the GPUs to use to config.bat')
    print('Add the line "CUDA_VISIBLE_DEVICES=N" where N is the GPUs to use to config.sh')

del display_warning

# start the browser ui
import webbrowser; webbrowser.open('http://localhost:9000')