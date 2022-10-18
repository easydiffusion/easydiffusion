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
TASK_TTL = 15 * 60 # Discard last session's task timeout

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import logging
import queue, threading, time
from typing import Any, Generator, Hashable, Optional, Union

from sd_internal import Request, Response, task_manager

app = FastAPI()

modifiers_cache = None
outpath = os.path.join(os.path.expanduser("~"), OUTPUT_DIRNAME)

# don't show access log entries for URLs that start with the given prefix
ACCESS_LOG_SUPPRESS_PATH_PREFIXES = ['/ping', '/image', '/modifier-thumbnails']

NOCACHE_HEADERS={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
app.mount('/media', StaticFiles(directory=os.path.join(SD_UI_DIR, 'media/')), name="media")

class SetAppConfigRequest(BaseModel):
    update_branch: str = "main"

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

@app.on_event("shutdown")
def shutdown_event(): # Signal render thread to close on shutdown
    task_manager.current_state_error = SystemExit('Application shutting down.')

@app.get('/')
def read_root():
    return FileResponse(os.path.join(SD_UI_DIR, 'index.html'), headers=NOCACHE_HEADERS)

@app.get('/ping') # Get server and optionally session status.
def ping(session_id:str=None):
    if not task_manager.render_thread.is_alive(): # Render thread is dead.
        if task_manager.current_state_error: return HTTPException(status_code=500, detail=str(current_state_error))
        return HTTPException(status_code=500, detail='Render thread is dead.')
    if task_manager.current_state_error and not isinstance(task_manager.current_state_error, StopAsyncIteration): return HTTPException(status_code=500, detail=str(current_state_error))
    # Alive
    response = {'status': str(task_manager.current_state)}
    if session_id:
        task = task_manager.task_cache.tryGet(session_id)
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
    try:
        save_model_to_config(req.use_stable_diffusion_model)
        req.use_stable_diffusion_model = resolve_model_to_use(req.use_stable_diffusion_model)
        new_task = task_manager.render(req)
        response = {
            'status': str(task_manager.current_state), 
            'queue': task_manager.tasks_queue.qsize(),
            'stream': f'/image/stream/{req.session_id}/{id(new_task)}',
            'task': id(new_task)
        }
        return JSONResponse(response, headers=NOCACHE_HEADERS)
    except ChildProcessError as e: # Render thread is dead
        return HTTPException(status_code=500, detail=f'Rendering thread has died.') # HTTP500 Internal Server Error
    except ConnectionRefusedError as e: # Unstarted task pending, deny queueing more than one.
        return HTTPException(status_code=503, detail=f'Session {req.session_id} has an already pending task.') # HTTP503 Service Unavailable
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

@app.get('/image/stream/{session_id:str}/{task_id:int}')
def stream(session_id:str, task_id:int):
    #TODO Move to WebSockets ??
    task = task_manager.task_cache.tryGet(session_id)
    if not task: return HTTPException(status_code=410, detail='No request received.') # HTTP410 Gone
    if (id(task) != task_id): return HTTPException(status_code=409, detail=f'Wrong task id received. Expected:{id(task)}, Received:{task_id}') # HTTP409 Conflict
    if task.buffer_queue.empty() and not task.lock.locked():
        if task.response:
            #print(f'Session {session_id} sending cached response')
            return JSONResponse(task.response, headers=NOCACHE_HEADERS)
        return HTTPException(status_code=425, detail='Too Early, task not started yet.') # HTTP425 Too Early
    #print(f'Session {session_id} opened live render stream {id(task.buffer_queue)}')
    return StreamingResponse(task.read_buffer_generator(), media_type='application/json')

@app.get('/image/stop')
def stop(session_id:str=None):
    if not session_id:
        if task_manager.current_state == task_manager.ServerStates.Online or task_manager.current_state == task_manager.ServerStates.Unavailable:
            return HTTPException(status_code=409, detail='Not currently running any tasks.') # HTTP409 Conflict
        task_manager.current_state_error = StopAsyncIteration('')
        return {'OK'}
    task = task_manager.task_cache.tryGet(session_id)
    if not task: return HTTPException(status_code=404, detail=f'Session {session_id} has no active task.') # HTTP404 Not Found
    if isinstance(task.error, StopAsyncIteration): return HTTPException(status_code=409, detail=f'Session {session_id} task is already stopped.') # HTTP409 Conflict
    task.error = StopAsyncIteration('')
    return {'OK'}

@app.get('/image/tmp/{session_id}/{img_id:int}')
def get_image(session_id, img_id):
    task = task_manager.task_cache.tryGet(session_id)
    if not task: return HTTPException(status_code=410, detail=f'Session {session_id} has not submitted a task.') # HTTP410 Gone
    if not task.temp_images[img_id]: return HTTPException(status_code=425, detail='Too Early, task data is not available yet.') # HTTP425 Too Early
    try:
        img_data = task.temp_images[img_id]
        if isinstance(img_data, str):
            return img_data
        img_data.seek(0)
        return StreamingResponse(img_data, media_type='image/jpeg')
    except KeyError as e:
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

        with open(config_json_path, 'w', encoding='utf-8') as f:
            f.write(config_json_str)

        with open(config_bat_path, 'w', encoding='utf-8') as f:
            f.write(config_bat_str)

        with open(config_sh_path, 'w', encoding='utf-8') as f:
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
        with open(config_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        return default_val

def setConfig(config):
    try:
        config_json_path = os.path.join(CONFIG_DIR, 'config.json')
        with open(config_json_path, 'w', encoding='utf-8') as f:
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

@app.get('/get/{key:path}')
def read_web_data(key:str=None):
    if not key: # /get without parameters, stable-diffusion easter egg.
        return HTTPException(status_code=418, detail="StableDiffusion is drawing a teapot!") # HTTP418 I'm a teapot
    elif key == 'app_config':
        config = getConfig(default_val=None)
        if config is None:
            return HTTPException(status_code=500, detail="Config file is missing or unreadable")
        return JSONResponse(config, headers=NOCACHE_HEADERS)
    elif key == 'models':
        return JSONResponse(getModels(), headers=NOCACHE_HEADERS)
    elif key == 'modifiers': return FileResponse(os.path.join(SD_UI_DIR, 'modifiers.json'), headers=NOCACHE_HEADERS)
    elif key == 'output_dir': return JSONResponse({ 'output_dir': outpath }, headers=NOCACHE_HEADERS)
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

task_manager.default_model_to_load = get_initial_model_to_load()
task_manager.start_render_thread()

# start the browser ui
import webbrowser; webbrowser.open('http://localhost:9000')