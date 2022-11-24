"""server.py: FastAPI SD-UI Web Host.
Notes:
    async endpoints always run on the main thread. Without they run on the thread pool.
"""
import json
import traceback

import sys
import os
import re
import picklescan.scanner
import rich

SD_DIR = os.getcwd()
print('started in ', SD_DIR)

SD_UI_DIR = os.getenv('SD_UI_PATH', None)
sys.path.append(os.path.dirname(SD_UI_DIR))

CONFIG_DIR = os.path.abspath(os.path.join(SD_UI_DIR, '..', 'scripts'))
MODELS_DIR = os.path.abspath(os.path.join(SD_DIR, '..', 'models'))

USER_UI_PLUGINS_DIR = os.path.abspath(os.path.join(SD_DIR, '..', 'plugins', 'ui'))
CORE_UI_PLUGINS_DIR = os.path.abspath(os.path.join(SD_UI_DIR, 'plugins', 'ui'))
UI_PLUGINS_SOURCES = ((CORE_UI_PLUGINS_DIR, 'core'), (USER_UI_PLUGINS_DIR, 'user'))

STABLE_DIFFUSION_MODEL_EXTENSIONS = ['.ckpt']
VAE_MODEL_EXTENSIONS = ['.vae.pt', '.ckpt']

OUTPUT_DIRNAME = "Stable Diffusion UI" # in the user's home folder
TASK_TTL = 15 * 60 # Discard last session's task timeout
APP_CONFIG_DEFAULTS = {
    # auto: selects the cuda device with the most free memory, cuda: use the currently active cuda device.
    'render_devices': 'auto', # valid entries: 'auto', 'cpu' or 'cuda:N' (where N is a GPU index)
    'update_branch': 'main',
    'ui': {
        'open_browser_on_start': True,
    },
}
APP_CONFIG_DEFAULT_MODELS = [
    # needed to support the legacy installations
    'custom-model', # Check if user has a custom model, use it first.
    'sd-v1-4', # Default fallback.
]

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
import aiohttp
import asyncio
from aiofile import async_open
from time import time
from pydantic import BaseModel
import logging
#import queue, threading, time
from typing import Any, Generator, Hashable, List, Optional, Union

from sd_internal import Request, Response, task_manager

app = FastAPI()

modifiers_cache = None
outpath = os.path.join(os.path.expanduser("~"), OUTPUT_DIRNAME)

os.makedirs(USER_UI_PLUGINS_DIR, exist_ok=True)

# don't show access log entries for URLs that start with the given prefix
ACCESS_LOG_SUPPRESS_PATH_PREFIXES = ['/ping', '/image', '/modifier-thumbnails']

NOCACHE_HEADERS={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}

class NoCacheStaticFiles(StaticFiles):
    def is_not_modified(self, response_headers, request_headers) -> bool:
        if 'content-type' in response_headers and ('javascript' in response_headers['content-type'] or 'css' in response_headers['content-type']):
            response_headers.update(NOCACHE_HEADERS)
            return False

        return super().is_not_modified(response_headers, request_headers)

app.mount('/media', NoCacheStaticFiles(directory=os.path.join(SD_UI_DIR, 'media')), name="media")

for plugins_dir, dir_prefix in UI_PLUGINS_SOURCES:
    app.mount(f'/plugins/{dir_prefix}', NoCacheStaticFiles(directory=plugins_dir), name=f"plugins-{dir_prefix}")

def getConfig(default_val=APP_CONFIG_DEFAULTS):
    try:
        config_json_path = os.path.join(CONFIG_DIR, 'config.json')
        if not os.path.exists(config_json_path):
            return default_val
        with open(config_json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            if 'net' not in config:
                config['net'] = {}
            if os.getenv('SD_UI_BIND_PORT') is not None:
                config['net']['listen_port'] = int(os.getenv('SD_UI_BIND_PORT'))
            if os.getenv('SD_UI_BIND_IP') is not None:
                config['net']['listen_to_network'] = ( os.getenv('SD_UI_BIND_IP') == '0.0.0.0' )
            return config
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        return default_val

def setConfig(config):
    print( json.dumps(config) )
    try: # config.json
        config_json_path = os.path.join(CONFIG_DIR, 'config.json')
        with open(config_json_path, 'w', encoding='utf-8') as f:
            json.dump(config, f)
    except:
        print(traceback.format_exc())

    try: # config.bat
        config_bat_path = os.path.join(CONFIG_DIR, 'config.bat')
        config_bat = []

        if 'update_branch' in config:
            config_bat.append(f"@set update_branch={config['update_branch']}")

        config_bat.append(f"@set SD_UI_BIND_PORT={config['net']['listen_port']}")
        bind_ip = '0.0.0.0' if config['net']['listen_to_network'] else '127.0.0.1'
        config_bat.append(f"@set SD_UI_BIND_IP={bind_ip}")

        if len(config_bat) > 0:
            with open(config_bat_path, 'w', encoding='utf-8') as f:
                f.write('\r\n'.join(config_bat))
    except:
        print(traceback.format_exc())

    try: # config.sh
        config_sh_path = os.path.join(CONFIG_DIR, 'config.sh')
        config_sh = ['#!/bin/bash']

        if 'update_branch' in config:
            config_sh.append(f"export update_branch={config['update_branch']}")

        config_sh.append(f"export SD_UI_BIND_PORT={config['net']['listen_port']}")
        bind_ip = '0.0.0.0' if config['net']['listen_to_network'] else '127.0.0.1'
        config_sh.append(f"export SD_UI_BIND_IP={bind_ip}")

        if len(config_sh) > 1:
            with open(config_sh_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(config_sh))
    except:
        print(traceback.format_exc())

def resolve_model_to_use(model_name:str, model_type:str, model_dir:str, model_extensions:list, default_models=[]):
    model_dirs = [os.path.join(MODELS_DIR, model_dir), SD_DIR]
    if not model_name: # When None try user configured model.
        config = getConfig()
        if 'model' in config and model_type in config['model']:
            model_name = config['model'][model_type]
    if model_name:
        # Check models directory
        models_dir_path = os.path.join(MODELS_DIR, model_dir, model_name)
        for model_extension in model_extensions:
            if os.path.exists(models_dir_path + model_extension):
                return models_dir_path
            if os.path.exists(model_name + model_extension):
                # Direct Path to file
                model_name = os.path.abspath(model_name)
                return model_name
    # Default locations
    if model_name in default_models:
        default_model_path = os.path.join(SD_DIR, model_name)
        for model_extension in model_extensions:
            if os.path.exists(default_model_path + model_extension):
                return default_model_path
    # Can't find requested model, check the default paths.
    for default_model in default_models:
        for model_dir in model_dirs:
            default_model_path = os.path.join(model_dir, default_model)
            for model_extension in model_extensions:
                if os.path.exists(default_model_path + model_extension):
                    if model_name is not None:
                        print(f'Could not find the configured custom model {model_name}{model_extension}. Using the default one: {default_model_path}{model_extension}')
                    return default_model_path
    raise Exception('No valid models found.')

def resolve_ckpt_to_use(model_name:str=None):
    return resolve_model_to_use(model_name, model_type='stable-diffusion', model_dir='stable-diffusion', model_extensions=STABLE_DIFFUSION_MODEL_EXTENSIONS, default_models=APP_CONFIG_DEFAULT_MODELS)

def resolve_vae_to_use(model_name:str=None):
    try:
        return resolve_model_to_use(model_name, model_type='vae', model_dir='vae', model_extensions=VAE_MODEL_EXTENSIONS, default_models=[])
    except:
        return None

class SetAppConfigRequest(BaseModel):
    update_branch: str = None
    render_devices: Union[List[str], List[int], str, int] = None
    model_vae: str = None
    ui_open_browser_on_start: bool = None
    listen_to_network: bool = None
    listen_port: int = None

@app.post('/app_config')
async def setAppConfig(req : SetAppConfigRequest):
    config = getConfig()
    if req.update_branch is not None:
        config['update_branch'] = req.update_branch
    if req.render_devices is not None:
        update_render_devices_in_config(config, req.render_devices)
    if req.ui_open_browser_on_start is not None:
        if 'ui' not in config:
            config['ui'] = {}
        config['ui']['open_browser_on_start'] = req.ui_open_browser_on_start
    if req.listen_to_network is not None:
       if 'net' not in config:
           config['net'] = {}
       config['net']['listen_to_network'] = bool(req.listen_to_network)
    if req.listen_port is not None:
       if 'net' not in config:
           config['net'] = {}
       config['net']['listen_port'] = int(req.listen_port)
    try:
        setConfig(config)

        if req.render_devices:
            update_render_threads()

        return JSONResponse({'status': 'OK'}, headers=NOCACHE_HEADERS)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


class ModelDownloadRequest(BaseModel):
    url: str = None
    type: str = 'stable-diffusion'

async def downloadFile(url, folder, filename="", offset=0):
    afp = None
    total_size = 0
    start = time()
    last_msg = start
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=None, headers={"Range": f"bytes={str(offset)}-"}) as resp:
	        # First chunk, so there's no output file opened yet. First determine the filename, than open the file
                if afp == None:
                    if filename == "" and "Content-Disposition" in resp.headers:
                        m = re.match('filename="([^"]*)"', resp.headers["Content-Disposition"])
                        if m:
                            filename = m.group(1)
                    if filename == "":
                        filename = url.split("/")[-1]
                    mode = "wb" if offset == 0 else "ab"
                    afp = await async_open(os.path.join( folder, filename), mode)
                while True:
                    chunk = await resp.content.read(1024*1024)
                    if not chunk:
                        rich.print("[orange]not chunk[/orange]")
                        break
                    await afp.write(chunk)
                    total_size += len(chunk)
                    if time() - last_msg > 2:
                        rich.print(f'[cyan]{time() - start:0.2f}s, downloaded: {total_size / (1024 * 1024):0.0f}MB of {int(resp.headers["content-length"])/1024/1024:0.0f}MB[/cyan]')
                        last_msg = time()
    except Exception as e:
        rich.print(f"[yellow]{traceback.format_exc()}[/yellow]")
        rich.print(f"[yellow]Exception: {e}[/yellow]")
    if afp != None:
        await afp.close()
    if total_size != 0 and total_size != int(resp.headers["content-length"]):
        # Resume the request
        await asyncio.sleep(2)
        await downloadFile(url, folder, filename, total_size)


async def modelDownloadTask(req: ModelDownloadRequest):
    models_dir = os.path.join(MODELS_DIR, req.type)
    await downloadFile(req.url, models_dir)

@app.post('/model/download')
async def modelDownload(req: ModelDownloadRequest, background_tasks: BackgroundTasks):
    rich.print("[cyan]Download Model %s[/cyan]" % (req.url))
    if req.type not in ['stable-diffusion', 'vae']:
        return JSONResponse({'status': 'Error', 'message': 'Invalid type'}, headers=NOCACHE_HEADERS)
    background_tasks.add_task(modelDownloadTask, req)
    return JSONResponse({'status': 'OK'}, headers=NOCACHE_HEADERS)

def is_malicious_model(file_path):
    try:
        scan_result = picklescan.scanner.scan_file_path(file_path)
        if scan_result.issues_count > 0 or scan_result.infected_files > 0:
            rich.print(":warning: [bold red]Scan %s: %d scanned, %d issue, %d infected.[/bold red]" % (file_path, scan_result.scanned_files, scan_result.issues_count, scan_result.infected_files))
            return True
        else:
            rich.print("Scan %s: [green]%d scanned, %d issue, %d infected.[/green]" % (file_path, scan_result.scanned_files, scan_result.issues_count, scan_result.infected_files))
            return False
    except Exception as e:
        print('error while scanning', file_path, 'error:', e)

    return False

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

    def listModels(models_dirname, model_type, model_extensions):
        models_dir = os.path.join(MODELS_DIR, models_dirname)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        for file in os.listdir(models_dir):
            for model_extension in model_extensions:
                if not file.endswith(model_extension):
                    continue

                if is_malicious_model(os.path.join(models_dir, file)):
                    models['scan-error'] = file
                    return

                model_name = file[:-len(model_extension)]
                models['options'][model_type].append(model_name)

        models['options'][model_type] = [*set(models['options'][model_type])] # remove duplicates
        models['options'][model_type].sort()

    # custom models
    listModels(models_dirname='stable-diffusion', model_type='stable-diffusion', model_extensions=STABLE_DIFFUSION_MODEL_EXTENSIONS)
    listModels(models_dirname='vae', model_type='vae', model_extensions=VAE_MODEL_EXTENSIONS)

    # legacy
    custom_weight_path = os.path.join(SD_DIR, 'custom-model.ckpt')
    if os.path.exists(custom_weight_path):
        models['options']['stable-diffusion'].append('custom-model')

    return models

def getUIPlugins():
    plugins = []

    for plugins_dir, dir_prefix in UI_PLUGINS_SOURCES:
        for file in os.listdir(plugins_dir):
            if file.endswith('.plugin.js'):
                plugins.append(f'/plugins/{dir_prefix}/{file}')

    return plugins

@app.get('/get/{key:path}')
def read_web_data(key:str=None):
    if not key: # /get without parameters, stable-diffusion easter egg.
        raise HTTPException(status_code=418, detail="StableDiffusion is drawing a teapot!") # HTTP418 I'm a teapot
    elif key == 'app_config':
        config = getConfig(default_val=None)
        if config is None:
            config = APP_CONFIG_DEFAULTS
        return JSONResponse(config, headers=NOCACHE_HEADERS)
    elif key == 'devices':
        config = getConfig()
        devices = task_manager.get_devices()
        devices['config'] = config.get('render_devices', "auto")
        return JSONResponse(devices, headers=NOCACHE_HEADERS)
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
    response['devices'] = task_manager.get_devices()
    return JSONResponse(response, headers=NOCACHE_HEADERS)

def save_model_to_config(ckpt_model_name, vae_model_name):
    config = getConfig()
    if 'model' not in config:
        config['model'] = {}

    config['model']['stable-diffusion'] = ckpt_model_name
    config['model']['vae'] = vae_model_name

    if vae_model_name is None or vae_model_name == "":
        del config['model']['vae']

    setConfig(config)

def update_render_devices_in_config(config, render_devices):
    if render_devices not in ('cpu', 'auto') and not render_devices.startswith('cuda:'):
        raise HTTPException(status_code=400, detail=f'Invalid render device requested: {render_devices}')

    if render_devices.startswith('cuda:'):
        render_devices = render_devices.split(',')

    config['render_devices'] = render_devices

@app.post('/render')
def render(req : task_manager.ImageRequest):
    try:
        save_model_to_config(req.use_stable_diffusion_model, req.use_vae_model)
        req.use_stable_diffusion_model = resolve_ckpt_to_use(req.use_stable_diffusion_model)
        req.use_vae_model = resolve_vae_to_use(req.use_vae_model)
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

# Start the task_manager
task_manager.default_model_to_load = resolve_ckpt_to_use()
task_manager.default_vae_to_load = resolve_vae_to_use()

def update_render_threads():
    config = getConfig()
    render_devices = config.get('render_devices', 'auto')
    active_devices = task_manager.get_devices()['active'].keys()

    print('requesting for render_devices', render_devices)
    task_manager.update_render_threads(render_devices, active_devices)

update_render_threads()

# start the browser ui
def open_browser():
    config = getConfig()
    ui = config.get('ui', {})
    net = config.get('net', {'listen_port':9000})
    port = net.get('listen_port', 9000)
    if ui.get('open_browser_on_start', True):
        import webbrowser; webbrowser.open(f"http://localhost:{port}")

open_browser()
